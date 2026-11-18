"""实测扫描校准。

把单块纸白整体增益增强为基于多张同设置扫描色卡的可复用校准。模型显式区分两件事：

* 扫描颜色响应 f：设备把入射线性光映射到线性 RGB 读数的逐通道多项式（与位置无关）；
* 画面照明不匀 g(x, y)：低阶、平滑的乘法空间场，只能吸收缓慢的灯光变化，
  不会抹平纸张本身的深浅变化（纸张纹理不是平滑场，保留在校准后图像里）。

候选方案固定为两套，由用户在证据页面前选择：

``simple``   逐通道多项式响应，无空间项（较简单、受数据支持时优先）；
``spatial``  同一响应模型加上双线性/双二次平滑照明场（较灵活，仅在独立整组
             留出误差确实改善时才允许推荐）。

验证只按整组留出：以（扫描, 色卡）为分组单位，整次扫描或整块色卡整体进入
训练或验证，同一色块的相邻像素永远不会被拆到训练和验证两边。
"""
import csv
import io
import math

import cv2
import numpy as np

from .pixels import integer, luminance, number, rgb_image, srgb_to_linear, linear_to_srgb

# 拟合在 sRGB 线性化空间进行；工作空间名称只做记录，不做 ICC 转换。
WORKSPACE = 'srgb_linear_numeric'
LUT_SIZE = 256
# 归一化坐标相对训练观测范围外推时，空间场钳制到观测边界，避免多项式外推爆炸。
FIELD_MARGIN = 0.02
# 可靠范围判定时允许的训练覆盖外裕量（位置/亮度分别设置）。
POSITION_MARGIN = 0.05
TONE_MARGIN = 0.08
# 岭回归强度（相对设计矩阵列尺度），数值稳定用，不改变模型族。
RESPONSE_RIDGE = 1e-3
FIELD_RIDGE = 0.1
# 照明场增益允许偏离1的最大幅度（钳制外推爆炸，正常平滑场远小于此）。
FIELD_GAIN_LIMIT = 0.25
# 留出组少于该数量时无法给出独立证据。
MIN_GROUPS_FOR_EVIDENCE = 2
LUMINANCE_BINS = [(0.0, 0.15), (0.15, 0.35), (0.35, 0.6), (0.6, 0.85), (0.85, 1.01)]
POSITION_ZONES = ['center', 'edge', 'corner']


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def linearize(rgb):
    array = np.asarray(rgb, dtype=np.float32)
    if array.ndim == 2 and array.shape[1] == 3:
        return srgb_to_linear(array.reshape(1, -1, 3)).reshape(-1, 3).astype(np.float64)
    return srgb_to_linear(rgb_image(rgb)).astype(np.float64)


def encode(linear_rgb):
    return linear_to_srgb(np.clip(linear_rgb, 0, 1))


def patch_delta_e76(rgb_a, rgb_b):
    """两组 (n,3) sRGB 色块之间的逐行 CIE76 近似距离。"""
    a = np.clip(np.asarray(rgb_a, dtype=np.float64), 0, 1).reshape(-1, 1, 3).astype(np.float32)
    b = np.clip(np.asarray(rgb_b, dtype=np.float64), 0, 1).reshape(-1, 1, 3).astype(np.float32)
    lab_a = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    lab_b = cv2.cvtColor(b, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    return np.linalg.norm(lab_a - lab_b, axis=1)


def _validate_rgb(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(label + '需要三个RGB分量')
    return [float(number(v, label, 0, 1)) for v in value]


def _rectangle_values(image, geometry, inset_ratio):
    height, width = image.shape[:2]
    x = integer(geometry.get('x'), '色块x', 0, width - 1)
    y = integer(geometry.get('y'), '色块y', 0, height - 1)
    w = integer(geometry.get('width'), '色块宽度', 1, width - x)
    h = integer(geometry.get('height'), '色块高度', 1, height - y)
    ratio = number(inset_ratio, '色块内缩比例', 0, 0.45)
    dx = math.floor(w * ratio)
    dy = math.floor(h * ratio)
    x0, x1 = x + dx, x + w - dx
    y0, y1 = y + dy, h + y - dy
    values = image[y0:y1, x0:x1].reshape(-1, 3)
    if len(values) < 4:
        raise ValueError('色块内缩后至少需要4个像素')
    return values, {'x': x0, 'y': y0, 'width': x1 - x0, 'height': y1 - y0}


def auto_quality(values_linear):
    """根据读数统计自动判定 saturated / dirty / ok。

    saturated：任一通道贴近设备端点（0 或 1 附近），读数已被裁切，不能约束曲线；
    dirty：色块内部离散度过大（污损、划痕、错位到边缘），中位数代表性不足。
    """
    floor = np.count_nonzero(values_linear <= 1 / 255)
    ceiling = np.count_nonzero(values_linear >= 254 / 255)
    clipped_fraction = (floor + ceiling) / values_linear.shape[0]
    center = np.median(values_linear, axis=0)
    deviations = np.linalg.norm(values_linear - center, axis=1)
    p95 = float(np.percentile(deviations, 95))
    std = values_linear.std(axis=0)
    if clipped_fraction >= 0.05:
        return 'saturated', {'clipped_fraction': float(clipped_fraction), 'p95_deviation': p95, 'rgb_std': std.tolist()}
    if p95 > 0.08:
        return 'dirty', {'clipped_fraction': float(clipped_fraction), 'p95_deviation': p95, 'rgb_std': std.tolist()}
    return 'ok', {'clipped_fraction': float(clipped_fraction), 'p95_deviation': p95, 'rgb_std': std.tolist()}


# ---------------------------------------------------------------------------
# 色块采样与会话数据
# ---------------------------------------------------------------------------

def sample_patch(image_rgb, geometry, reference_rgb, inset=0.15, manual_quality=None):
    """对单个色块采样，返回包含实测、参考、位置与质量标记的记录。"""
    image = rgb_image(image_rgb)
    values_srgb, box = _rectangle_values(image, geometry, inset)
    values = linearize(values_srgb)
    measured = np.median(values, axis=0)
    reference = np.asarray(_validate_rgb(reference_rgb, '参考颜色'), dtype=np.float64)
    reference_linear = srgb_to_linear(reference.astype(np.float32)).astype(np.float64)
    quality, stats = auto_quality(values)
    auto_quality_label = quality
    if manual_quality is not None:
        if manual_quality not in ('ok', 'saturated', 'dirty', 'excluded'):
            raise ValueError('质量标记只支持 ok、saturated、dirty、excluded')
        quality = manual_quality
    included = quality != 'excluded'
    height, width = image.shape[:2]
    return {
        'geometry': {'type': 'rectangle', 'x': int(geometry['x']), 'y': int(geometry['y']),
                     'width': int(geometry['width']), 'height': int(geometry['height'])},
        'sample_box': box,
        'pixels': int(len(values)),
        'measured_rgb': measured.tolist(),
        'measured_srgb': encode(measured).tolist(),
        'reference_rgb': reference.tolist(),
        'reference_linear': reference_linear.tolist(),
        'nx': (box['x'] + box['width'] / 2) / max(width - 1, 1),
        'ny': (box['y'] + box['height'] / 2) / max(height - 1, 1),
        'quality': quality,
        'auto_quality': auto_quality_label,
        'quality_stats': stats,
        'included': bool(included),
    }


def parse_patch_csv(text):
    """色块 CSV：name,x,y,width,height,ref_r,ref_g,ref_b[,card,quality]。"""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('请提供色块CSV内容')
    if len(text.encode('utf-8')) > 1024 * 1024:
        raise ValueError('色块CSV最多1 MiB')
    reader = csv.DictReader(io.StringIO(text.lstrip('﻿')))
    required = {'name', 'x', 'y', 'width', 'height', 'ref_r', 'ref_g', 'ref_b'}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError('CSV需要 name,x,y,width,height,ref_r,ref_g,ref_b 列；可选 card,quality')
    patches = []
    names = set()
    for line, record in enumerate(reader, start=2):
        if not any(str(v or '').strip() for v in record.values()):
            continue
        if len(patches) >= 1024:
            raise ValueError('单次最多支持1024个色块')
        name = str(record.get('name') or '').strip()
        if not name or len(name) > 80:
            raise ValueError(f'第{line}行色块名称需要1到80字符')
        if name in names:
            raise ValueError('色块名称不能重复：' + name)
        names.add(name)
        quality = (record.get('quality') or '').strip() or None
        if quality and quality not in ('ok', 'saturated', 'dirty', 'excluded'):
            raise ValueError(f'第{line}行 quality 只支持 ok/saturated/dirty/excluded')
        patches.append({
            'name': name,
            'geometry': {'type': 'rectangle', 'x': int(record['x']), 'y': int(record['y']),
                         'width': int(record['width']), 'height': int(record['height'])},
            'reference_rgb': [float(record['ref_r']), float(record['ref_g']), float(record['ref_b'])],
            'card_id': (record.get('card') or '').strip() or 'card_1',
            'quality': quality,
        })
    if not patches:
        raise ValueError('CSV中没有有效色块')
    return patches


def collect_readings(scans):
    """把会话里的色块记录整理成拟合用读数表，只纳入 included 的色块。"""
    readings = []
    for scan in scans:
        scan_id = scan['scan_id']
        for card in scan.get('cards', []):
            card_id = card['card_id']
            for patch in card['patches']:
                if not patch.get('included', True):
                    continue
                readings.append({
                    'scan_id': scan_id,
                    'card_id': card_id,
                    'patch_name': patch['name'],
                    'group_id': scan_id + '|' + card_id,
                    'measured': np.asarray(patch['measured_rgb'], dtype=np.float64),
                    'reference': np.asarray(patch['reference_linear'], dtype=np.float64),
                    'nx': float(patch['nx']),
                    'ny': float(patch['ny']),
                    'quality': patch['quality'],
                })
    if not readings:
        raise ValueError('没有纳入拟合的色块；请先圈定色块并把至少一个色块标记为纳入')
    return readings


# ---------------------------------------------------------------------------
# 设计矩阵与岭回归
# ---------------------------------------------------------------------------

def response_design(values, degree):
    """逐通道多项式基 [1, v, v^2, ...]，values 形状 (n, 3)，返回 (3, n, k)。"""
    k = degree + 1
    planes = np.empty((3, values.shape[0], k), dtype=np.float64)
    for channel in range(3):
        v = values[:, channel]
        column = np.empty((values.shape[0], k), dtype=np.float64)
        column[:, 0] = 1.0
        for power in range(1, k):
            column[:, power] = v ** power
        planes[channel] = column
    return planes


def _ridge_solve(design, target, ridge):
    scale = np.sqrt(np.mean(design ** 2, axis=0))
    scale[scale < 1e-12] = 1.0
    normalized = design / scale
    gram = normalized.T @ normalized
    # 岭强度按样本数缩放，使其表示相对总信息量的惩罚，避免小样本下系数爆炸。
    gram += ridge * design.shape[0] * np.eye(design.shape[1])
    rhs = normalized.T @ target
    coefficients = np.linalg.solve(gram, rhs) / scale
    return coefficients


def _fit_response(measured, reference, degree, ridge=RESPONSE_RIDGE):
    coefficients = np.zeros((3, degree + 1), dtype=np.float64)
    for channel in range(3):
        design = response_design(measured, degree)[channel]
        coefficients[channel] = _ridge_solve(design, reference[:, channel], ridge)
    return coefficients


def response_lut(coefficients, size=LUT_SIZE):
    """单调不减的响应查找表：多项式可能在暗部下冲，用累积钳制保证物理单调性。"""
    grid = np.linspace(0, 1, size)
    lut = np.empty((size, 3), dtype=np.float64)
    for channel in range(3):
        powers = np.vstack([grid ** p for p in range(coefficients.shape[1])]).T
        raw = powers @ coefficients[channel]
        monotone = np.maximum.accumulate(np.maximum(raw, 0.0))
        lut[:, channel] = np.clip(monotone, 0, 1)
    return lut


def field_design(nx, ny, field_degree):
    """照明场基，坐标中心化到 [-0.5, 0.5] 以改善数值条件。

    degree 1：常数+中心化 x/y/xy（4项）；degree 2：再加 x²/y²（6项）。
    常数项由调用方锚定，场始终写成 1 + 中心化基 @ beta。
    """
    nx = np.asarray(nx, dtype=np.float64) - 0.5
    ny = np.asarray(ny, dtype=np.float64) - 0.5
    if field_degree == 1:
        return np.column_stack([np.ones_like(nx), nx, ny, nx * ny])
    if field_degree == 2:
        return np.column_stack([np.ones_like(nx), nx, ny, nx * ny, nx ** 2, ny ** 2])
    raise ValueError('照明场阶数只支持1（双线性）或2（双二次）')


def _fit_field(measured, reference, response_coefficients, nx, ny, field_degree, ridge=FIELD_RIDGE):
    """拟合乘法照明场 g = 1 + Z @ beta，常数项锚定为1。

    锚定保证整体颜色增益由响应多项式承担，空间场只解释*相对*的灯光不匀，
    避免两个自由度互相吸收导致整扫描留出劣化。beta 按通道独立岭回归，
    并把增益钳制在物理合理范围，防止小样本下外推爆炸。
    """
    lut = response_lut(response_coefficients)
    corrected = np.empty_like(measured)
    for channel in range(3):
        index = np.clip(np.rint(measured[:, channel] * (LUT_SIZE - 1)).astype(int), 0, LUT_SIZE - 1)
        corrected[:, channel] = lut[index, channel]
    full = field_design(nx, ny, field_degree)
    centered = full[:, 1:]
    field = np.zeros((3, full.shape[1]), dtype=np.float64)
    field[:, 0] = 1.0
    for channel in range(3):
        ratio = reference[:, channel] / np.maximum(corrected[:, channel], 1e-6)
        beta = _ridge_solve(centered, ratio - 1.0, ridge)
        # 钳制 beta，使训练位置上的增益落在 [1-limit, 1+limit]。
        gain_deviation = centered @ beta
        peak = float(np.max(np.abs(gain_deviation)))
        if peak > FIELD_GAIN_LIMIT:
            beta *= FIELD_GAIN_LIMIT / peak
        field[channel, 1:] = beta
    return field


# ---------------------------------------------------------------------------
# 两套候选模型
# ---------------------------------------------------------------------------

def build_simple(readings, degree=2):
    """较简单方案：逐通道多项式响应，无空间项。"""
    degree = integer(degree, '响应多项式阶数', 1, 3, 2)
    measured = np.vstack([r['measured'] for r in readings])
    reference = np.vstack([r['reference'] for r in readings])
    if len(readings) < degree + 2:
        raise ValueError(f'较简单方案至少需要{degree + 2}个纳入色块')
    coefficients = _fit_response(measured, reference, degree)
    return _model_package('simple', degree, coefficients, None, readings)


def build_spatial(readings, degree=2, field_degree=1, iterations=12, tolerance=1e-6):
    """较灵活方案：响应多项式 + 平滑乘法照明场，交替最小二乘。

    模型（线性空间）： corrected = f(measured) * g(nx, ny)。
    场 g 是低阶平滑乘法场，只吸收缓慢灯光变化；纸张本身的深浅纹理不是
    平滑场，不会被这一项抹平。
    """
    degree = integer(degree, '响应多项式阶数', 1, 3, 2)
    field_degree = integer(field_degree, '照明场阶数', 1, 2, 1)
    if len(readings) < max(degree + 2, 8):
        raise ValueError('含照明场的灵活方案至少需要8个纳入色块，且要覆盖多个画面位置')
    measured = np.vstack([r['measured'] for r in readings])
    reference = np.vstack([r['reference'] for r in readings])
    nx = np.asarray([r['nx'] for r in readings], dtype=np.float64)
    ny = np.asarray([r['ny'] for r in readings], dtype=np.float64)
    coefficients = _fit_response(measured, reference, degree)
    field = _fit_field(measured, reference, coefficients, nx, ny, field_degree)
    previous = None
    package = None
    for _ in range(max(2, iterations)):
        # 给定场 g：用 reference / g 作为响应目标重新拟合 f
        design = field_design(nx, ny, field_degree)
        gain = np.column_stack([design @ field[c] for c in range(3)])
        response_target = np.clip(reference / np.maximum(gain, 1e-6), 0, None)
        coefficients = _fit_response(measured, response_target, degree)
        # 给定响应 f：用 reference / f(measured) 作为场目标重新拟合 g
        field = _fit_field(measured, reference, coefficients, nx, ny, field_degree)
        package = _model_package('spatial', degree, coefficients, field, readings, build=False)
        residual = float(np.mean((np.clip(predict_readings(readings, package), 0, 1) - reference) ** 2))
        if previous is not None and abs(previous - residual) < tolerance:
            break
        previous = residual
    package['training_fit'] = fit_metrics(package, readings)
    return package


def _model_package(kind, degree, coefficients, field, readings, build=True):
    package = {
        'kind': kind,
        'workspace': WORKSPACE,
        'degree': int(degree),
        'response_coefficients': np.asarray(coefficients, dtype=np.float64).tolist(),
        'field_coefficients': None if field is None else np.asarray(field, dtype=np.float64).tolist(),
        'lut': response_lut(np.asarray(coefficients, dtype=np.float64)).tolist(),
        'training': {
            'readings': len(readings),
            'scans': sorted({r['scan_id'] for r in readings}),
            'cards': sorted({r['group_id'] for r in readings}),
            'position_bounds': {
                'nx_min': float(min(r['nx'] for r in readings)), 'nx_max': float(max(r['nx'] for r in readings)),
                'ny_min': float(min(r['ny'] for r in readings)), 'ny_max': float(max(r['ny'] for r in readings)),
            },
            'luminance_range': [float(min(float(np.mean(r['measured'])) for r in readings)),
                                float(max(float(np.mean(r['measured'])) for r in readings))],
            'measured_points': np.vstack([r['measured'] for r in readings]).tolist(),
        },
    }
    if build:
        package['training_fit'] = fit_metrics(package, readings)
    return package


def predict_readings(readings, model):
    """与图像应用路径一致：响应走单调 LUT，空间场再做乘法修正。"""
    coefficients = np.asarray(model['response_coefficients'], dtype=np.float64)
    measured = np.vstack([r['measured'] for r in readings])
    lut = response_lut(coefficients)
    corrected = np.empty_like(measured)
    for channel in range(3):
        index = np.clip(np.rint(measured[:, channel] * (LUT_SIZE - 1)).astype(int), 0, LUT_SIZE - 1)
        corrected[:, channel] = lut[index, channel]
    if model['kind'] == 'spatial' and model.get('field_coefficients') is not None:
        field = np.asarray(model['field_coefficients'], dtype=np.float64)
        degree = 1 if field.shape[1] == 4 else 2
        design = field_design([r['nx'] for r in readings], [r['ny'] for r in readings], degree)
        gain = np.column_stack([design @ field[c] for c in range(3)])
        corrected = corrected * gain
    return corrected


def fit_metrics(model, readings):
    reference = np.vstack([r['reference'] for r in readings])
    predicted = np.clip(predict_readings(readings, model), 0, 1)
    reference_srgb = encode(reference)
    predicted_srgb = encode(predicted)
    return {
        'readings': len(readings),
        'rgb_rmse': float(np.sqrt(np.mean((predicted_srgb - reference_srgb) ** 2))),
        'mean_delta_e76': float(np.mean(patch_delta_e76(predicted_srgb, reference_srgb))),
        'p95_delta_e76': float(np.percentile(patch_delta_e76(predicted_srgb, reference_srgb), 95)),
        'max_delta_e76': float(np.max(patch_delta_e76(predicted_srgb, reference_srgb))),
    }


# ---------------------------------------------------------------------------
# 分组留出交叉验证（整次扫描 / 整块色卡）
# ---------------------------------------------------------------------------

def _error_rows(predicted, reference):
    predicted_srgb = encode(np.clip(predicted, 0, 1))
    reference_srgb = encode(reference)
    de = patch_delta_e76(predicted_srgb, reference_srgb)
    per_channel = np.abs(predicted_srgb - reference_srgb)
    return predicted_srgb, reference_srgb, de, per_channel


def _luminance_bin(value):
    light = float(luminance(value.reshape(1, 1, 3))[0, 0])
    for index, (low, high) in enumerate(LUMINANCE_BINS):
        if low <= light < high:
            return index
    return len(LUMINANCE_BINS) - 1


def _position_zone(nx, ny):
    edge_x = not (0.25 <= nx <= 0.75)
    edge_y = not (0.25 <= ny <= 0.75)
    if edge_x and edge_y:
        return 'corner'
    if edge_x or edge_y:
        return 'edge'
    return 'center'


def stratified_metrics(holdout, predicted):
    """按亮度区间与画面位置给出误差；没有独立读数的区间明确标记 unjudged。"""
    reference = np.vstack([r['reference'] for r in holdout])
    predicted_srgb, reference_srgb, de, per_channel = _error_rows(predicted, reference)
    strata = []
    for index, (low, high) in enumerate(LUMINANCE_BINS):
        selector = np.asarray([_luminance_bin(encode(r['reference'].reshape(1, 1, 3))) == index for r in holdout])
        strata.append(_stratum_row('luminance', f'{low:.2f}-{high:.2f}', selector, de, per_channel))
    for zone in POSITION_ZONES:
        selector = np.asarray([_position_zone(r['nx'], r['ny']) == zone for r in holdout])
        strata.append(_stratum_row('position', zone, selector, de, per_channel))
    return strata


def _stratum_row(dimension, label, selector, de, per_channel):
    count = int(selector.sum())
    if count == 0:
        return {'dimension': dimension, 'stratum': label, 'holdout_patches': 0, 'status': 'unjudged',
                'mean_delta_e76': None, 'p95_delta_e76': None, 'mean_abs_rgb': None,
                'note': '该区间没有独立留出参考，不能判断，禁止按整图可靠外推'}
    return {
        'dimension': dimension, 'stratum': label, 'holdout_patches': count, 'status': 'judged',
        'mean_delta_e76': float(np.mean(de[selector])),
        'p95_delta_e76': float(np.percentile(de[selector], 95)),
        'mean_abs_rgb': float(np.mean(per_channel[selector])),
        'note': '独立整组留出证据',
    }


def cross_validation(readings, builder, holdout='scan'):
    """整组留出。holdout='scan' 留整次扫描；'card' 留整块（扫描,色卡）。"""
    if holdout not in ('scan', 'card'):
        raise ValueError('留出方式只支持 scan（整次扫描）或 card（整块色卡）')
    groups = sorted({r['scan_id'] for r in readings}) if holdout == 'scan' else sorted({r['group_id'] for r in readings})
    if len(groups) < MIN_GROUPS_FOR_EVIDENCE:
        raise ValueError(f'{holdout} 留出至少需要{MIN_GROUPS_FOR_EVIDENCE}个不同分组，当前只有{len(groups)}个')
    folds = []
    all_de = []
    all_per_channel = []
    for group in groups:
        train = [r for r in readings if (r['scan_id'] if holdout == 'scan' else r['group_id']) != group]
        test = [r for r in readings if (r['scan_id'] if holdout == 'scan' else r['group_id']) == group]
        model = builder(train)
        predicted = predict_readings(test, model)
        predicted_srgb, reference_srgb, de, per_channel = _error_rows(predicted, np.vstack([r['reference'] for r in test]))
        all_de.append(de)
        all_per_channel.append(per_channel)
        folds.append({
            'holdout': group,
            'holdout_kind': holdout,
            'train_patches': len(train),
            'holdout_patches': len(test),
            'holdout_scans': sorted({r['scan_id'] for r in test}),
            'rgb_rmse': float(np.sqrt(np.mean((predicted_srgb - reference_srgb) ** 2))),
            'mean_delta_e76': float(np.mean(de)),
            'p95_delta_e76': float(np.percentile(de, 95)),
            'max_delta_e76': float(np.max(de)),
            'mean_abs_rgb': float(np.mean(per_channel)),
            'strata': stratified_metrics(test, predicted),
        })
    de = np.concatenate(all_de)
    per_channel = np.concatenate(all_per_channel)
    summary = {
        'holdout_kind': holdout,
        'folds': len(folds),
        'pooled_holdout_patches': int(len(de)),
        'pooled_mean_delta_e76': float(np.mean(de)),
        'pooled_p95_delta_e76': float(np.percentile(de, 95)),
        'pooled_max_delta_e76': float(np.max(de)),
        'pooled_mean_abs_rgb': float(np.mean(per_channel)),
        'leakage_rule': '整组留出：同一色块的全部像素随（扫描,色卡）组整体进入训练或验证，不做像素级拆分',
    }
    return {'folds': folds, 'summary': summary}


def compare_candidates(readings, degree=2, field_degree=1):
    """构建两套候选并给出独立留出证据与保守推荐。"""
    simple_builder = lambda train: build_simple(train, degree=degree)
    spatial_builder = lambda train: build_spatial(train, degree=degree, field_degree=field_degree)
    simple_model = build_simple(readings, degree=degree)
    spatial_model = build_spatial(readings, degree=degree, field_degree=field_degree)
    evidence = {'simple': {'model': simple_model, 'training_fit': simple_model.pop('training_fit')},
                'spatial': {'model': spatial_model, 'training_fit': spatial_model.pop('training_fit')}}
    scan_groups = {r['scan_id'] for r in readings}
    card_groups = {r['group_id'] for r in readings}
    for label, builder in (('simple', simple_builder), ('spatial', spatial_builder)):
        entry = evidence[label]
        entry['scan_holdout'] = cross_validation(readings, builder, 'scan') if len(scan_groups) >= MIN_GROUPS_FOR_EVIDENCE else None
        entry['card_holdout'] = cross_validation(readings, builder, 'card') if len(card_groups) >= MIN_GROUPS_FOR_EVIDENCE else None
    recommendation = choose_recommendation(evidence, len(scan_groups), len(card_groups))
    return evidence, recommendation


def choose_recommendation(evidence, scan_count, card_count):
    """保守选择：默认简单方案；灵活方案必须在独立整组留出上确实更好才推荐。"""
    simple = evidence['simple']
    spatial = evidence['spatial']
    primary_simple = simple['scan_holdout'] or simple['card_holdout']
    primary_spatial = spatial['scan_holdout'] or spatial['card_holdout']
    reasons = []
    if primary_simple is None or primary_spatial is None:
        return {'recommended': 'simple', 'reasons': ['独立留出组不足，无法证明灵活方案的泛化改善，按保守原则使用简单方案'],
                'rule': '训练误差低不构成选择灵活方案的理由'}
    simple_de = primary_simple['summary']['pooled_mean_delta_e76']
    spatial_de = primary_spatial['summary']['pooled_mean_delta_e76']
    simple_p95 = primary_simple['summary']['pooled_p95_delta_e76']
    spatial_p95 = primary_spatial['summary']['pooled_p95_delta_e76']
    improvement = (simple_de - spatial_de) / max(simple_de, 1e-9)
    reasons.append(f'整扫描留出平均 ΔE76：简单 {simple_de:.3f}，灵活 {spatial_de:.3f}（相对改善 {improvement * 100:.1f}%）')
    # 灵活方案需要平均误差改善至少10%，且 p95 不劣化，才允许被推荐。
    if improvement >= 0.10 and spatial_p95 <= simple_p95 * 1.05:
        fold_wins = sum(1 for a, b in zip(primary_spatial['folds'], primary_simple['folds'])
                        if a['mean_delta_e76'] < b['mean_delta_e76'])
        reasons.append(f'各留出组胜出 {fold_wins}/{len(primary_simple["folds"])}，且 p95 误差未劣化')
        if fold_wins * 2 >= len(primary_simple['folds']):
            return {'recommended': 'spatial', 'reasons': reasons,
                    'rule': '灵活方案在独立整组留出上稳定改善；最终仍由用户确认'}
        reasons.append('胜出组数未过半，证据不充分')
    else:
        reasons.append('灵活方案未在独立留出上取得≥10%的稳定改善，或 p95 劣化')
    reasons.append('默认采用较简单方案，避免用额外自由度拟合噪声')
    return {'recommended': 'simple', 'reasons': reasons,
            'rule': '训练误差低不构成选择灵活方案的理由；选择必须基于未参与拟合的整组证据'}


# ---------------------------------------------------------------------------
# 可靠范围判定与图像应用
# ---------------------------------------------------------------------------

def _luts(model):
    coefficients = np.asarray(model['response_coefficients'], dtype=np.float64)
    return response_lut(coefficients)


def reliability_domain(model, image_rgb):
    """逐像素判定校准是否有数据支持。返回布尔支持掩码与逐原因外推掩码。"""
    image = rgb_image(image_rgb)
    linear = linearize(image)
    height, width = image.shape[:2]
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    nx = xx / max(width - 1, 1)
    ny = yy / max(height - 1, 1)
    bounds = model['training']['position_bounds']
    position_unsupported = (
        (nx < bounds['nx_min'] - POSITION_MARGIN) | (nx > bounds['nx_max'] + POSITION_MARGIN)
        | (ny < bounds['ny_min'] - POSITION_MARGIN) | (ny > bounds['ny_max'] + POSITION_MARGIN)
    )
    low, high = model['training']['luminance_range']
    observed_light = np.mean(linear, axis=2)
    tone_unsupported = (observed_light < low - TONE_MARGIN) | (observed_light > high + TONE_MARGIN)
    points = np.asarray(model['training']['measured_points'], dtype=np.float64)
    flat = linear.reshape(-1, 3)
    if len(points) >= 2:
        # 训练点两两距离的较大分位作为“颜色邻域”尺度，其1.5倍外为外推。
        pair_distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2).reshape(-1)
        threshold = max(float(np.percentile(pair_distance, 95)) * 1.5, 0.05)
    else:
        threshold = 0.2
    nearest = np.linalg.norm(flat[:, None, :] - points[None, :, :], axis=2).min(axis=1)
    color_unsupported = (nearest > threshold).reshape(height, width)
    supported = ~(position_unsupported | tone_unsupported | color_unsupported)
    return {
        'supported': supported,
        'reasons': {'position': position_unsupported, 'tone': tone_unsupported, 'color': color_unsupported},
        'color_distance_threshold': float(threshold),
        'fractions': {
            'supported': float(supported.mean()),
            'position_extrapolation': float(position_unsupported.mean()),
            'tone_extrapolation': float(tone_unsupported.mean()),
            'color_extrapolation': float(color_unsupported.mean()),
        },
    }


def apply_calibration(model, image_rgb, include_reliability=True):
    """把已确认模型应用到同设置扫描，输出校准图与可靠范围图层。"""
    image = rgb_image(image_rgb)
    linear = linearize(image)
    lut = np.asarray(_luts(model), dtype=np.float64)
    corrected_linear = np.empty_like(linear)
    for channel in range(3):
        index = np.clip(np.rint(linear[:, :, channel] * (LUT_SIZE - 1)).astype(int), 0, LUT_SIZE - 1)
        corrected_linear[:, :, channel] = lut[index, channel]
    if model['kind'] == 'spatial' and model.get('field_coefficients') is not None:
        field = np.asarray(model['field_coefficients'], dtype=np.float64)
        degree = 1 if field.shape[1] == 4 else 2
        height, width = image.shape[:2]
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        nx = np.clip(xx / max(width - 1, 1), 0, 1)
        ny = np.clip(yy / max(height - 1, 1), 0, 1)
        bounds = model['training']['position_bounds']
        # 钳制到训练观测范围，范围外按边缘场处理并仍标记为外推。
        nx_c = np.clip(nx, bounds['nx_min'], bounds['nx_max'])
        ny_c = np.clip(ny, bounds['ny_min'], bounds['ny_max'])
        design = field_design(nx_c.ravel(), ny_c.ravel(), degree)
        for channel in range(3):
            gain = (design @ field[channel]).reshape(height, width)
            corrected_linear[:, :, channel] *= gain
    out_of_range = (corrected_linear < 0) | (corrected_linear > 1)
    corrected_srgb = encode(np.clip(corrected_linear, 0, 1))
    result = {
        'calibrated': corrected_srgb.astype(np.float32),
        'clipped_fraction': float(np.mean(out_of_range)),
    }
    if include_reliability:
        domain = reliability_domain(model, image)
        result['reliability'] = domain
        overlay = corrected_srgb.copy()
        overlay[~domain['supported']] = overlay[~domain['supported']] * 0.55 + np.asarray([0.75, 0.25, 0.15]) * 0.45
        result['reliability_overlay'] = overlay.astype(np.float32)
        result['supported_mask'] = domain['supported'].astype(np.float32)
    return result


def calibrated_density(calibrated_rgb, reference_white_rgb, channel='K'):
    """校准后相对密度：以校准参考纸白为底，未支持像素不参与可靠统计。"""
    image = rgb_image(calibrated_rgb)
    white = np.asarray(_validate_rgb(reference_white_rgb, '校准参考纸白'), dtype=np.float64)
    linear = linearize(image)
    white_linear = srgb_to_linear(white.astype(np.float32)).astype(np.float64)
    if channel == 'K':
        reflected = np.dot(linear, [0.2126, 0.7152, 0.0722])
        base = float(np.dot(white_linear, [0.2126, 0.7152, 0.0722]))
    elif channel in ('C', 'M', 'Y'):
        index = {'C': 0, 'M': 1, 'Y': 2}[channel]
        reflected = linear[:, :, index]
        base = float(white_linear[index])
    else:
        raise ValueError('密度通道只支持 C/M/Y/K')
    density = -np.log10(np.maximum(reflected, 1e-6) / max(base, 1e-6))
    return density.astype(np.float32)


def coverage_gaps(readings):
    """汇总当前数据在暗部、边缘、颜色上的覆盖缺口。"""
    gaps = []
    reference_light = [float(np.mean(r['reference'])) for r in readings]
    if min(reference_light) > 0.15:
        gaps.append({'dimension': 'shadow', 'detail': '缺少线性亮度低于0.15的独立参考，暗部只能外推，不能给可靠标记'})
    nx = [r['nx'] for r in readings]
    ny = [r['ny'] for r in readings]
    if min(nx) > 0.15 or max(nx) < 0.85 or min(ny) > 0.15 or max(ny) < 0.85:
        gaps.append({'dimension': 'edge', 'detail': '训练色块未覆盖画面边缘/四角，边缘位置只能外推'})
    measured = np.vstack([r['measured'] for r in readings])
    chroma = measured.max(axis=1) - measured.min(axis=1)
    if float(np.max(chroma)) < 0.3:
        gaps.append({'dimension': 'chroma', 'detail': '缺少高饱和颜色参考，饱和色区域不能判断'})
    if not gaps:
        gaps.append({'dimension': 'none', 'detail': '暗部、边缘与饱和色均有纳入色块覆盖；仍以各留出组的分区间证据为准'})
    return gaps
