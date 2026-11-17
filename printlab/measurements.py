import csv
import io
import math

import cv2
import numpy as np

from .pixels import CHANNELS, heatmap, integer, number, percentile_statistics, rgb_image
from .screening import _put_text


def parse_patch_csv(text, shape):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('请提供色块位置CSV')
    if len(text.encode('utf-8')) > 1024 * 1024:
        raise ValueError('色块CSV最多1 MiB')
    reader = csv.DictReader(io.StringIO(text.lstrip('\ufeff')))
    required = {'name', 'x', 'y', 'width', 'height'}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError('CSV需要name,x,y,width,height列；可选channel,nominal')
    height, width = shape[:2]
    patches = []
    names = set()
    for line, record in enumerate(reader, start=2):
        if len(patches) >= 512:
            raise ValueError('最多支持512个色块')
        if None in record:
            raise ValueError(f'第{line}行列数与标题不一致')
        if not any(str(value or '').strip() for value in record.values()):
            continue
        name = str(record.get('name') or '').strip()
        if not name or len(name) > 80:
            raise ValueError(f'第{line}行色块名称需要1到80字符')
        if name in names:
            raise ValueError('色块名称不能重复：' + name)
        names.add(name)
        x = integer(record.get('x'), f'{name} x', 0, width - 1)
        y = integer(record.get('y'), f'{name} y', 0, height - 1)
        patch_width = integer(record.get('width'), f'{name}宽度', 1, width - x)
        patch_height = integer(record.get('height'), f'{name}高度', 1, height - y)
        channel = str(record.get('channel') or 'K').strip().upper()
        if channel not in CHANNELS:
            raise ValueError('色块channel需要C/M/Y/K')
        nominal_text = str(record.get('nominal') or '').strip()
        nominal = number(nominal_text, '名义覆盖率', 0, 1) if nominal_text else None
        patches.append({'name': name, 'x': x, 'y': y, 'width': patch_width, 'height': patch_height, 'channel': channel, 'nominal': nominal})
    if not patches:
        raise ValueError('CSV中没有有效色块')
    return patches


def sample_patches(rgb, patches, inset=0.15, paper_rgb=(1, 1, 1), robust=True):
    image = rgb_image(rgb)
    ratio = number(inset, '色块内缩比例', 0, 0.45)
    if len(paper_rgb) != 3:
        raise ValueError('纸白需要RGB三分量')
    paper = np.asarray([number(value, '纸白', 0.05, 1) for value in paper_rgb], dtype=np.float64)
    if not isinstance(patches, list) or not 1 <= len(patches) <= 512:
        raise ValueError('需要1到512个色块')
    height, width = image.shape[:2]
    output = image.copy()
    rows = []
    occupancy = np.zeros((height, width), dtype=np.uint16)
    for index, patch in enumerate(patches):
        x = integer(patch.get('x'), '色块x', 0, width - 1)
        y = integer(patch.get('y'), '色块y', 0, height - 1)
        w = integer(patch.get('width'), '色块宽度', 1, width - x)
        h = integer(patch.get('height'), '色块高度', 1, height - y)
        dx = math.floor(w * ratio)
        dy = math.floor(h * ratio)
        x0, x1 = x + dx, x + w - dx
        y0, y1 = y + dy, y + h - dy
        values = image[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
        if len(values) < 4:
            raise ValueError('色块内缩后至少需要4个像素')
        center = np.median(values, axis=0) if robust else np.mean(values, axis=0)
        deviations = np.linalg.norm(values - center, axis=1)
        spread = float(np.percentile(deviations, 95))
        channel = patch.get('channel', 'K')
        if channel not in CHANNELS:
            raise ValueError('色块channel需要C/M/Y/K')
        if channel == 'K':
            reflected = float(np.dot(center, [0.2126, 0.7152, 0.0722]))
            base = float(np.dot(paper, [0.2126, 0.7152, 0.0722]))
        else:
            component = CHANNELS.index(channel)
            reflected = float(center[component])
            base = float(paper[component])
        density = -math.log10(max(reflected / base, 1e-6))
        name = str(patch.get('name', patch.get('label', index + 1)))
        row = {
            'name': name,
            'channel': channel,
            'nominal': patch.get('nominal'),
            'x': x0,
            'y': y0,
            'width': x1 - x0,
            'height': y1 - y0,
            'pixels': len(values),
            'r': float(center[0]),
            'g': float(center[1]),
            'b': float(center[2]),
            'density': density,
            'rgb_std': values.std(axis=0).tolist(),
            'p95_rgb_deviation': spread,
            'clipped_pixels': int(np.count_nonzero(np.any(values <= 1 / 255, axis=1))),
            'sampling': 'median' if robust else 'mean',
        }
        rows.append(row)
        occupancy[y0:y1, x0:x1] += 1
        color = (0.9, 0.1, 0.1) if spread > 0.08 else (0.1, 0.6, 0.2)
        cv2.rectangle(output, (x0, y0), (x1 - 1, y1 - 1), color, 1)
        if w > 30 and h > 16:
            label = name.encode('ascii', 'replace').decode()[:12]
            _put_text(output, label, (x0 + 1, y0 + min(h - 1, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    groups = []
    for channel in CHANNELS:
        selected = [row for row in rows if row['channel'] == channel]
        if not selected:
            continue
        densities = [row['density'] for row in selected]
        groups.append({'channel': channel, 'patches': len(selected), 'mean_density': float(np.mean(densities)), 'minimum_density': float(np.min(densities)), 'maximum_density': float(np.max(densities)), 'nonuniform_patches': sum(row['p95_rgb_deviation'] > 0.08 for row in selected)})
    return {'summary': {'patches': len(rows), 'inset': ratio, 'robust': bool(robust), 'paper_rgb': paper.tolist(), 'overlapping_sample_pixels': int(np.count_nonzero(occupancy > 1)), 'nonuniform_patches': sum(row['p95_rgb_deviation'] > 0.08 for row in rows), 'density_model': '选定互补RGB分量的反射率负对数，仅作相对测量'}, 'images': {'色块测量位置': output, '色块重叠次数': heatmap(occupancy, maximum=max(int(occupancy.max()), 1))}, 'tables': {'色块读数': rows, '通道读数汇总': groups}}


def measure_csv(rgb, csv_text, inset=0.15, paper_rgb=(1, 1, 1), robust=True):
    image = rgb_image(rgb)
    patches = parse_patch_csv(csv_text, image.shape)
    return sample_patches(image, patches, inset, paper_rgb, robust)


def patch_template(rgb, rows=4, columns=6, margin=12, gap=4):
    image = rgb_image(rgb)
    row_count = integer(rows, '行数', 1, 16)
    column_count = integer(columns, '列数', 1, 16)
    inset = integer(margin, '外边距', 0, min(image.shape[:2]) // 3)
    spacing = integer(gap, '格间距', 0, 50)
    height, width = image.shape[:2]
    patch_width = (width - 2 * inset - (column_count - 1) * spacing) // column_count
    patch_height = (height - 2 * inset - (row_count - 1) * spacing) // row_count
    if min(patch_width, patch_height) < 4:
        raise ValueError('色块网格太密')
    records = []
    for row in range(row_count):
        for column in range(column_count):
            records.append({'name': f'R{row + 1}C{column + 1}', 'x': inset + column * (patch_width + spacing), 'y': inset + row * (patch_height + spacing), 'width': patch_width, 'height': patch_height, 'channel': 'K', 'nominal': None})
    return sample_patches(image, records, inset=0)


def compare_patch_tables(first, second, name_key='name'):
    if not isinstance(first, list) or not isinstance(second, list):
        raise ValueError('测量表需要记录列表')
    left = {}
    right = {}
    for table, target in ((first, left), (second, right)):
        for row in table:
            if not isinstance(row, dict):
                raise ValueError('测量记录必须为对象')
            name = str(row.get(name_key, ''))
            if not name:
                raise ValueError('测量记录缺少色块名')
            if name in target:
                raise ValueError('单次测量中的色块名不能重复')
            target[name] = row
    common = sorted(set(left) & set(right))
    if not common:
        raise ValueError('两组测量没有同名色块')
    rows = []
    for name in common:
        a = left[name]
        b = right[name]
        values_a = np.asarray([number(a.get(key), key, 0, 1) for key in ('r', 'g', 'b')])
        values_b = np.asarray([number(b.get(key), key, 0, 1) for key in ('r', 'g', 'b')])
        density_a = number(a.get('density'), '第一次密度', -5, 10)
        density_b = number(b.get('density'), '第二次密度', -5, 10)
        difference = values_b - values_a
        rows.append({'name': name, 'channel_first': a.get('channel'), 'channel_second': b.get('channel'), 'delta_r': float(difference[0]), 'delta_g': float(difference[1]), 'delta_b': float(difference[2]), 'rgb_distance': float(np.linalg.norm(difference)), 'density_first': density_a, 'density_second': density_b, 'delta_density': density_b - density_a})
    distance = [row['rgb_distance'] for row in rows]
    return {'matched_patches': len(common), 'missing_from_second': sorted(set(left) - set(right)), 'missing_from_first': sorted(set(right) - set(left)), 'rgb_distance': percentile_statistics(distance), 'rows': rows, 'interpretation': '按色块名对应的相对读数比较；不自动拟合印机响应或校准曲线'}
