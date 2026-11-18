from .screening import _put_text
import copy

import numpy as np

from .pixels import CHANNELS, integer, number


DEFAULT_CURVES = {channel: [[0, 0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.75], [1, 1]] for channel in CHANNELS}
DEFAULT_INKS = [
    {'name': 'Cyan', 'rgb': [0.08, 0.74, 0.82], 'density': 1.25, 'cost_per_kg': 90},
    {'name': 'Magenta', 'rgb': [0.85, 0.12, 0.48], 'density': 1.3, 'cost_per_kg': 105},
    {'name': 'Yellow', 'rgb': [0.98, 0.9, 0.08], 'density': 1.1, 'cost_per_kg': 85},
    {'name': 'Black', 'rgb': [0.08, 0.08, 0.08], 'density': 1.5, 'cost_per_kg': 75},
]
DEFAULT_PAPER = {'rgb': [0.97, 0.96, 0.92], 'gsm': 120, 'name': '暖白演示纸', 'width_mm': 210, 'height_mm': 297}
DEFAULT_SCREEN = {'method': 'am', 'lpi': 50, 'dpi': 300, 'angles': [15, 75, 0, 45], 'dot_shape': 'round', 'seed': 42}
DEFAULT_SPOT_INKS = [
    {'name': '青', 'rgb': [0.05, 0.66, 0.8], 'density': 1.1, 'cost_per_kg': 120},
    {'name': '品红', 'rgb': [0.85, 0.1, 0.42], 'density': 1.15, 'cost_per_kg': 130},
    {'name': '黄', 'rgb': [0.98, 0.85, 0.05], 'density': 1.0, 'cost_per_kg': 110},
    {'name': '黑', 'rgb': [0.1, 0.1, 0.11], 'density': 1.3, 'cost_per_kg': 90},
    {'name': '专橙', 'rgb': [0.95, 0.42, 0.08], 'density': 1.1, 'cost_per_kg': 160},
    {'name': '专绿', 'rgb': [0.1, 0.55, 0.3], 'density': 1.1, 'cost_per_kg': 160},
]


def text_name(value, fallback):
    result = str(value if value is not None else fallback).strip()
    if not result or len(result) > 80:
        raise ValueError('配置名称需要1到80个字符')
    return result


def validate_rgb(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(label + '需要三个RGB分量')
    return [number(x, label, 0, 1) for x in value]


def validate_curves(payload):
    if not isinstance(payload, dict):
        raise ValueError('曲线配置必须为对象')
    curves = payload.get('curves', payload)
    result = {}
    for channel in CHANNELS:
        points = curves.get(channel)
        if not isinstance(points, list) or not 2 <= len(points) <= 33:
            raise ValueError(channel + '曲线需要2到33个控制点')
        checked = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError('曲线点需要输入值和输出值')
            checked.append([number(point[0], '输入值', 0, 1), number(point[1], '输出值', 0, 1)])
        checked.sort(key=lambda point: point[0])
        if checked[0][0] != 0 or checked[-1][0] != 1:
            raise ValueError(channel + '曲线必须包含输入0和1')
        for previous, current in zip(checked, checked[1:]):
            if current[0] <= previous[0]:
                raise ValueError('曲线输入点不能重复')
            if current[1] < previous[1]:
                raise ValueError('曲线覆盖率必须单调不减')
        result[channel] = checked
    return {'curves': result, 'interpolation': 'piecewise_linear', 'description': str(payload.get('description', ''))[:1000]}


def validate_inks(payload):
    if not isinstance(payload, dict):
        raise ValueError('油墨配置必须为对象')
    inks = payload.get('inks')
    if not isinstance(inks, list) or len(inks) != 4:
        raise ValueError('油墨配置需要CMYK四种油墨')
    result = []
    for index, ink in enumerate(inks):
        if not isinstance(ink, dict):
            raise ValueError('油墨项必须为对象')
        result.append({'channel': CHANNELS[index], 'name': text_name(ink.get('name'), CHANNELS[index]), 'rgb': validate_rgb(ink.get('rgb'), '实色RGB'), 'density': number(ink.get('density'), '墨层密度', 0.1, 4, 1.2), 'cost_per_kg': number(ink.get('cost_per_kg'), '每公斤价格', 0, 10000, 100)})
    paper = validate_rgb(payload.get('paper_rgb', [1, 1, 1]), '纸白RGB')
    return {'inks': result, 'paper_rgb': paper, 'model': 'multiplicative_rgb_transmittance', 'description': str(payload.get('description', ''))[:1000]}


def validate_paper(payload):
    if not isinstance(payload, dict):
        raise ValueError('纸张配置必须为对象')
    return {'name': text_name(payload.get('name'), '纸张'), 'rgb': validate_rgb(payload.get('rgb', [1, 1, 1]), '纸白RGB'), 'gsm': number(payload.get('gsm'), '克重', 20, 600, 120), 'width_mm': number(payload.get('width_mm'), '纸张宽度', 10, 2000, 210), 'height_mm': number(payload.get('height_mm'), '纸张高度', 10, 2000, 297), 'description': str(payload.get('description', ''))[:1000]}


def validate_screen(payload):
    if not isinstance(payload, dict):
        raise ValueError('网屏配置必须为对象')
    method = payload.get('method', 'am')
    if method not in ('am', 'fm'):
        raise ValueError('网屏方式必须为am或fm')
    angles = payload.get('angles', [15, 75, 0, 45])
    if not isinstance(angles, list) or len(angles) != 4:
        raise ValueError('四个色版需要四个网角')
    lpi = number(payload.get('lpi'), '网线数', 5, 200, 50)
    dpi = number(payload.get('dpi'), '输出DPI', 36, 2400, 300)
    if dpi / lpi < 2:
        raise ValueError('每网目至少需要2个输出像素')
    shape = payload.get('dot_shape', 'round')
    if shape not in ('round', 'ellipse', 'line', 'square'):
        raise ValueError('网点形状不支持')
    return {'method': method, 'lpi': lpi, 'dpi': dpi, 'angles': [number(x, '网角', -180, 180) for x in angles], 'dot_shape': shape, 'seed': integer(payload.get('seed'), '随机种子', 0, 2 ** 31 - 1, 42)}


def validate_tolerances(payload):
    if not isinstance(payload, dict):
        raise ValueError('检查阈值必须为对象')
    return {'rgb_rmse_warning': number(payload.get('rgb_rmse_warning'), 'RGB误差警戒', 0.001, 1, 0.08), 'density_warning': number(payload.get('density_warning'), '密度偏差警戒', 0.001, 4, 0.1), 'total_ink_warning': number(payload.get('total_ink_warning'), '总墨量警戒', 1, 4, 2.8), 'minimum_component_area': integer(payload.get('minimum_component_area'), '最小缺陷面积', 1, 100000, 9), 'sharpness_warning': number(payload.get('sharpness_warning'), '清晰度警戒', 0, 10, 0.05)}


def validate_profile(kind, payload):
    from .spotcolor import validate_spot_inks
    validators = {'curves': validate_curves, 'inks': validate_inks, 'paper': validate_paper, 'screen': validate_screen, 'tolerances': validate_tolerances, 'spot_inks': validate_spot_inks}
    if kind not in validators:
        raise ValueError('配置种类不支持')
    return validators[kind](payload)


def default_profiles():
    from .spotcolor import validate_spot_inks
    return [
        {'kind': 'curves', 'name': '线性墨量曲线', 'payload': validate_curves({'curves': copy.deepcopy(DEFAULT_CURVES)})},
        {'kind': 'inks', 'name': '演示CMYK油墨', 'payload': validate_inks({'inks': copy.deepcopy(DEFAULT_INKS), 'paper_rgb': [0.97, 0.96, 0.92]})},
        {'kind': 'paper', 'name': '暖白演示纸', 'payload': validate_paper(copy.deepcopy(DEFAULT_PAPER))},
        {'kind': 'screen', 'name': '50线演示网屏', 'payload': validate_screen(copy.deepcopy(DEFAULT_SCREEN))},
        {'kind': 'tolerances', 'name': '基础质量警戒', 'payload': validate_tolerances({})},
        {'kind': 'spot_inks', 'name': '演示专色油墨库', 'payload': validate_spot_inks({'inks': copy.deepcopy(DEFAULT_SPOT_INKS), 'paper_rgb': [0.97, 0.96, 0.92]})},
    ]


def curve_table(profile, samples=21):
    checked = validate_curves(profile)
    count = integer(samples, '采样点数', 3, 257)
    x = np.linspace(0, 1, count)
    rows = []
    for index, value in enumerate(x):
        row = {'input': float(value)}
        for channel in CHANNELS:
            points = np.asarray(checked['curves'][channel], dtype=np.float64)
            row[channel] = float(np.interp(value, points[:, 0], points[:, 1]))
        rows.append(row)
    return rows


def curve_preview(profile, width=512, height=320):
    import cv2
    checked = validate_curves(profile)
    w = integer(width, '预览宽度', 128, 1024)
    h = integer(height, '预览高度', 128, 1024)
    canvas = np.ones((h, w, 3), dtype=np.float32)
    margin = 24
    for ratio in np.linspace(0, 1, 5):
        x = round(margin + ratio * (w - 2 * margin))
        y = round(h - margin - ratio * (h - 2 * margin))
        cv2.line(canvas, (x, margin), (x, h - margin), (0.85, 0.85, 0.85), 1)
        cv2.line(canvas, (margin, y), (w - margin, y), (0.85, 0.85, 0.85), 1)
    colors = [(0, 0.6, 0.7), (0.85, 0.1, 0.5), (0.75, 0.65, 0), (0.1, 0.1, 0.1)]
    for index, channel in enumerate(CHANNELS):
        points = checked['curves'][channel]
        screen_points = np.asarray([[round(margin + p[0] * (w - 2 * margin)), round(h - margin - p[1] * (h - 2 * margin))] for p in points], dtype=np.int32)
        cv2.polylines(canvas, [screen_points], False, colors[index], 2)
        for x, y in screen_points:
            cv2.circle(canvas, (int(x), int(y)), 3, colors[index], -1)
    return canvas


def profile_differences(first, second):
    if first.get('kind') != second.get('kind'):
        raise ValueError('只能比较同类配置')
    kind = first['kind']
    a = validate_profile(kind, first['payload'])
    b = validate_profile(kind, second['payload'])
    rows = []
    if kind == 'curves':
        samples = np.linspace(0, 1, 101)
        for channel in CHANNELS:
            left = np.asarray(a['curves'][channel])
            right = np.asarray(b['curves'][channel])
            delta = np.interp(samples, right[:, 0], right[:, 1]) - np.interp(samples, left[:, 0], left[:, 1])
            rows.append({'channel': channel, 'mean_absolute_change': float(np.abs(delta).mean()), 'maximum_change': float(np.abs(delta).max()), 'midpoint_change': float(delta[50])})
    elif kind == 'inks':
        for index, channel in enumerate(CHANNELS):
            delta = np.asarray(b['inks'][index]['rgb']) - np.asarray(a['inks'][index]['rgb'])
            rows.append({'channel': channel, 'rgb_distance': float(np.linalg.norm(delta)), 'density_change': b['inks'][index]['density'] - a['inks'][index]['density'], 'cost_change': b['inks'][index]['cost_per_kg'] - a['inks'][index]['cost_per_kg']})
    else:
        for key in a:
            if a[key] == b.get(key):
                continue
            rows.append({'field': key, 'before': a[key], 'after': b.get(key)})
    return {'kind': kind, 'first_name': first['name'], 'second_name': second['name'], 'differences': rows}
