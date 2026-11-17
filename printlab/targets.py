from .screening import _put_text
import math

import cv2
import numpy as np

from .pixels import CHANNELS, MAX_PIXELS, integer, number
from .separation import composite_simple


def canvas_size(width, height):
    w = integer(width, '靶标宽度', 128, 2048)
    h = integer(height, '靶标高度', 128, 2048)
    if w * h > MAX_PIXELS:
        raise ValueError('靶标超过像素预算')
    return w, h


def text(canvas, value, x, y, scale=0.4, color=(0.1, 0.1, 0.1)):
    _put_text(canvas, str(value), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def rgb_artwork(width=720, height=540, seed=17):
    w, h = canvas_size(width, height)
    rng = np.random.default_rng(integer(seed, '素材种子', 0, 2 ** 31 - 1))
    yy, xx = np.indices((h, w), dtype=np.float32)
    x = xx / max(w - 1, 1)
    y = yy / max(h - 1, 1)
    image = np.ones((h, w, 3), dtype=np.float32)
    image[:, :, 0] = 0.96 - 0.12 * y
    image[:, :, 1] = 0.95 - 0.1 * y
    image[:, :, 2] = 0.89 - 0.1 * y
    margin = max(12, w // 30)
    title_height = max(40, h // 7)
    cv2.rectangle(image, (margin, margin), (w - margin, title_height), (0.08, 0.16, 0.2), -1)
    text(image, 'PRINTLAB / RASTER PROOF', margin + 12, title_height - 18, 0.65, (0.98, 0.88, 0.55))
    x0, y0 = margin, title_height + 18
    x1, y1 = w // 2 - 8, h - 2 * margin - 40
    patch_x = x[y0:y1, x0:x1]
    patch_y = y[y0:y1, x0:x1]
    image[y0:y1, x0:x1, 0] = 0.1 + 0.85 * patch_x / 0.5
    image[y0:y1, x0:x1, 1] = 0.15 + 0.6 * patch_y
    image[y0:y1, x0:x1, 2] = 0.8 - 0.55 * patch_x / 0.5
    for radius in (70, 53, 34):
        cv2.circle(image, (int(w * 0.3), int(h * 0.5)), max(3, round(radius * w / 720)), (0.92, 0.3 + radius / 250, 0.12), -1)
    right_x = w // 2 + 12
    colors = [(0.05, 0.8, 0.9), (0.9, 0.1, 0.5), (0.98, 0.85, 0.05), (0.08, 0.08, 0.08), (0.2, 0.65, 0.25), (0.3, 0.25, 0.8)]
    patch_width = max(12, (w - right_x - margin) // 3)
    patch_height = max(12, (h // 3) // 2)
    for i, color in enumerate(colors):
        left = right_x + (i % 3) * patch_width
        top = y0 + (i // 3) * patch_height
        cv2.rectangle(image, (left + 2, top + 2), (left + patch_width - 4, top + patch_height - 4), color, -1)
    text_y = y0 + 2 * patch_height + 26
    text(image, 'Fine type and registration', right_x, text_y, 0.44)
    text(image, '0123456789 ABCDEFG', right_x, text_y + 25, 0.4)
    text(image, 'thin lines / neutral shadows', right_x, text_y + 48, 0.32)
    for index in range(14):
        line_y = text_y + 65 + index * 3
        if line_y >= h - margin - 28:
            break
        cv2.line(image, (right_x, line_y), (w - margin, line_y), (0.05, 0.05, 0.05), 1)
    ramp_y = h - margin - 22
    ramp = np.linspace(0, 1, w - 2 * margin, dtype=np.float32)
    image[ramp_y:ramp_y + 18, margin:w - margin] = ramp[None, :, None]
    for point in ((margin, margin), (w - margin - 1, margin), (margin, h - margin - 1), (w - margin - 1, h - margin - 1)):
        cx, cy = point
        cv2.line(image, (cx - 7, cy), (cx + 7, cy), (0, 0, 0), 1)
        cv2.line(image, (cx, cy - 7), (cx, cy + 7), (0, 0, 0), 1)
    noise = rng.normal(0, 0.0005, image.shape).astype(np.float32)
    image = np.clip(image + noise, 0, 1)
    return {'summary': {'synthetic': True, 'kind': '合成栅格印刷原稿', 'seed': seed, 'size': [w, h]}, 'images': {'合成印刷原稿': image}, 'tables': {}}


def scanned_proof(width=720, height=540, seed=17, shift_x=3, shift_y=-2, blur=0.55, defects=True):
    artwork = rgb_artwork(width, height, seed)['images']['合成印刷原稿']
    w, h = canvas_size(width, height)
    dx = number(shift_x, '横向偏移', -30, 30)
    dy = number(shift_y, '纵向偏移', -30, 30)
    sigma = number(blur, '模糊半径', 0, 4)
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    proof = cv2.warpAffine(artwork, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(1, 1, 1))
    if sigma > 0:
        proof = cv2.GaussianBlur(proof, (0, 0), sigma)
    proof = np.clip(proof * np.asarray([0.99, 0.97, 0.92], dtype=np.float32), 0, 1)
    rng = np.random.default_rng(int(seed) + 20)
    proof += rng.normal(0, 0.002, proof.shape).astype(np.float32)
    defect_mask = np.zeros((h, w), dtype=np.float32)
    if defects:
        center = (round(w * 0.26), round(h * 0.42))
        radius = max(3, w // 120)
        cv2.circle(proof, center, radius, (0.96, 0.96, 0.9), -1)
        cv2.circle(defect_mask, center, radius, 1, -1)
        x0 = round(w * 0.73)
        y0 = round(h * 0.32)
        cv2.line(proof, (x0, y0), (x0 + 35, y0 + 12), (0.1, 0.1, 0.1), 2)
        cv2.line(defect_mask, (x0, y0), (x0 + 35, y0 + 12), 1, 2)
    proof = np.clip(proof, 0, 1)
    return {'summary': {'synthetic': True, 'kind': '合成扫描校样', 'translation': [dx, dy], 'blur_sigma': sigma, 'defects_inserted': bool(defects), 'warning': '演示合成图，不代表真实印机测量'}, 'images': {'合成扫描校样': proof, '演示缺陷位置': defect_mask}, 'tables': {}}


def repeat_texture(width=320, height=240, seed=9):
    w, h = canvas_size(width, height)
    yy, xx = np.indices((h, w), dtype=np.float32)
    x = xx / w
    y = yy / h
    base = 0.5 + 0.12 * np.sin(2 * np.pi * (3 * x + 2 * y)) + 0.08 * np.cos(2 * np.pi * (5 * x - y))
    image = np.stack((base * 0.8 + 0.18, base * 0.65 + 0.2, base * 0.45 + 0.2), axis=2)
    rng = np.random.default_rng(integer(seed, '素材种子', 0, 2 ** 31 - 1))
    centers = rng.uniform(0.1, 0.9, (8, 2))
    for cx, cy in centers:
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                dx = x - (cx + ox)
                dy = y - (cy + oy)
                petal = np.exp(-(dx * dx / 0.003 + dy * dy / 0.008))
                image[:, :, 0] += petal * 0.18
                image[:, :, 1] -= petal * 0.09
                image[:, :, 2] -= petal * 0.1
    original = np.clip(image, 0, 1)
    discontinuous = original.copy()
    gradient = np.linspace(-0.06, 0.06, w, dtype=np.float32)
    discontinuous = np.clip(discontinuous + gradient[None, :, None], 0, 1)
    return {'summary': {'synthetic': True, 'kind': '周期纹样与曝光接缝样本', 'seed': seed}, 'images': {'周期纹样': original, '有曝光接缝纹样': discontinuous}, 'tables': {}}


def registration_target(width=720, height=540, pitch=80, line_width=1):
    w, h = canvas_size(width, height)
    spacing = integer(pitch, '标记间距', 24, 256)
    stroke = integer(line_width, '线宽', 1, 5)
    plates = np.zeros((h, w, 4), dtype=np.float32)
    rows = []
    for y in range(spacing // 2, h, spacing):
        for x in range(spacing // 2, w, spacing):
            size = max(8, spacing // 5)
            for channel in range(4):
                plane = np.ascontiguousarray(plates[:, :, channel])
                cv2.line(plane, (x - size, y), (x + size, y), 1, stroke)
                cv2.line(plane, (x, y - size), (x, y + size), 1, stroke)
                cv2.circle(plane, (x, y), size // 2, 1, stroke)
                plates[:, :, channel] = plane
            rows.append({'x': x, 'y': y, 'size': size})
    return {'summary': {'synthetic': True, 'target': '套准十字与圆环', 'pitch_pixels': spacing, 'mark_count': len(rows)}, 'images': {'套准靶标': composite_simple(plates), '套准分色': plates}, 'kinds': {'套准分色': 'cmyk'}, 'tables': {'套准标记': rows}}


def slanted_edge_target(width=640, height=480, angle=5, contrast=0.8, blur=0):
    w, h = canvas_size(width, height)
    degrees = number(angle, '斜边角度', -20, 20)
    amplitude = number(contrast, '对比度', 0.1, 1)
    sigma = number(blur, '模拟模糊', 0, 6)
    yy, xx = np.indices((h, w), dtype=np.float32)
    boundary = w / 2 + (yy - h / 2) * math.tan(math.radians(degrees))
    gray = np.where(xx < boundary, 0.5 - amplitude / 2, 0.5 + amplitude / 2).astype(np.float32)
    if sigma:
        gray = cv2.GaussianBlur(gray, (0, 0), sigma)
    image = np.repeat(gray[:, :, None], 3, axis=2)
    return {'summary': {'synthetic': True, 'target': '斜边清晰度靶标', 'angle': degrees, 'contrast': amplitude, 'blur_sigma': sigma}, 'images': {'斜边靶标': image}, 'tables': {}}


def star_target(width=640, height=640, spokes=36, inner_radius=4):
    w, h = canvas_size(width, height)
    count = integer(spokes, '明暗线对数', 4, 120)
    inner = number(inner_radius, '中心留白半径', 0, min(w, h) / 4)
    yy, xx = np.indices((h, w), dtype=np.float32)
    dx = xx - (w - 1) / 2
    dy = yy - (h - 1) / 2
    radius = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)
    maximum = min(w, h) * 0.45
    ring = (radius >= inner) & (radius <= maximum)
    gray = np.ones((h, w), dtype=np.float32)
    gray[ring] = (np.cos(angle[ring] * count) >= 0).astype(np.float32)
    image = np.repeat(gray[:, :, None], 3, axis=2)
    text(image, f'Siemens star / {count} line pairs', 20, 24)
    return {'summary': {'synthetic': True, 'target': '星形分辨率靶标', 'line_pairs_per_turn': count, 'outer_radius': maximum}, 'images': {'星形靶标': image}, 'tables': {}}


def line_resolution_target(width=800, height=500, frequencies=None):
    w, h = canvas_size(width, height)
    periods = frequencies or [2, 3, 4, 6, 8, 12, 16, 24]
    if not isinstance(periods, list) or len(periods) < 2 or len(periods) > 16:
        raise ValueError('线条周期需要2到16档')
    periods = [integer(x, '线条周期', 2, 128) for x in periods]
    image = np.ones((h, w, 3), dtype=np.float32)
    margin = 20
    region_width = (w - 2 * margin) // len(periods)
    if region_width < 12:
        raise ValueError('靶标宽度不足')
    rows = []
    for index, period in enumerate(periods):
        x0 = margin + index * region_width
        x1 = x0 + region_width - 4
        y0 = 50
        y1 = h // 2 - 10
        x = np.arange(x1 - x0)
        vertical = ((x % period) < period / 2).astype(np.float32)
        image[y0:y1, x0:x1] = vertical[None, :, None]
        y2 = h // 2 + 20
        y3 = h - 30
        y = np.arange(y3 - y2)
        horizontal = ((y % period) < period / 2).astype(np.float32)
        image[y2:y3, x0:x1] = horizontal[:, None, None]
        text(image, str(period) + ' px', x0, 30, 0.35)
        rows.append({'period_pixels': period, 'frequency_cycles_per_pixel': 1 / period, 'x': x0, 'width': x1 - x0, 'vertical_y': y0, 'horizontal_y': y2})
    return {'summary': {'synthetic': True, 'target': '横纵线条分辨率靶标', 'periods': periods}, 'images': {'线条分辨率靶标': image}, 'tables': {'线条区域': rows}}


def gray_balance_target(width=800, height=600, steps=9):
    w, h = canvas_size(width, height)
    count = integer(steps, '灰阶档数', 3, 17)
    plates = np.zeros((h, w, 4), dtype=np.float32)
    margin = 20
    patch_width = (w - 2 * margin) // count
    row_height = (h - 2 * margin) // 4
    table = []
    modes = ['K only', 'CMY neutral', 'CMY warm', 'CMY cool']
    for row, label in enumerate(modes):
        for column in range(count):
            level = column / (count - 1)
            if row == 0:
                values = [0, 0, 0, level]
            elif row == 1:
                values = [level, level, level, 0]
            elif row == 2:
                values = [max(0, level - 0.04), level, min(1, level + 0.04), 0]
            else:
                values = [min(1, level + 0.04), level, max(0, level - 0.04), 0]
            x0 = margin + column * patch_width
            y0 = margin + row * row_height
            plates[y0 + 20:y0 + row_height - 6, x0 + 2:x0 + patch_width - 3] = values
            table.append({'row': row, 'column': column, 'mode': label, 'nominal': level, 'cmyk': values, 'x': x0 + 2, 'y': y0 + 20, 'width': patch_width - 5, 'height': row_height - 26})
    preview = composite_simple(plates)
    for row, label in enumerate(modes):
        text(preview, label, margin, margin + row * row_height + 14, 0.38)
    return {'summary': {'synthetic': True, 'target': '单黑和三色灰平衡比较靶标', 'steps': count}, 'images': {'灰平衡靶标': preview, '灰平衡分色': plates}, 'kinds': {'灰平衡分色': 'cmyk'}, 'tables': {'灰平衡色块': table}}


def overprint_target(width=720, height=480, levels=None):
    w, h = canvas_size(width, height)
    amounts = levels or [0.25, 0.5, 0.75, 1]
    if not isinstance(amounts, list) or not 2 <= len(amounts) <= 8:
        raise ValueError('叠印覆盖率需要2到8档')
    amounts = [number(x, '覆盖率', 0, 1) for x in amounts]
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
    plates = np.zeros((h, w, 4), dtype=np.float32)
    rows = []
    margin = 12
    cell_w = (w - 2 * margin) // len(amounts)
    cell_h = (h - 2 * margin) // len(pairs)
    for row, (first, second) in enumerate(pairs):
        for column, level in enumerate(amounts):
            x0 = margin + column * cell_w
            y0 = margin + row * cell_h
            plates[y0 + 10:y0 + cell_h - 4, x0 + 3:x0 + cell_w - 3, first] = level
            plates[y0 + 10:y0 + cell_h - 4, x0 + 3:x0 + cell_w - 3, second] = level
            rows.append({'pair': CHANNELS[first] + CHANNELS[second], 'coverage': level, 'x': x0 + 3, 'y': y0 + 10, 'width': cell_w - 6, 'height': cell_h - 14})
    return {'summary': {'synthetic': True, 'target': '双色叠印色块', 'pairs': len(pairs), 'levels': amounts}, 'images': {'叠印靶标': composite_simple(plates), '叠印靶标分色': plates}, 'kinds': {'叠印靶标分色': 'cmyk'}, 'tables': {'叠印色块': rows}}


def dot_target(width=640, height=480, radius=2, pitch=12, jitter=0, seed=20):
    w, h = canvas_size(width, height)
    r = integer(radius, '圆点半径', 1, 20)
    step = integer(pitch, '圆点间距', 2 * r + 1, 100)
    amount = number(jitter, '位置抖动', 0, step / 4)
    rng = np.random.default_rng(integer(seed, '种子', 0, 2 ** 31 - 1))
    gray = np.ones((h, w), dtype=np.float32)
    centers = []
    for y in range(step, h - step, step):
        for x in range(step, w - step, step):
            cx = round(x + rng.uniform(-amount, amount))
            cy = round(y + rng.uniform(-amount, amount))
            cv2.circle(gray, (cx, cy), r, 0, -1)
            centers.append({'x': cx, 'y': cy, 'radius': r})
    return {'summary': {'synthetic': True, 'target': '规则圆点检测靶标', 'radius': r, 'pitch': step, 'dots': len(centers)}, 'images': {'圆点靶标': np.repeat(gray[:, :, None], 3, axis=2)}, 'tables': {'圆点真值': centers}}


def target_catalog():
    return [
        {'id': 'artwork', 'name': '合成印刷原稿', 'description': '渐变、文字、细线、色块和套准标记'},
        {'id': 'proof', 'name': '合成扫描校样', 'description': '轻微平移、纸白变化和两个演示缺陷'},
        {'id': 'repeat', 'name': '循环纹样', 'description': '周期纹样及带曝光接缝的对照'},
        {'id': 'registration', 'name': '套准标记', 'description': '规则十字及圆环的四版靶标'},
        {'id': 'slanted_edge', 'name': '斜边靶标', 'description': '检查模糊与边缘响应'},
        {'id': 'star', 'name': '星形靶标', 'description': '环向不同空间频率'},
        {'id': 'lines', 'name': '线条分辨率', 'description': '横纵多档周期黑白条纹'},
        {'id': 'gray_balance', 'name': '灰平衡', 'description': '单黑与三色灰阶并列'},
        {'id': 'overprint', 'name': '叠印色块', 'description': '六组双色覆盖率阶梯'},
        {'id': 'dots', 'name': '圆点靶标', 'description': '可控半径及间距圆点'},
    ]


def generate_target(kind, parameters):
    generators = {'artwork': rgb_artwork, 'proof': scanned_proof, 'repeat': repeat_texture, 'registration': registration_target, 'slanted_edge': slanted_edge_target, 'star': star_target, 'lines': line_resolution_target, 'gray_balance': gray_balance_target, 'overprint': overprint_target, 'dots': dot_target}
    if kind not in generators:
        raise ValueError('演示素材类型不支持')
    return generators[kind](**parameters)
