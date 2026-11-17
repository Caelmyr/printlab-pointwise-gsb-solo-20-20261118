import cv2
import numpy as np
from scipy import ndimage

from .pixels import gray_image, heatmap, integer, luminance, mask_bounds, mask_image, mask_outline, number, overlay_mask, percentile_statistics, polygon_mask, rectangle_mask, rgb_image


def validate_geometry(geometry, shape):
    if not isinstance(geometry, dict):
        raise ValueError('区域几何必须为对象')
    kind = geometry.get('type')
    height, width = shape[:2]
    if kind == 'rectangle':
        x = integer(geometry.get('x'), 'x', 0, width - 1)
        y = integer(geometry.get('y'), 'y', 0, height - 1)
        region_width = integer(geometry.get('width'), '宽度', 1, width - x)
        region_height = integer(geometry.get('height'), '高度', 1, height - y)
        return {'type': kind, 'x': x, 'y': y, 'width': region_width, 'height': region_height}
    if kind == 'polygon':
        points = geometry.get('points')
        polygon_mask(shape, points)
        return {'type': kind, 'points': [[float(p[0]), float(p[1])] for p in points]}
    if kind == 'ellipse':
        cx = number(geometry.get('cx'), '中心x', 0, width - 1)
        cy = number(geometry.get('cy'), '中心y', 0, height - 1)
        rx = number(geometry.get('rx'), '横向半径', 0.5, width)
        ry = number(geometry.get('ry'), '纵向半径', 0.5, height)
        if cx - rx < 0 or cy - ry < 0 or cx + rx > width - 1 or cy + ry > height - 1:
            raise ValueError('椭圆区域不能超出图像')
        return {'type': kind, 'cx': cx, 'cy': cy, 'rx': rx, 'ry': ry}
    raise ValueError('区域支持矩形、多边形或椭圆')


def geometry_mask(geometry, shape):
    checked = validate_geometry(geometry, shape)
    kind = checked['type']
    if kind == 'rectangle':
        return rectangle_mask(shape, checked['x'], checked['y'], checked['width'], checked['height'])
    if kind == 'polygon':
        return polygon_mask(shape, checked['points'])
    yy, xx = np.indices(shape[:2])
    normalized = ((xx - checked['cx']) / checked['rx']) ** 2 + ((yy - checked['cy']) / checked['ry']) ** 2
    return normalized <= 1


def combine_regions(regions, shape, roles=None, default_full=False):
    selected = np.zeros(shape[:2], dtype=bool)
    included = []
    excluded = []
    for region in regions:
        role = region.get('role', 'measure')
        if roles is not None and role not in roles and role != 'exclude':
            continue
        current = geometry_mask(region['geometry'], shape)
        if role == 'exclude':
            excluded.append(current)
        else:
            included.append(current)
    if included:
        for current in included:
            selected |= current
    elif default_full:
        selected[:] = True
    for current in excluded:
        selected &= ~current
    return selected


def region_summary(rgb, regions):
    image = rgb_image(rgb)
    rows = []
    preview = image.copy()
    palette = {'measure': (1, 0.25, 0.1), 'exclude': (0.6, 0.6, 0.6), 'protect': (0.2, 0.8, 0.2), 'sample': (0.2, 0.4, 1)}
    light = luminance(image)
    for region in regions:
        selected = geometry_mask(region['geometry'], image.shape)
        stats = percentile_statistics(light, selected)
        values = image[selected]
        row = {'id': region.get('id'), 'name': region.get('name', '区域'), 'role': region.get('role', 'measure'), 'pixels': int(selected.sum()), 'mean_luminance': stats['mean'], 'std_luminance': stats['std'], 'mean_rgb': values.mean(axis=0).tolist() if len(values) else None, 'bounds': mask_bounds(selected)}
        rows.append(row)
        preview = mask_outline(preview, selected, palette.get(region.get('role'), (1, 0.2, 0.1)))
    return {'summary': {'region_count': len(rows), 'image_size': [image.shape[1], image.shape[0]]}, 'images': {'区域总览': preview}, 'tables': {'区域测量': rows}}


def threshold_region(rgb, lower=0.0, upper=0.5, invert=False, minimum_area=1, fill_holes=False):
    image = rgb_image(rgb)
    low = number(lower, '亮度下限', 0, 1)
    high = number(upper, '亮度上限', 0, 1)
    if low > high:
        raise ValueError('亮度下限不能高于上限')
    area = integer(minimum_area, '最小面积', 1, image.shape[0] * image.shape[1])
    light = luminance(image)
    selected = (light >= low) & (light <= high)
    if bool(invert):
        selected = ~selected
    if bool(fill_holes):
        selected = ndimage.binary_fill_holes(selected)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(selected.astype(np.uint8), connectivity=8)
    retained = np.zeros(selected.shape, dtype=bool)
    rows = []
    for label in range(1, count):
        size = int(stats[label, cv2.CC_STAT_AREA])
        if size < area:
            continue
        retained |= labels == label
        rows.append({'component': label, 'area': size, 'x': int(stats[label, 0]), 'y': int(stats[label, 1]), 'width': int(stats[label, 2]), 'height': int(stats[label, 3]), 'centroid_x': float(centroids[label, 0]), 'centroid_y': float(centroids[label, 1])})
    return {'summary': {'lower': low, 'upper': high, 'components': len(rows), 'selected_pixels': int(retained.sum()), 'selected_fraction': float(retained.mean())}, 'images': {'亮度选区': retained.astype(np.float32), '亮度选区叠图': overlay_mask(image, retained)}, 'tables': {'选区连通块': rows}}


def edge_region(rgb, sigma=1, threshold=0.15, radius=1):
    image = rgb_image(rgb)
    blur = number(sigma, '预平滑', 0, 8)
    cutoff = number(threshold, '边缘阈值', 0.001, 2)
    expand = integer(radius, '边缘扩展', 0, 20)
    light = luminance(image)
    if blur > 0:
        light = ndimage.gaussian_filter(light, blur)
    dx = ndimage.sobel(light, axis=1) / 8
    dy = ndimage.sobel(light, axis=0) / 8
    magnitude = np.hypot(dx, dy)
    selected = magnitude >= cutoff
    if expand:
        selected = ndimage.binary_dilation(selected, iterations=expand)
    return {'summary': {'sigma': blur, 'threshold': cutoff, 'radius': expand, 'edge_pixels': int(selected.sum())}, 'images': {'边缘选区': selected.astype(np.float32), '边缘强度': heatmap(magnitude, maximum=max(cutoff * 3, 0.01)), '边缘叠图': overlay_mask(image, selected, opacity=0.65)}, 'tables': {'梯度统计': [percentile_statistics(magnitude)]}}


def morphology_mask(mask, operation='dilate', radius=1, iterations=1):
    values = gray_image(mask)
    selected = values > 0.5
    size = integer(radius, '形态半径', 1, 30)
    passes = integer(iterations, '重复次数', 1, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * size + 1, 2 * size + 1))
    operations = {'dilate': cv2.MORPH_DILATE, 'erode': cv2.MORPH_ERODE, 'open': cv2.MORPH_OPEN, 'close': cv2.MORPH_CLOSE, 'gradient': cv2.MORPH_GRADIENT}
    if operation not in operations:
        raise ValueError('选区形态操作不支持')
    output = cv2.morphologyEx(selected.astype(np.uint8), operations[operation], kernel, iterations=passes).astype(bool)
    return {'summary': {'operation': operation, 'radius': size, 'iterations': passes, 'before_pixels': int(selected.sum()), 'after_pixels': int(output.sum())}, 'images': {'处理后选区': output.astype(np.float32)}, 'tables': {}}


def masked_edit(rgb, mask, mode='white', strength=1):
    image = rgb_image(rgb)
    selected = mask_image(mask, image.shape)
    alpha = number(strength, '处理强度', 0, 1)
    if not np.any(selected):
        raise ValueError('选区为空')
    if mode == 'white':
        replacement = np.ones_like(image)
    elif mode == 'black':
        replacement = np.zeros_like(image)
    elif mode == 'gray':
        replacement = np.repeat(luminance(image)[:, :, None], 3, axis=2)
    elif mode == 'invert':
        replacement = 1 - image
    elif mode == 'mean':
        replacement = np.broadcast_to(image[selected].mean(axis=0), image.shape)
    else:
        raise ValueError('区域处理方式不支持')
    output = image.copy()
    output[selected] = image[selected] * (1 - alpha) + replacement[selected] * alpha
    return {'summary': {'mode': mode, 'strength': alpha, 'selected_pixels': int(selected.sum())}, 'images': {'区域处理': output, '处理区域': selected.astype(np.float32)}, 'tables': {}}


def sample_grid(rgb, columns=6, rows=4, margin=0.1):
    image = rgb_image(rgb)
    column_count = integer(columns, '采样列数', 1, 32)
    row_count = integer(rows, '采样行数', 1, 32)
    inset = number(margin, '单格内缩比例', 0, 0.45)
    height, width = image.shape[:2]
    if width < column_count * 3 or height < row_count * 3:
        raise ValueError('每个采样格至少需要3像素')
    preview = image.copy()
    light = luminance(image)
    table = []
    for row in range(row_count):
        for column in range(column_count):
            left = round(column * width / column_count)
            right = round((column + 1) * width / column_count)
            top = round(row * height / row_count)
            bottom = round((row + 1) * height / row_count)
            dx = round((right - left) * inset)
            dy = round((bottom - top) * inset)
            x0, x1 = left + dx, right - dx
            y0, y1 = top + dy, bottom - dy
            patch = image[y0:y1, x0:x1]
            row_data = {'row': row, 'column': column, 'x': x0, 'y': y0, 'width': x1 - x0, 'height': y1 - y0, 'mean_r': float(patch[:, :, 0].mean()), 'mean_g': float(patch[:, :, 1].mean()), 'mean_b': float(patch[:, :, 2].mean()), 'std_rgb': patch.std(axis=(0, 1)).tolist(), 'mean_luminance': float(light[y0:y1, x0:x1].mean())}
            table.append(row_data)
            cv2.rectangle(preview, (x0, y0), (x1 - 1, y1 - 1), (1, 0.2, 0.1), 1)
    return {'summary': {'columns': column_count, 'rows': row_count, 'margin': inset}, 'images': {'网格采样位置': preview}, 'tables': {'网格RGB采样': table}}


def region_distance(mask, inside=True):
    selected = gray_image(mask) > 0.5
    if not np.any(selected) or np.all(selected):
        raise ValueError('距离图需要同时包含区域内外像素')
    distance = ndimage.distance_transform_edt(selected if inside else ~selected)
    maximum = float(distance.max())
    normalized = distance / max(maximum, 1)
    return {'summary': {'inside': bool(inside), 'maximum_distance_pixels': maximum, 'selected_pixels': int(selected.sum())}, 'images': {'距离热图': heatmap(distance, maximum=maximum), '归一化距离': normalized}, 'tables': {'距离统计': [percentile_statistics(distance[selected if inside else ~selected])]}}
