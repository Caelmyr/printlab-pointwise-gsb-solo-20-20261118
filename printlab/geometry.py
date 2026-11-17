import math

import cv2
import numpy as np
from scipy import ndimage

from .pixels import CHANNELS, MAX_PIXELS, channel_preview, choice, cmyk_image, gray_image, heatmap, integer, luminance, number, rgb_image
from .separation import composite_simple, plate_statistics


def crop_image(rgb, x=0, y=0, width=None, height=None):
    image = rgb_image(rgb)
    source_height, source_width = image.shape[:2]
    left = integer(x, '裁切x', 0, source_width - 2)
    top = integer(y, '裁切y', 0, source_height - 2)
    target_width = integer(width, '裁切宽度', 2, source_width - left, source_width - left)
    target_height = integer(height, '裁切高度', 2, source_height - top, source_height - top)
    cropped = image[top:top + target_height, left:left + target_width].copy()
    marked = image.copy()
    cv2.rectangle(marked, (left, top), (left + target_width - 1, top + target_height - 1), (1, 0.1, 0.1), 2)
    return {'summary': {'source_size': [source_width, source_height], 'crop': {'x': left, 'y': top, 'width': target_width, 'height': target_height}, 'coordinate_mapping': {'source_x': f'x+{left}', 'source_y': f'y+{top}'}}, 'images': {'裁切图像': cropped, '裁切位置': marked}, 'tables': {}}


def add_bleed(rgb, pixels=16, mode='reflect', color=(1, 1, 1), trim_marks=True):
    image = rgb_image(rgb)
    border = integer(pixels, '出血像素', 1, 512)
    policy = choice(mode, '出血填充', ('reflect', 'replicate', 'constant'))
    height, width = image.shape[:2]
    if (height + 2 * border) * (width + 2 * border) > MAX_PIXELS:
        raise ValueError('加出血后超过419万像素预算')
    colors = tuple(number(x, '底色', 0, 1) for x in color)
    if len(colors) != 3:
        raise ValueError('出血底色需要RGB')
    border_mode = {'reflect': cv2.BORDER_REFLECT_101, 'replicate': cv2.BORDER_REPLICATE, 'constant': cv2.BORDER_CONSTANT}[policy]
    expanded = cv2.copyMakeBorder(image, border, border, border, border, border_mode, value=colors)
    marked = expanded.copy()
    if trim_marks:
        length = max(1, border - 3)
        left, top = border, border
        right, bottom = border + width - 1, border + height - 1
        for x in (left, right):
            cv2.line(marked, (x, 0), (x, length), (0, 0, 0), 1)
            cv2.line(marked, (x, bottom + 3), (x, bottom + 2 + length), (0, 0, 0), 1)
        for y in (top, bottom):
            cv2.line(marked, (0, y), (length, y), (0, 0, 0), 1)
            cv2.line(marked, (right + 3, y), (right + 2 + length, y), (0, 0, 0), 1)
    return {'summary': {'bleed_pixels': border, 'mode': policy, 'trim_box': [border, border, width, height], 'output_size': [width + border * 2, height + border * 2]}, 'images': {'出血图像': expanded, '裁切标记': marked}, 'tables': {}}


def rotate_image(rgb, angle=0, expand=True, background=(1, 1, 1)):
    image = rgb_image(rgb)
    degrees = number(angle, '旋转角度', -180, 180)
    height, width = image.shape[:2]
    center = ((width - 1) / 2, (height - 1) / 2)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1)
    if expand:
        cos_value = abs(matrix[0, 0])
        sin_value = abs(matrix[0, 1])
        target_width = int(math.ceil(width * cos_value + height * sin_value))
        target_height = int(math.ceil(width * sin_value + height * cos_value))
        matrix[0, 2] += (target_width - width) / 2
        matrix[1, 2] += (target_height - height) / 2
    else:
        target_width, target_height = width, height
    if target_width * target_height > MAX_PIXELS:
        raise ValueError('旋转后超过像素预算')
    color = tuple(number(x, '背景色', 0, 1) for x in background)
    if len(color) != 3:
        raise ValueError('背景色需要RGB')
    result = cv2.warpAffine(image, matrix, (target_width, target_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=color)
    valid = cv2.warpAffine(np.ones((height, width), dtype=np.uint8), matrix, (target_width, target_height), flags=cv2.INTER_NEAREST, borderValue=0)
    return {'summary': {'angle': degrees, 'expanded': bool(expand), 'source_to_output': matrix.tolist(), 'output_size': [target_width, target_height]}, 'images': {'旋转图像': result, '有效图像区域': valid.astype(np.float32)}, 'tables': {}}


def perspective_rectify(rgb, corners, width=800, height=600):
    image = rgb_image(rgb)
    if not isinstance(corners, list) or len(corners) != 4:
        raise ValueError('需要按左上、右上、右下、左下顺序给出四角')
    coordinates = np.asarray(corners, dtype=np.float32)
    if coordinates.shape != (4, 2) or not np.all(np.isfinite(coordinates)):
        raise ValueError('四角坐标无效')
    image_height, image_width = image.shape[:2]
    if np.any(coordinates[:, 0] < 0) or np.any(coordinates[:, 0] > image_width - 1) or np.any(coordinates[:, 1] < 0) or np.any(coordinates[:, 1] > image_height - 1):
        raise ValueError('四角必须在图像内')
    contour = coordinates.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 16:
        raise ValueError('四角必须组成不自交的有效凸四边形')
    target_width = integer(width, '输出宽度', 16, 4096)
    target_height = integer(height, '输出高度', 16, 4096)
    if target_width * target_height > MAX_PIXELS:
        raise ValueError('输出超过像素预算')
    destination = np.asarray([[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(coordinates, destination)
    warped = cv2.warpPerspective(image, matrix, (target_width, target_height), flags=cv2.INTER_LINEAR, borderValue=(1, 1, 1))
    marked = image.copy()
    cv2.polylines(marked, [coordinates.astype(np.int32)], True, (1, 0.1, 0.1), 2)
    return {'summary': {'corners': coordinates.tolist(), 'output_size': [target_width, target_height], 'source_to_output': matrix.tolist(), 'model': '四点平面单应，不校正纸张局部弯曲'}, 'images': {'四角校正': warped, '四角位置': marked}, 'tables': {}}


def translate_plates(cmyk, shifts=None, interpolation='linear'):
    plates = cmyk_image(cmyk)
    selected = shifts or [[0, 0], [0, 0], [0, 0], [0, 0]]
    if len(selected) != 4:
        raise ValueError('四个色版各需要x/y偏移')
    mode = choice(interpolation, '采样方式', ('linear', 'nearest'))
    flag = cv2.INTER_LINEAR if mode == 'linear' else cv2.INTER_NEAREST
    height, width = plates.shape[:2]
    output = np.zeros_like(plates)
    rows = []
    for index, channel in enumerate(CHANNELS):
        if len(selected[index]) != 2:
            raise ValueError('色版偏移需要x/y')
        x = number(selected[index][0], channel + '横向偏移', -100, 100)
        y = number(selected[index][1], channel + '纵向偏移', -100, 100)
        matrix = np.float32([[1, 0, x], [0, 1, y]])
        output[:, :, index] = cv2.warpAffine(plates[:, :, index], matrix, (width, height), flags=flag, borderValue=0)
        rows.append({'channel': channel, 'dx': x, 'dy': y, 'before_ink_sum': float(plates[:, :, index].sum()), 'after_ink_sum': float(output[:, :, index].sum())})
    return {'summary': {'interpolation': mode, 'canvas_policy': '固定纸张画布，移出部分裁去', 'shifts': selected}, 'images': {'偏移分色': output, '套印模拟': composite_simple(output)}, 'kinds': {'偏移分色': 'cmyk'}, 'tables': {'色版偏移': rows}}


def register_translation(source, reference, max_shift=40, window=True):
    moving = rgb_image(source)
    fixed = rgb_image(reference)
    if moving.shape != fixed.shape:
        raise ValueError('平移套准要求两张图像大小相同')
    limit = number(max_shift, '最大平移', 1, 256)
    first = luminance(fixed).astype(np.float32)
    second = luminance(moving).astype(np.float32)
    if np.std(first) < 1e-5 or np.std(second) < 1e-5:
        raise ValueError('纯色图没有可用的套准纹理')
    height, width = first.shape
    apodization = cv2.createHanningWindow((width, height), cv2.CV_32F) if window else None
    shift, response = cv2.phaseCorrelate(second, first, apodization)
    dx, dy = float(shift[0]), float(shift[1])
    if abs(dx) > limit or abs(dy) > limit:
        raise ValueError('估计平移超出搜索范围，请先粗裁切')
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(moving, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(1, 1, 1))
    valid = cv2.warpAffine(np.ones((height, width), dtype=np.float32), matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=0) > 0.999
    if not np.any(valid):
        raise ValueError('套准后没有共同有效区域')
    error = np.sqrt(np.mean((aligned - fixed) ** 2, axis=2))
    before = np.sqrt(np.mean((moving - fixed) ** 2, axis=2))
    blend = (fixed + aligned) / 2
    error_visual = heatmap(error, maximum=0.2)
    error_visual[~valid] = (0.5, 0.5, 0.5)
    return {'summary': {'dx': dx, 'dy': dy, 'phase_response': float(response), 'valid_fraction': float(valid.mean()), 'before_rmse': float(np.sqrt(np.mean(before[valid] ** 2))), 'after_rmse': float(np.sqrt(np.mean(error[valid] ** 2))), 'source_to_reference': matrix.tolist(), 'model': '单一全图平移；不提供局部变形或遮挡估计'}, 'images': {'平移套准': aligned, '套准叠图': blend, '套准误差': error_visual, '共同有效区': valid.astype(np.float32)}, 'tables': {}}


def register_landmarks(source, reference, source_points, reference_points, model='affine'):
    moving = rgb_image(source)
    fixed = rgb_image(reference)
    mode = choice(model, '标点模型', ('similarity', 'affine'))
    left = np.asarray(source_points, dtype=np.float32)
    right = np.asarray(reference_points, dtype=np.float32)
    minimum = 3
    if left.ndim != 2 or left.shape[1] != 2 or left.shape != right.shape or len(left) < minimum or len(left) > 100:
        raise ValueError('需要3到100对对应标点')
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError('标点坐标必须有限')
    if np.linalg.matrix_rank(left - left.mean(axis=0)) < 2:
        raise ValueError('标点不能共线')
    if mode == 'affine':
        design = np.column_stack((left, np.ones(len(left))))
        coefficients, _, _, _ = np.linalg.lstsq(design, right, rcond=None)
        matrix = coefficients.T.astype(np.float32)
    else:
        design = np.zeros((len(left) * 2, 4), dtype=np.float64)
        design[0::2, 0] = left[:, 0]
        design[0::2, 1] = -left[:, 1]
        design[0::2, 2] = 1
        design[1::2, 0] = left[:, 1]
        design[1::2, 1] = left[:, 0]
        design[1::2, 3] = 1
        parameters, _, _, _ = np.linalg.lstsq(design, right.reshape(-1), rcond=None)
        a, b, tx, ty = parameters
        matrix = np.asarray([[a, -b, tx], [b, a, ty]], dtype=np.float32)
    determinant = float(np.linalg.det(matrix[:, :2]))
    if determinant <= 0 or determinant < 0.05 or determinant > 20:
        raise ValueError('标点拟合出现翻折或过大的缩放')
    predicted = np.column_stack((left, np.ones(len(left)))) @ matrix.T
    residual = np.linalg.norm(predicted - right, axis=1)
    height, width = fixed.shape[:2]
    warped = cv2.warpAffine(moving, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(1, 1, 1))
    valid = cv2.warpAffine(np.ones(moving.shape[:2], dtype=np.float32), matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=0) > 0.999
    rows = [{'index': i + 1, 'source_x': float(left[i, 0]), 'source_y': float(left[i, 1]), 'reference_x': float(right[i, 0]), 'reference_y': float(right[i, 1]), 'residual_pixels': float(residual[i])} for i in range(len(left))]
    return {'summary': {'model': mode, 'source_to_reference': matrix.tolist(), 'landmarks': len(left), 'rms_landmark_error': float(np.sqrt(np.mean(residual ** 2))), 'max_landmark_error': float(residual.max()), 'determinant': determinant}, 'images': {'标点套准': warped, '标点套准叠图': (warped + fixed) / 2, '标点有效区': valid.astype(np.float32)}, 'tables': {'标点残差': rows}}


def manual_trap(cmyk, channel='C', radius=1, mode='spread', threshold=0.02):
    plates = cmyk_image(cmyk)
    ink = choice(channel, '补偿色版', CHANNELS)
    distance = integer(radius, '边缘补偿半径', 1, 12)
    method = choice(mode, '补偿方式', ('spread', 'choke'))
    floor = number(threshold, '有效墨阈值', 0, 1)
    index = CHANNELS.index(ink)
    plane = plates[:, :, index]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * distance + 1, 2 * distance + 1))
    if method == 'spread':
        changed = cv2.dilate(plane, kernel)
    else:
        changed = cv2.erode(plane, kernel)
    changed[changed < floor] = 0
    result = plates.copy()
    result[:, :, index] = changed
    delta = np.abs(changed - plane)
    return {'summary': {'channel': ink, 'mode': method, 'radius_pixels': distance, 'changed_pixels': int(np.count_nonzero(delta > 1e-6)), 'limitations': '手工单版形态扩缩；不识别相邻油墨、细字保护或多版冲突'}, 'images': {'边缘补偿分色': result, '边缘补偿预览': composite_simple(result), '补偿变化': heatmap(delta, maximum=1)}, 'kinds': {'边缘补偿分色': 'cmyk'}, 'tables': {'补偿后分色': plate_statistics(result)}}


def tile_repeat(rgb, rows=2, columns=2, gap=0, offset=0, mode='straight'):
    image = rgb_image(rgb)
    repeat_rows = integer(rows, '重复行数', 1, 8)
    repeat_columns = integer(columns, '重复列数', 1, 8)
    spacing = integer(gap, '间距', 0, 128)
    policy = choice(mode, '重复方式', ('straight', 'half_drop', 'brick'))
    height, width = image.shape[:2]
    output_height = height * repeat_rows + spacing * (repeat_rows - 1)
    output_width = width * repeat_columns + spacing * (repeat_columns - 1)
    if output_height * output_width > MAX_PIXELS:
        raise ValueError('重复后超过419万像素预算')
    canvas = np.ones((output_height, output_width, 3), dtype=np.float32)
    placements = []
    for row in range(repeat_rows):
        for column in range(repeat_columns):
            x = column * (width + spacing)
            y = row * (height + spacing)
            patch = image
            if policy == 'half_drop' and column % 2:
                patch = np.roll(patch, height // 2, axis=0)
            if policy == 'brick' and row % 2:
                patch = np.roll(patch, width // 2, axis=1)
            canvas[y:y + height, x:x + width] = patch
            placements.append({'row': row, 'column': column, 'x': x, 'y': y, 'width': width, 'height': height})
    return {'summary': {'rows': repeat_rows, 'columns': repeat_columns, 'gap': spacing, 'mode': policy, 'output_size': [output_width, output_height], 'seams_repaired': False}, 'images': {'循环排布': canvas}, 'tables': {'图块位置': placements}}


def offset_preview(rgb, x_fraction=0.5, y_fraction=0.5):
    image = rgb_image(rgb)
    x_part = number(x_fraction, '横向循环偏移比例', 0, 1)
    y_part = number(y_fraction, '纵向循环偏移比例', 0, 1)
    height, width = image.shape[:2]
    dx = round(width * x_part)
    dy = round(height * y_part)
    result = np.roll(np.roll(image, dx, axis=1), dy, axis=0)
    marked = result.copy()
    if dx % width:
        cv2.line(marked, (dx % width, 0), (dx % width, height - 1), (1, 0.05, 0.05), 1)
    if dy % height:
        cv2.line(marked, (0, dy % height), (width - 1, dy % height), (1, 0.05, 0.05), 1)
    return {'summary': {'dx': dx, 'dy': dy, 'wrap': True, 'operation': '循环偏移只移动像素，不修补接缝'}, 'images': {'循环偏移': result, '接缝位置': marked}, 'tables': {}}
