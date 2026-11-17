from .screening import _put_text
import math

import cv2
import numpy as np

from .pixels import MAX_PIXELS, integer, number, rgb_image


def mm_to_pixels(value, dpi):
    return round(float(value) / 25.4 * dpi)


def validate_page(width_mm, height_mm, dpi):
    width = number(width_mm, '纸张宽度', 20, 1200)
    height = number(height_mm, '纸张高度', 20, 1200)
    resolution = number(dpi, '排版DPI', 36, 600)
    pixels_w = mm_to_pixels(width, resolution)
    pixels_h = mm_to_pixels(height, resolution)
    if pixels_w * pixels_h > MAX_PIXELS:
        raise ValueError('纸张像素超过419万，请降低排版DPI')
    if min(pixels_w, pixels_h) < 32:
        raise ValueError('纸张像素尺寸太小')
    return width, height, resolution, pixels_w, pixels_h


def resize_for_box(rgb, width, height, fit='contain', background=(1, 1, 1)):
    image = rgb_image(rgb)
    source_h, source_w = image.shape[:2]
    target_w = integer(width, '放置宽度', 2, 4096)
    target_h = integer(height, '放置高度', 2, 4096)
    if fit not in ('contain', 'cover', 'stretch'):
        raise ValueError('适配方式需要contain、cover或stretch')
    if fit == 'stretch':
        result = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return result, {'scale_x': target_w / source_w, 'scale_y': target_h / source_h, 'crop': [0, 0, source_w, source_h]}
    scale = min(target_w / source_w, target_h / source_h) if fit == 'contain' else max(target_w / source_w, target_h / source_h)
    resized_w = max(1, round(source_w * scale))
    resized_h = max(1, round(source_h * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)
    canvas = np.ones((target_h, target_w, 3), dtype=np.float32) * np.asarray(background, dtype=np.float32)
    source_x = max(0, (resized_w - target_w) // 2)
    source_y = max(0, (resized_h - target_h) // 2)
    destination_x = max(0, (target_w - resized_w) // 2)
    destination_y = max(0, (target_h - resized_h) // 2)
    copy_w = min(target_w, resized_w)
    copy_h = min(target_h, resized_h)
    canvas[destination_y:destination_y + copy_h, destination_x:destination_x + copy_w] = resized[source_y:source_y + copy_h, source_x:source_x + copy_w]
    return canvas, {'scale_x': scale, 'scale_y': scale, 'source_crop_pixels': [source_x / scale, source_y / scale, copy_w / scale, copy_h / scale], 'destination_offset': [destination_x, destination_y]}


def crop_marks(canvas, x, y, width, height, length=8, offset=3):
    left, top = x, y
    right, bottom = x + width - 1, y + height - 1
    for px in (left, right):
        cv2.line(canvas, (px, top - offset - length), (px, top - offset), (0, 0, 0), 1)
        cv2.line(canvas, (px, bottom + offset), (px, bottom + offset + length), (0, 0, 0), 1)
    for py in (top, bottom):
        cv2.line(canvas, (left - offset - length, py), (left - offset, py), (0, 0, 0), 1)
        cv2.line(canvas, (right + offset, py), (right + offset + length, py), (0, 0, 0), 1)


def grid_sheet(rgb, page_width_mm=210, page_height_mm=297, dpi=100, columns=2, rows=3, margin_mm=10, gap_mm=4, fit='contain', marks=True):
    image = rgb_image(rgb)
    page_w, page_h, resolution, pixel_w, pixel_h = validate_page(page_width_mm, page_height_mm, dpi)
    count_x = integer(columns, '列数', 1, 12)
    count_y = integer(rows, '行数', 1, 12)
    margin = number(margin_mm, '页边距', 0, min(page_w, page_h) / 3)
    gap = number(gap_mm, '格间距', 0, 50)
    usable_w = page_w - 2 * margin - (count_x - 1) * gap
    usable_h = page_h - 2 * margin - (count_y - 1) * gap
    if usable_w <= count_x or usable_h <= count_y:
        raise ValueError('纸张无法容纳指定行列和间距')
    cell_w = usable_w / count_x
    cell_h = usable_h / count_y
    canvas = np.ones((pixel_h, pixel_w, 3), dtype=np.float32)
    rows_data = []
    for row in range(count_y):
        for column in range(count_x):
            x_mm = margin + column * (cell_w + gap)
            y_mm = margin + row * (cell_h + gap)
            left = mm_to_pixels(x_mm, resolution)
            top = mm_to_pixels(y_mm, resolution)
            right = mm_to_pixels(x_mm + cell_w, resolution)
            bottom = mm_to_pixels(y_mm + cell_h, resolution)
            patch, mapping = resize_for_box(image, right - left, bottom - top, fit)
            canvas[top:bottom, left:right] = patch
            if marks:
                crop_marks(canvas, left, top, right - left, bottom - top, max(2, mm_to_pixels(2, resolution)), max(1, mm_to_pixels(0.8, resolution)))
            rows_data.append({'row': row, 'column': column, 'x_mm': x_mm, 'y_mm': y_mm, 'width_mm': cell_w, 'height_mm': cell_h, 'pixel_box': [left, top, right - left, bottom - top], 'mapping': mapping})
    return {'summary': {'page_mm': [page_w, page_h], 'dpi': resolution, 'copies_per_sheet': count_x * count_y, 'cell_mm': [cell_w, cell_h], 'fit': fit, 'marks': bool(marks)}, 'images': {'规则拼版': canvas}, 'tables': {'拼版位置': rows_data}}


def fixed_size_sheet(rgb, item_width_mm=60, item_height_mm=40, page_width_mm=210, page_height_mm=297, dpi=100, margin_mm=10, gap_mm=3, rotate=False, fit='contain'):
    image = rgb_image(rgb)
    page_w, page_h, resolution, pixel_w, pixel_h = validate_page(page_width_mm, page_height_mm, dpi)
    item_w = number(item_width_mm, '单张宽度', 2, page_w)
    item_h = number(item_height_mm, '单张高度', 2, page_h)
    margin = number(margin_mm, '页边距', 0, min(page_w, page_h) / 3)
    gap = number(gap_mm, '间距', 0, 50)
    if rotate:
        image = np.rot90(image, 1).copy()
        item_w, item_h = item_h, item_w
    columns = math.floor((page_w - 2 * margin + gap) / (item_w + gap))
    rows = math.floor((page_h - 2 * margin + gap) / (item_h + gap))
    if columns < 1 or rows < 1:
        raise ValueError('当前纸张放不下一张指定尺寸图像')
    if columns * rows > 144:
        raise ValueError('单页最多144个位置')
    occupied_w = columns * item_w + (columns - 1) * gap
    occupied_h = rows * item_h + (rows - 1) * gap
    offset_x = (page_w - occupied_w) / 2
    offset_y = (page_h - occupied_h) / 2
    canvas = np.ones((pixel_h, pixel_w, 3), dtype=np.float32)
    placements = []
    for row in range(rows):
        for column in range(columns):
            x = offset_x + column * (item_w + gap)
            y = offset_y + row * (item_h + gap)
            left = mm_to_pixels(x, resolution)
            top = mm_to_pixels(y, resolution)
            right = mm_to_pixels(x + item_w, resolution)
            bottom = mm_to_pixels(y + item_h, resolution)
            patch, mapping = resize_for_box(image, right - left, bottom - top, fit)
            canvas[top:bottom, left:right] = patch
            placements.append({'index': len(placements) + 1, 'x_mm': x, 'y_mm': y, 'width_mm': item_w, 'height_mm': item_h, 'pixel_box': [left, top, right - left, bottom - top]})
    utilization = columns * rows * item_w * item_h / (page_w * page_h)
    return {'summary': {'copies_per_sheet': len(placements), 'columns': columns, 'rows': rows, 'rotated': bool(rotate), 'paper_utilization': utilization, 'page_mm': [page_w, page_h], 'item_mm': [item_w, item_h]}, 'images': {'定尺寸拼版': canvas}, 'tables': {'拼版位置': placements}}


def compare_sheet_layouts(rgb, item_width_mm=60, item_height_mm=40, page_width_mm=210, page_height_mm=297, dpi=100, margin_mm=10, gap_mm=3):
    parameters = {'item_width_mm': item_width_mm, 'item_height_mm': item_height_mm, 'page_width_mm': page_width_mm, 'page_height_mm': page_height_mm, 'dpi': dpi, 'margin_mm': margin_mm, 'gap_mm': gap_mm}
    rows = []
    images = {}
    for rotate in (False, True):
        label = '旋转90度' if rotate else '原方向'
        try:
            result = fixed_size_sheet(rgb, rotate=rotate, **parameters)
            images[label] = result['images']['定尺寸拼版']
            rows.append({'orientation': label, 'fits': True, 'copies_per_sheet': result['summary']['copies_per_sheet'], 'paper_utilization': result['summary']['paper_utilization']})
        except ValueError as exc:
            rows.append({'orientation': label, 'fits': False, 'copies_per_sheet': 0, 'reason': str(exc)})
    if not images:
        raise ValueError('两个方向均无法放入纸张')
    best = max(rows, key=lambda row: row['copies_per_sheet'])
    return {'summary': {'recommended_orientation': best['orientation'], 'maximum_copies': best['copies_per_sheet'], 'search_scope': '仅比较统一方向规则网格，不自动混排'}, 'images': images, 'tables': {'方向比较': rows}}


def contact_sheet(images, names=None, columns=3, thumb_width=240, thumb_height=180, padding=12):
    if not isinstance(images, list) or not 1 <= len(images) <= 36:
        raise ValueError('联系表需要1到36张图像')
    col_count = integer(columns, '列数', 1, 8)
    cell_w = integer(thumb_width, '缩略图宽度', 64, 512)
    cell_h = integer(thumb_height, '缩略图高度', 64, 512)
    gap = integer(padding, '留白', 4, 40)
    row_count = math.ceil(len(images) / col_count)
    caption_height = 28
    width = col_count * (cell_w + gap) + gap
    height = row_count * (cell_h + caption_height + gap) + gap
    if width * height > MAX_PIXELS:
        raise ValueError('联系表超过像素预算，请减小缩略图')
    canvas = np.ones((height, width, 3), dtype=np.float32)
    rows = []
    labels = names or [f'Image {index + 1}' for index in range(len(images))]
    if len(labels) != len(images):
        raise ValueError('图像名称数量不匹配')
    for index, source in enumerate(images):
        image = rgb_image(source)
        row, column = divmod(index, col_count)
        left = gap + column * (cell_w + gap)
        top = gap + row * (cell_h + caption_height + gap)
        patch, mapping = resize_for_box(image, cell_w, cell_h)
        canvas[top:top + cell_h, left:left + cell_w] = patch
        cv2.rectangle(canvas, (left, top), (left + cell_w - 1, top + cell_h - 1), (0.7, 0.7, 0.7), 1)
        ascii_name = str(labels[index]).encode('ascii', 'replace').decode()[:30]
        _put_text(canvas, f'{index + 1}: {ascii_name}', (left, top + cell_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0.1, 0.1, 0.1), 1, cv2.LINE_AA)
        rows.append({'index': index + 1, 'name': str(labels[index]), 'source_size': [image.shape[1], image.shape[0]], 'position': [left, top, cell_w, cell_h]})
    return {'summary': {'images': len(images), 'columns': col_count, 'rows': row_count, 'output_size': [width, height]}, 'images': {'图像联系表': canvas}, 'tables': {'联系表索引': rows}}


def ruler_overlay(rgb, dpi=300, interval_mm=10, color=(0.1, 0.1, 0.1)):
    image = rgb_image(rgb)
    resolution = number(dpi, 'DPI', 36, 2400)
    interval = number(interval_mm, '刻度间距', 1, 100)
    height, width = image.shape[:2]
    step = interval / 25.4 * resolution
    if step < 8:
        raise ValueError('刻度过密，请增大间距')
    output = image.copy()
    tint = tuple(number(x, '刻度颜色', 0, 1) for x in color)
    if len(tint) != 3:
        raise ValueError('刻度颜色需要RGB')
    tick = min(20, max(5, round(resolution / 25.4)))
    horizontal = []
    vertical = []
    index = 0
    while index * step < width:
        x = round(index * step)
        cv2.line(output, (x, 0), (x, tick), tint, 1)
        cv2.line(output, (x, height - 1), (x, height - 1 - tick), tint, 1)
        if index:
            _put_text(output, str(round(index * interval, 2)), (x + 2, tick + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, tint, 1)
        horizontal.append({'pixel': x, 'millimeters': index * interval})
        index += 1
    index = 0
    while index * step < height:
        y = round(index * step)
        cv2.line(output, (0, y), (tick, y), tint, 1)
        cv2.line(output, (width - 1, y), (width - 1 - tick, y), tint, 1)
        vertical.append({'pixel': y, 'millimeters': index * interval})
        index += 1
    return {'summary': {'dpi': resolution, 'interval_mm': interval, 'physical_size_mm': [width / resolution * 25.4, height / resolution * 25.4]}, 'images': {'尺寸标尺': output}, 'tables': {'横向刻度': horizontal, '纵向刻度': vertical}}
