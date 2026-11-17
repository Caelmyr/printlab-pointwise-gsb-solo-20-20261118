import cv2
import numpy as np
from scipy import ndimage

from .pixels import CHANNELS, channel_preview, cmyk_image, gray_image, heatmap, integer, luminance, mask_image, number, rgb_image
from .separation import composite_simple, plate_statistics


def assemble_channels(channels, polarity='coverage', reference_shape=None):
    if not isinstance(channels, dict):
        raise ValueError('色版输入必须为按C/M/Y/K命名的对象')
    unknown = set(channels) - set(CHANNELS)
    if unknown:
        raise ValueError('未知色版：' + '、'.join(sorted(unknown)))
    if not channels:
        raise ValueError('至少提供一张色版')
    if polarity not in ('coverage', 'printed'):
        raise ValueError('色版极性需要coverage或printed')
    converted = {}
    shape = tuple(reference_shape) if reference_shape is not None else None
    for channel, array in channels.items():
        values = gray_image(array)
        if shape is None:
            shape = values.shape
        if values.shape != shape:
            raise ValueError('各色版尺寸必须一致，不能自动缩放套准')
        converted[channel] = 1 - values if polarity == 'printed' else values
    output = np.zeros((*shape, 4), dtype=np.float32)
    for index, channel in enumerate(CHANNELS):
        if channel in converted:
            output[:, :, index] = converted[channel]
    return {'summary': {'provided_channels': list(converted), 'empty_channels': [channel for channel in CHANNELS if channel not in converted], 'polarity': polarity, 'size': [shape[1], shape[0]], 'alignment': '输入像素原位，不做隐式配准'}, 'images': {'合并CMYK分色': output, '合并分色预览': composite_simple(output)}, 'kinds': {'合并CMYK分色': 'cmyk'}, 'tables': {'色版覆盖率': plate_statistics(output)}}


def replace_channel(cmyk, replacement, channel='K', polarity='coverage', mask=None):
    plates = cmyk_image(cmyk)
    if channel not in CHANNELS:
        raise ValueError('替换色版需要C/M/Y/K')
    source = gray_image(replacement)
    if source.shape != plates.shape[:2]:
        raise ValueError('替换版与分色图大小必须一致')
    if polarity not in ('coverage', 'printed'):
        raise ValueError('色版极性不支持')
    if polarity == 'printed':
        source = 1 - source
    selected = mask_image(mask, plates.shape)
    output = plates.copy()
    index = CHANNELS.index(channel)
    previous = output[:, :, index].copy()
    output[:, :, index][selected] = source[selected]
    delta = output[:, :, index] - previous
    return {'summary': {'channel': channel, 'polarity': polarity, 'selected_pixels': int(selected.sum()), 'changed_pixels': int(np.count_nonzero(np.abs(delta) > 1e-6)), 'mean_coverage_change': float(delta[selected].mean()) if np.any(selected) else 0}, 'images': {'替换后分色': output, '替换后预览': composite_simple(output), '色版变化': np.clip(delta / 2 + 0.5, 0, 1)}, 'kinds': {'替换后分色': 'cmyk'}, 'tables': {'替换后覆盖率': plate_statistics(output)}}


def channel_union(cmyk, channels=('C', 'M', 'Y'), destination='K', mode='maximum', clear_sources=False):
    plates = cmyk_image(cmyk)
    chosen = list(dict.fromkeys(channels))
    if not chosen or any(channel not in CHANNELS for channel in chosen):
        raise ValueError('请选择有效的源色版')
    if destination not in CHANNELS:
        raise ValueError('目标色版需要C/M/Y/K')
    if mode not in ('maximum', 'add_clipped', 'transmittance'):
        raise ValueError('合版方式不支持')
    values = np.stack([plates[:, :, CHANNELS.index(channel)] for channel in chosen], axis=2)
    if mode == 'maximum':
        merged = values.max(axis=2)
    elif mode == 'add_clipped':
        merged = np.clip(values.sum(axis=2), 0, 1)
    else:
        merged = 1 - np.prod(1 - values, axis=2)
    output = plates.copy()
    if clear_sources:
        for channel in chosen:
            output[:, :, CHANNELS.index(channel)] = 0
    output[:, :, CHANNELS.index(destination)] = merged
    return {'summary': {'sources': chosen, 'destination': destination, 'mode': mode, 'cleared_sources': bool(clear_sources), 'merged_mean': float(merged.mean())}, 'images': {'合版分色': output, '合版预览': composite_simple(output), '目标色版': channel_preview(merged, destination)}, 'kinds': {'合版分色': 'cmyk'}, 'tables': {'合版覆盖率': plate_statistics(output)}}


def plate_threshold(cmyk, thresholds=(0.5, 0.5, 0.5, 0.5), minimum_area=0):
    plates = cmyk_image(cmyk)
    if len(thresholds) != 4:
        raise ValueError('四个色版需要四个阈值')
    cutoffs = [number(value, '色版阈值', 0, 1) for value in thresholds]
    minimum = integer(minimum_area, '最小连通面积', 0, 10000)
    output = np.zeros_like(plates)
    rows = []
    for index, channel in enumerate(CHANNELS):
        selected = plates[:, :, index] >= cutoffs[index]
        removed = 0
        if minimum:
            count, labels, stats, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8), connectivity=8)
            retained = np.zeros(selected.shape, dtype=bool)
            for label in range(1, count):
                size = int(stats[label, cv2.CC_STAT_AREA])
                if size >= minimum:
                    retained |= labels == label
                else:
                    removed += size
            selected = retained
        output[:, :, index] = selected.astype(np.float32)
        rows.append({'channel': channel, 'threshold': cutoffs[index], 'solid_pixels': int(selected.sum()), 'removed_pixels': removed})
    return {'summary': {'thresholds': cutoffs, 'minimum_area': minimum, 'operation': '逐版二值线稿，不进行调幅或调频加网'}, 'images': {'二值色版': output, '二值分色预览': composite_simple(output)}, 'kinds': {'二值色版': 'cmyk'}, 'tables': {'色版二值统计': rows}}


def plate_edges(cmyk, channel='K', radius=1):
    plates = cmyk_image(cmyk)
    if channel not in CHANNELS:
        raise ValueError('色版需要C/M/Y/K')
    distance = integer(radius, '边缘半径', 1, 12)
    plane = plates[:, :, CHANNELS.index(channel)]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * distance + 1, 2 * distance + 1))
    high = cv2.dilate(plane, kernel)
    low = cv2.erode(plane, kernel)
    boundary = high - low
    gradient_x = ndimage.sobel(plane, axis=1) / 8
    gradient_y = ndimage.sobel(plane, axis=0) / 8
    magnitude = np.hypot(gradient_x, gradient_y)
    return {'summary': {'channel': channel, 'radius': distance, 'boundary_pixels': int(np.count_nonzero(boundary > 0.05)), 'mean_gradient': float(magnitude.mean()), 'purpose': '色版边缘人工检查，不自动修改分色'}, 'images': {'色版边缘': boundary, '色版梯度': heatmap(magnitude, maximum=0.25), '原色版': channel_preview(plane, channel)}, 'tables': {}}


def plate_overlap(cmyk, threshold=0.05):
    plates = cmyk_image(cmyk)
    cutoff = number(threshold, '有效覆盖阈值', 0.001, 1)
    active = plates > cutoff
    count = active.sum(axis=2)
    palette = np.asarray([[1, 1, 1], [0.35, 0.75, 0.95], [0.35, 0.85, 0.35], [0.95, 0.75, 0.15], [0.9, 0.2, 0.15]], dtype=np.float32)
    overlap = palette[count]
    rows = []
    for first in range(4):
        for second in range(first + 1, 4):
            common = active[:, :, first] & active[:, :, second]
            union = active[:, :, first] | active[:, :, second]
            union_size = int(union.sum())
            rows.append({'first': CHANNELS[first], 'second': CHANNELS[second], 'overlap_pixels': int(common.sum()), 'overlap_fraction': float(common.mean()), 'intersection_over_union': float(common.sum() / union_size) if union_size else None})
    layers = [{'active_plates': number, 'pixels': int(np.count_nonzero(count == number)), 'fraction': float(np.mean(count == number))} for number in range(5)]
    return {'summary': {'threshold': cutoff, 'four_plate_pixels': int(np.count_nonzero(count == 4)), 'unprinted_pixels': int(np.count_nonzero(count == 0))}, 'images': {'叠印版数': overlap, '分色预览': composite_simple(plates)}, 'tables': {'两版重叠': rows, '重叠版数': layers}}
