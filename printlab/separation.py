import cv2
import numpy as np
from scipy import ndimage

from .pixels import CHANNELS, channel_preview, choice, cmyk_image, heatmap, histogram, luminance, mask_image, number, percentile_statistics, rgb_image


def basic_separation(rgb, black_generation=0.8, total_ink_limit=3.0, black_start=0.0):
    image = rgb_image(rgb)
    strength = number(black_generation, '黑版生成量', 0, 1)
    limit = number(total_ink_limit, '总墨量上限', 1, 4)
    start = number(black_start, '起黑点', 0, 0.9)
    cmy = 1 - image
    neutral = np.min(cmy, axis=2)
    black = np.clip((neutral - start) / (1 - start), 0, 1) * strength
    denominator = np.maximum(1 - black, 1e-6)
    chromatic = np.clip((cmy - black[:, :, None]) / denominator[:, :, None], 0, 1)
    plates = np.concatenate((chromatic, black[:, :, None]), axis=2)
    before = np.sum(plates, axis=2)
    budget = np.maximum(limit - black, 0)
    cmy_sum = chromatic.sum(axis=2)
    scale = np.minimum(1.0, budget / np.maximum(cmy_sum, 1e-6))
    plates[:, :, :3] *= scale[:, :, None]
    preview = composite_simple(plates)
    loss = np.sqrt(np.mean((preview - image) ** 2, axis=2))
    result = {
        'summary': {
            'method': '基础灰成分替代与保黑总墨量压缩',
            'black_generation': strength,
            'black_start': start,
            'total_ink_limit_percent': limit * 100,
            'limited_pixels': int(np.count_nonzero(before > limit)),
            'limited_fraction': float(np.mean(before > limit)),
            'rms_rgb_error': float(np.sqrt(np.mean((preview - image) ** 2))),
            'limitations': '无ICC色域映射，无细字/边缘语义识别',
        },
        'images': {'CMYK分色': plates, '分色合成': preview, '总墨量': heatmap(plates.sum(axis=2), maximum=4), '近似颜色损失': heatmap(loss, maximum=0.3)},
        'kinds': {'CMYK分色': 'cmyk'},
        'tables': {'分色统计': plate_statistics(plates), '墨量分布': histogram(plates.sum(axis=2), 40, range_max=4)},
    }
    for index, channel in enumerate(CHANNELS):
        result['images'][channel + '版'] = channel_preview(plates[:, :, index], channel)
    return result


def composite_simple(cmyk):
    plates = cmyk_image(cmyk)
    return np.clip((1 - plates[:, :, :3]) * (1 - plates[:, :, 3:4]), 0, 1)


def plate_statistics(cmyk, mask=None):
    plates = cmyk_image(cmyk)
    selected = mask_image(mask, plates.shape)
    if not np.any(selected):
        raise ValueError('统计区域为空')
    rows = []
    for index, channel in enumerate(CHANNELS):
        values = plates[:, :, index][selected]
        row = {'channel': channel, 'mean_percent': float(values.mean() * 100), 'covered_percent': float(np.mean(values > 1 / 255) * 100), 'solid_percent': float(np.mean(values >= 0.98) * 100), 'p95_percent': float(np.percentile(values, 95) * 100), 'max_percent': float(values.max() * 100)}
        rows.append(row)
    return rows


def plate_extract(cmyk, channel='K', invert=False):
    plates = cmyk_image(cmyk)
    selected = choice(channel, '色版', CHANNELS)
    index = CHANNELS.index(selected)
    values = plates[:, :, index]
    gray = 1 - values if bool(invert) else values
    return {'summary': {'channel': selected, 'inverted': bool(invert), 'mean_coverage_percent': float(values.mean() * 100)}, 'images': {'灰度色版': gray, '油墨色版': channel_preview(values, selected)}, 'tables': {'分色统计': [plate_statistics(plates)[index]]}}


def limit_ink(cmyk, total_limit=3.0, strategy='preserve_black'):
    plates = cmyk_image(cmyk)
    limit = number(total_limit, '总墨量上限', 1, 4)
    method = choice(strategy, '压墨策略', ('preserve_black', 'proportional'))
    result = plates.copy()
    total = result.sum(axis=2)
    flagged = total > limit
    if method == 'proportional':
        scale = np.minimum(1, limit / np.maximum(total, 1e-8))
        result *= scale[:, :, None]
    else:
        black = result[:, :, 3]
        budget = np.maximum(limit - black, 0)
        color_sum = result[:, :, :3].sum(axis=2)
        scale = np.minimum(1, budget / np.maximum(color_sum, 1e-8))
        result[:, :, :3] *= scale[:, :, None]
    reduction = total - result.sum(axis=2)
    return {'summary': {'strategy': method, 'limit_percent': limit * 100, 'changed_pixels': int(flagged.sum()), 'mean_reduction_percent': float(reduction.mean() * 100), 'max_after_percent': float(result.sum(axis=2).max() * 100)}, 'images': {'限墨分色': result, '限墨预览': composite_simple(result), '压墨量': heatmap(reduction, maximum=max(float(reduction.max()), 0.01))}, 'kinds': {'限墨分色': 'cmyk'}, 'tables': {'限墨后分色': plate_statistics(result)}}


def channel_mix(cmyk, gains=None, offsets=None):
    plates = cmyk_image(cmyk)
    gains = gains or [1, 1, 1, 1]
    offsets = offsets or [0, 0, 0, 0]
    if len(gains) != 4 or len(offsets) != 4:
        raise ValueError('每个色版需要一个增益与偏移')
    gain_values = np.array([number(x, '增益', 0, 2) for x in gains], dtype=np.float32)
    offset_values = np.array([number(x, '偏移', -0.5, 0.5) for x in offsets], dtype=np.float32)
    before_clip = plates * gain_values + offset_values
    mixed = np.clip(before_clip, 0, 1)
    clipped = (before_clip < 0) | (before_clip > 1)
    rows = []
    for index, channel in enumerate(CHANNELS):
        rows.append({'channel': channel, 'gain': float(gain_values[index]), 'offset': float(offset_values[index]), 'clipped_pixels': int(clipped[:, :, index].sum())})
    return {'summary': {'operation': '每版线性墨量调整', 'clipped_values': int(clipped.sum())}, 'images': {'调整分色': mixed, '调整预览': composite_simple(mixed)}, 'kinds': {'调整分色': 'cmyk'}, 'tables': {'调整参数': rows, '调整后分色': plate_statistics(mixed)}}


def spot_mask(rgb, color=(0.9, 0.15, 0.1), tolerance=0.18, softness=0.06):
    image = rgb_image(rgb)
    if not isinstance(color, (list, tuple)) or len(color) != 3:
        raise ValueError('专色参考需要RGB三分量')
    target = np.asarray([number(x, '专色色值', 0, 1) for x in color], dtype=np.float32)
    threshold = number(tolerance, '选色半径', 0.001, 1.732)
    transition = number(softness, '过渡宽度', 0, 0.5)
    distance = np.linalg.norm(image - target, axis=2)
    if transition == 0:
        mask = (distance <= threshold).astype(np.float32)
    else:
        mask = np.clip((threshold + transition - distance) / transition, 0, 1)
    ink_preview = 1 - mask[:, :, None] * (1 - target)
    residual = image * (1 - mask[:, :, None]) + mask[:, :, None]
    return {'summary': {'target_rgb': target.tolist(), 'tolerance': threshold, 'softness': transition, 'mean_coverage_percent': float(mask.mean() * 100), 'selected_pixels': int(np.count_nonzero(mask > 0))}, 'images': {'专色灰度版': mask, '专色预览': ink_preview, '移除所选专色': residual}, 'tables': {'覆盖率分布': histogram(mask)}}


def grayscale_conversion(rgb, method='luminance', gamma=1.0):
    image = rgb_image(rgb)
    mode = choice(method, '灰度方式', ('luminance', 'average', 'lightness', 'black_channel'))
    exponent = number(gamma, '灰度伽马', 0.2, 4)
    if mode == 'luminance':
        gray = luminance(image)
    elif mode == 'average':
        gray = image.mean(axis=2)
    elif mode == 'lightness':
        gray = (image.max(axis=2) + image.min(axis=2)) / 2
    else:
        gray = image.max(axis=2)
    corrected = np.clip(gray, 0, 1) ** exponent
    plates = np.zeros((*corrected.shape, 4), dtype=np.float32)
    plates[:, :, 3] = 1 - corrected
    return {'summary': {'method': mode, 'gamma': exponent, 'black_coverage_percent': float((1 - corrected).mean() * 100)}, 'images': {'黑白图像': np.repeat(corrected[:, :, None], 3, axis=2), '单黑分色': plates}, 'kinds': {'单黑分色': 'cmyk'}, 'tables': {'灰度分布': histogram(corrected)}}


def duotone(rgb, shadow=(0.05, 0.1, 0.2), highlight=(0.98, 0.87, 0.65), gamma=1.0):
    image = rgb_image(rgb)
    if len(shadow) != 3 or len(highlight) != 3:
        raise ValueError('双色调需要阴影和高光RGB颜色')
    dark = np.asarray([number(x, '阴影色', 0, 1) for x in shadow], dtype=np.float32)
    light = np.asarray([number(x, '高光色', 0, 1) for x in highlight], dtype=np.float32)
    power = number(gamma, '亮度伽马', 0.2, 4)
    tone = luminance(image) ** power
    output = dark * (1 - tone[:, :, None]) + light * tone[:, :, None]
    return {'summary': {'shadow_rgb': dark.tolist(), 'highlight_rgb': light.tolist(), 'gamma': power}, 'images': {'双色调图像': output, '阴影油墨版': 1 - tone, '高光油墨版': tone}, 'tables': {'阶调分布': histogram(tone)}}


def neutral_balance(rgb, correction=(0.0, 0.0, 0.0), tonal_range='all'):
    image = rgb_image(rgb)
    if len(correction) != 3:
        raise ValueError('平衡调整需要RGB三个数')
    shifts = np.asarray([number(x, '平衡量', -0.5, 0.5) for x in correction], dtype=np.float32)
    mode = choice(tonal_range, '阶调范围', ('all', 'shadows', 'midtones', 'highlights'))
    light = luminance(image)
    if mode == 'shadows':
        weight = np.clip((0.6 - light) / 0.6, 0, 1)
    elif mode == 'highlights':
        weight = np.clip((light - 0.4) / 0.6, 0, 1)
    elif mode == 'midtones':
        weight = np.maximum(0, 1 - np.abs(light - 0.5) * 2)
    else:
        weight = np.ones(light.shape, dtype=np.float32)
    unclipped = image + weight[:, :, None] * shifts
    output = np.clip(unclipped, 0, 1)
    return {'summary': {'range': mode, 'correction_rgb': shifts.tolist(), 'clipped_fraction': float(np.mean((unclipped < 0) | (unclipped > 1)))}, 'images': {'平衡后图像': output, '调整权重': weight}, 'tables': {'调整前亮度': [percentile_statistics(light)], '调整后亮度': [percentile_statistics(luminance(output))]}}


def posterize(rgb, levels=5, dither=False):
    image = rgb_image(rgb)
    count = int(number(levels, '色阶数', 2, 32))
    source = image.copy()
    if bool(dither):
        bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float32)
        height, width = image.shape[:2]
        noise = np.tile((bayer + 0.5) / 16 - 0.5, (int(np.ceil(height / 4)), int(np.ceil(width / 4))))[:height, :width]
        source = np.clip(source + noise[:, :, None] / (count - 1), 0, 1)
    output = np.round(source * (count - 1)) / (count - 1)
    return {'summary': {'levels_per_channel': count, 'ordered_dither': bool(dither), 'rms_error': float(np.sqrt(np.mean((output - image) ** 2)))}, 'images': {'色阶压缩': output}, 'tables': {}}


def ink_overview(cmyk, threshold=2.8, tile_size=64):
    plates = cmyk_image(cmyk)
    level = number(threshold, '警戒墨量', 1, 4)
    size = int(number(tile_size, '区域边长', 8, 512))
    total = plates.sum(axis=2)
    height, width = total.shape
    rows = []
    preview = composite_simple(plates)
    for y in range(0, height, size):
        for x in range(0, width, size):
            patch = total[y:min(y + size, height), x:min(x + size, width)]
            rows.append({'x': x, 'y': y, 'width': patch.shape[1], 'height': patch.shape[0], 'mean_ink_percent': float(patch.mean() * 100), 'maximum_percent': float(patch.max() * 100), 'over_limit_pixels': int(np.count_nonzero(patch > level))})
            if np.any(patch > level):
                cv2.rectangle(preview, (x, y), (min(x + size - 1, width - 1), min(y + size - 1, height - 1)), (1, 0.1, 0.1), 1)
    return {'summary': {'warning_percent': level * 100, 'total_mean_percent': float(total.mean() * 100), 'over_limit_fraction': float(np.mean(total > level)), 'tiles': len(rows)}, 'images': {'总墨量检查': preview, '总墨量热图': heatmap(total, maximum=4)}, 'tables': {'区域墨量': rows, '色版统计': plate_statistics(plates)}}


def knockout(cmyk, mask, channels=('C', 'M', 'Y', 'K'), feather=0):
    plates = cmyk_image(cmyk)
    selected = mask_image(mask, plates.shape)
    if not np.any(selected):
        raise ValueError('挖空区域为空')
    chosen = [choice(channel, '挖空色版', CHANNELS) for channel in channels]
    if not chosen:
        raise ValueError('至少选择一个色版')
    radius = number(feather, '羽化半径', 0, 20)
    weight = selected.astype(np.float32)
    if radius > 0:
        weight = ndimage.gaussian_filter(weight, radius, mode='constant')
    output = plates.copy()
    for channel in chosen:
        output[:, :, CHANNELS.index(channel)] *= 1 - weight
    return {'summary': {'channels': chosen, 'selected_pixels': int(selected.sum()), 'feather': radius}, 'images': {'挖空分色': output, '挖空预览': composite_simple(output), '挖空权重': weight}, 'kinds': {'挖空分色': 'cmyk'}, 'tables': {'挖空后分色': plate_statistics(output)}}
