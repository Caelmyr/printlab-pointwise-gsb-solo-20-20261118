import math

import cv2
import numpy as np
from scipy import ndimage

from .pixels import choice, heatmap, histogram, integer, luminance, mask_image, number, percentile_statistics, rgb_image, srgb_to_linear, linear_to_srgb


def levels(rgb, black=0, white=1, gamma=1, per_channel=False):
    image = rgb_image(rgb)
    lower = number(black, '黑场', 0, 0.99)
    upper = number(white, '白场', 0.01, 1)
    power = number(gamma, '中间调伽马', 0.2, 5)
    if lower >= upper:
        raise ValueError('白场必须高于黑场')
    if per_channel:
        corrected = np.clip((image - lower) / (upper - lower), 0, 1) ** (1 / power)
    else:
        before = luminance(image)
        after = np.clip((before - lower) / (upper - lower), 0, 1) ** (1 / power)
        ratio = np.divide(after, before, out=np.zeros_like(after), where=before > 1e-6)
        corrected = np.clip(image * ratio[:, :, None], 0, 1)
    shadow_clip = np.all(image <= lower, axis=2)
    highlight_clip = np.all(image >= upper, axis=2)
    return {'summary': {'black': lower, 'white': upper, 'gamma': power, 'per_channel': bool(per_channel), 'shadow_clipped_pixels': int(shadow_clip.sum()), 'highlight_clipped_pixels': int(highlight_clip.sum())}, 'images': {'色阶调整': corrected, '黑场裁切': shadow_clip.astype(np.float32), '白场裁切': highlight_clip.astype(np.float32)}, 'tables': {'调整前亮度': histogram(luminance(image)), '调整后亮度': histogram(luminance(corrected))}}


def auto_levels(rgb, low_percentile=0.5, high_percentile=99.5, mask=None, neutral=False):
    image = rgb_image(rgb)
    low = number(low_percentile, '低百分位', 0, 20)
    high = number(high_percentile, '高百分位', 80, 100)
    selected = mask_image(mask, image.shape)
    if selected.sum() < 16:
        raise ValueError('自动色阶至少需要16个有效像素')
    rows = []
    output = np.zeros_like(image)
    if neutral:
        values = luminance(image)[selected]
        black, white = np.percentile(values, [low, high])
        if white - black < 1e-5:
            raise ValueError('区域亮度变化不足，不能自动拉伸')
        result = levels(image, black=float(black), white=float(white), per_channel=False)
        result['summary']['sample_pixels'] = int(selected.sum())
        result['summary']['automatic'] = True
        return result
    for channel in range(3):
        values = image[:, :, channel][selected]
        black, white = np.percentile(values, [low, high])
        if white - black < 1e-5:
            output[:, :, channel] = image[:, :, channel]
            rows.append({'channel': 'RGB'[channel], 'black': float(black), 'white': float(white), 'status': 'flat_unchanged'})
            continue
        output[:, :, channel] = np.clip((image[:, :, channel] - black) / (white - black), 0, 1)
        rows.append({'channel': 'RGB'[channel], 'black': float(black), 'white': float(white), 'status': 'stretched'})
    return {'summary': {'low_percentile': low, 'high_percentile': high, 'sample_pixels': int(selected.sum()), 'mode': '独立RGB直方图拉伸'}, 'images': {'自动色阶': output}, 'tables': {'通道端点': rows}}


def paper_white(rgb, sample_mask, target=(1, 1, 1), strength=1, preserve_black=True):
    image = rgb_image(rgb)
    selected = mask_image(sample_mask, image.shape)
    if np.count_nonzero(selected) < 9:
        raise ValueError('纸白参考区域至少需要9像素')
    if len(target) != 3:
        raise ValueError('目标纸白需要RGB分量')
    target_color = np.asarray([number(x, '目标纸白', 0.5, 1) for x in target], dtype=np.float32)
    alpha = number(strength, '校正强度', 0, 1)
    values = image[selected]
    measured = np.median(values, axis=0)
    if np.any(measured < 0.2):
        raise ValueError('所选区域过暗，不能作为纸白')
    gain = target_color / measured
    corrected = image * gain
    if preserve_black:
        weight = np.clip(luminance(image) / 0.2, 0, 1)
        corrected = image + (corrected - image) * weight[:, :, None]
    unbounded = image * (1 - alpha) + corrected * alpha
    output = np.clip(unbounded, 0, 1)
    spread = values.std(axis=0)
    return {'summary': {'measured_white': measured.tolist(), 'sample_std_rgb': spread.tolist(), 'target': target_color.tolist(), 'gain_rgb': gain.tolist(), 'strength': alpha, 'clipped_fraction': float(np.mean(unbounded > 1)), 'preserve_black': bool(preserve_black)}, 'images': {'纸白校正': output, '纸白参考': selected.astype(np.float32)}, 'tables': {'纸白样本': [percentile_statistics(values[:, i]) for i in range(3)]}}


def flatten_background(rgb, radius=35, strength=1, mode='divide'):
    image = rgb_image(rgb)
    scale = number(radius, '背景尺度', 3, 128)
    alpha = number(strength, '校正强度', 0, 1)
    method = choice(mode, '背景校正方式', ('divide', 'subtract'))
    background = np.empty_like(image)
    kernel = max(3, int(scale * 2 + 1))
    if kernel % 2 == 0:
        kernel += 1
    for channel in range(3):
        high = cv2.dilate(image[:, :, channel], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))
        background[:, :, channel] = cv2.GaussianBlur(high, (0, 0), scale / 2)
    reference = np.percentile(background.reshape(-1, 3), 95, axis=0)
    if method == 'divide':
        corrected = image / np.maximum(background, 0.05) * reference
    else:
        corrected = image - background + reference
    output = np.clip(image * (1 - alpha) + corrected * alpha, 0, 1)
    change = np.linalg.norm(output - image, axis=2)
    return {'summary': {'radius': scale, 'strength': alpha, 'mode': method, 'reference_rgb': reference.tolist(), 'mean_change': float(change.mean()), 'limitation': '背景尺度由用户指定；可能改变大面积深色图形'}, 'images': {'背景校正': output, '估计纸张背景': background, '背景校正幅度': heatmap(change, maximum=0.25)}, 'tables': {}}


def denoise(rgb, method='median', radius=1, strength=0.5):
    image = rgb_image(rgb)
    selected = choice(method, '降噪方式', ('median', 'gaussian', 'bilateral'))
    size = integer(radius, '降噪半径', 1, 10)
    alpha = number(strength, '降噪强度', 0, 1)
    if selected == 'median':
        planes = [ndimage.median_filter(image[:, :, i], size=2 * size + 1, mode='reflect') for i in range(3)]
        filtered = np.stack(planes, axis=2)
    elif selected == 'gaussian':
        filtered = cv2.GaussianBlur(image, (0, 0), size)
    else:
        filtered = cv2.bilateralFilter(image, 2 * size + 1, 0.08, float(size))
    output = image * (1 - alpha) + filtered * alpha
    residual = np.sqrt(np.mean((image - output) ** 2, axis=2))
    return {'summary': {'method': selected, 'radius': size, 'strength': alpha, 'rms_change': float(np.sqrt(np.mean((image - output) ** 2)))}, 'images': {'降噪图像': output, '降噪变化': heatmap(residual, maximum=0.1)}, 'tables': {'变化统计': [percentile_statistics(residual)]}}


def unsharp(rgb, radius=1.2, amount=0.8, threshold=0.02, luminance_only=True):
    image = rgb_image(rgb)
    sigma = number(radius, '锐化半径', 0.2, 10)
    gain = number(amount, '锐化量', 0, 3)
    cutoff = number(threshold, '细节阈值', 0, 0.5)
    if luminance_only:
        light = luminance(image)
        blurred = cv2.GaussianBlur(light, (0, 0), sigma)
        detail = light - blurred
        detail[np.abs(detail) < cutoff] = 0
        unbounded = image + (detail * gain)[:, :, None]
    else:
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        detail = image - blurred
        detail[np.abs(detail) < cutoff] = 0
        unbounded = image + detail * gain
    output = np.clip(unbounded, 0, 1)
    difference = luminance(output) - luminance(image)
    return {'summary': {'radius': sigma, 'amount': gain, 'threshold': cutoff, 'luminance_only': bool(luminance_only), 'clipped_fraction': float(np.mean((unbounded < 0) | (unbounded > 1)))}, 'images': {'锐化图像': output, '锐化细节': np.clip(difference + 0.5, 0, 1)}, 'tables': {'亮度改变量': [percentile_statistics(difference)]}}


def resample_print(rgb, source_dpi=300, target_dpi=150, method='area', width_mm=None):
    image = rgb_image(rgb)
    original_dpi = number(source_dpi, '输入DPI', 36, 2400)
    destination_dpi = number(target_dpi, '输出DPI', 36, 2400)
    height, width = image.shape[:2]
    if width_mm is None:
        scale = destination_dpi / original_dpi
    else:
        physical_width = number(width_mm, '印刷宽度', 1, 2000)
        scale = physical_width / 25.4 * destination_dpi / width
    target_width = max(2, round(width * scale))
    target_height = max(2, round(height * scale))
    from .pixels import resize_rgb
    output = resize_rgb(image, target_width, target_height, method)
    return {'summary': {'source_dpi': original_dpi, 'target_dpi': destination_dpi, 'source_size': [width, height], 'output_size': [target_width, target_height], 'output_width_mm': target_width / destination_dpi * 25.4, 'output_height_mm': target_height / destination_dpi * 25.4, 'resampling': method}, 'images': {'印刷重采样': output}, 'tables': {}}


def poster_preview(rgb, paper_rgb=(0.97, 0.96, 0.93), contrast=0.95, texture=0.015, seed=1):
    image = rgb_image(rgb)
    if len(paper_rgb) != 3:
        raise ValueError('纸色需要三个RGB分量')
    paper = np.asarray([number(x, '纸色', 0.4, 1) for x in paper_rgb], dtype=np.float32)
    factor = number(contrast, '反差', 0.5, 1.5)
    grain = number(texture, '纸张纹理量', 0, 0.1)
    rng = np.random.default_rng(integer(seed, '纸纹种子', 0, 2 ** 31 - 1))
    linear = srgb_to_linear(image)
    simulated = np.clip((linear - 0.5) * factor + 0.5, 0, 1)
    simulated *= srgb_to_linear(paper)
    noise = rng.normal(0, grain, image.shape[:2]).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.6)
    simulated *= np.clip(1 + noise[:, :, None], 0.5, 1.5)
    output = linear_to_srgb(simulated)
    return {'summary': {'paper_rgb': paper.tolist(), 'contrast': factor, 'texture': grain, 'seed': seed, 'simulation': '视觉演示近似，不是印刷色彩合同打样'}, 'images': {'纸张观感预览': output}, 'tables': {}}


def local_contrast(rgb, clip_limit=2, grid=8, strength=0.5):
    image = rgb_image(rgb)
    limit = number(clip_limit, '局部反差限幅', 0.1, 10)
    cells = integer(grid, '网格数', 2, 32)
    alpha = number(strength, '增强强度', 0, 1)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    original_light = lab[:, :, 0] / 100
    encoded = np.rint(original_light * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(cells, cells))
    equalized = clahe.apply(encoded).astype(np.float32) / 255
    lab[:, :, 0] = (original_light * (1 - alpha) + equalized * alpha) * 100
    output = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
    return {'summary': {'clip_limit': limit, 'grid': cells, 'strength': alpha, 'lightness_change_mean': float(np.mean(np.abs(equalized - original_light)) * alpha)}, 'images': {'局部反差': output, '亮度调整': np.clip((equalized - original_light) * alpha + 0.5, 0, 1)}, 'tables': {}}


def trim_whitespace(rgb, tolerance=0.03, padding=8, background=(1, 1, 1)):
    image = rgb_image(rgb)
    threshold = number(tolerance, '纸色容差', 0, 0.5)
    border = integer(padding, '保留边距', 0, 256)
    color = np.asarray([number(x, '纸色', 0, 1) for x in background], dtype=np.float32)
    if color.shape != (3,):
        raise ValueError('纸色需要RGB')
    distance = np.max(np.abs(image - color), axis=2)
    content = distance > threshold
    ys, xs = np.nonzero(content)
    if not len(xs):
        raise ValueError('未找到超出纸色容差的图像内容')
    height, width = image.shape[:2]
    left = max(0, int(xs.min()) - border)
    top = max(0, int(ys.min()) - border)
    right = min(width, int(xs.max()) + border + 1)
    bottom = min(height, int(ys.max()) + border + 1)
    result = image[top:bottom, left:right].copy()
    return {'summary': {'tolerance': threshold, 'padding': border, 'crop': [left, top, right - left, bottom - top], 'removed_pixels': width * height - (right - left) * (bottom - top)}, 'images': {'去白边图像': result, '内容选区': content.astype(np.float32)}, 'tables': {}}


def color_border(rgb, width=8, color=(0, 0, 0), inside=True):
    image = rgb_image(rgb)
    thickness = integer(width, '边框宽度', 1, 128)
    tint = np.asarray([number(x, '边框颜色', 0, 1) for x in color], dtype=np.float32)
    if tint.shape != (3,):
        raise ValueError('边框颜色需要RGB')
    if inside:
        if thickness * 2 >= min(image.shape[:2]):
            raise ValueError('内框不能覆盖整个图像')
        output = image.copy()
        output[:thickness] = tint
        output[-thickness:] = tint
        output[:, :thickness] = tint
        output[:, -thickness:] = tint
    else:
        from .geometry import add_bleed
        output = add_bleed(image, thickness, 'constant', color, False)['images']['出血图像']
    return {'summary': {'width': thickness, 'color': tint.tolist(), 'inside': bool(inside)}, 'images': {'边框图像': output}, 'tables': {}}


def threshold_halftone_preview(rgb, threshold=0.5, invert=False):
    image = rgb_image(rgb)
    cutoff = number(threshold, '二值阈值', 0, 1)
    light = luminance(image)
    binary = (light >= cutoff).astype(np.float32)
    if invert:
        binary = 1 - binary
    output = np.repeat(binary[:, :, None], 3, axis=2)
    transitions_x = np.count_nonzero(binary[:, 1:] != binary[:, :-1])
    transitions_y = np.count_nonzero(binary[1:, :] != binary[:-1, :])
    return {'summary': {'threshold': cutoff, 'invert': bool(invert), 'black_fraction': float(np.mean(binary == 0)), 'horizontal_transitions': int(transitions_x), 'vertical_transitions': int(transitions_y)}, 'images': {'二值线稿': output}, 'tables': {'输入亮度分布': histogram(light)}}
