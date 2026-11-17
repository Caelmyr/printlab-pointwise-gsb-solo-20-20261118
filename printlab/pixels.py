import io
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PIXELS = 4_194_304
MAX_DIMENSION = 4096
MAX_FILE_BYTES = 24 * 1024 * 1024
CHANNELS = ('C', 'M', 'Y', 'K')


def number(value, name, minimum=None, maximum=None, default=None):
    if value is None:
        value = default
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError(f'{name}需要数字')
    if not math.isfinite(result):
        raise ValueError(f'{name}必须是有限数字')
    if minimum is not None and result < minimum:
        raise ValueError(f'{name}不能小于{minimum}')
    if maximum is not None and result > maximum:
        raise ValueError(f'{name}不能大于{maximum}')
    return result


def integer(value, name, minimum=None, maximum=None, default=None):
    result = number(value, name, minimum, maximum, default)
    if result != int(result):
        raise ValueError(f'{name}需要整数')
    return int(result)


def choice(value, name, choices, default=None):
    candidate = default if value is None else value
    if candidate not in choices:
        raise ValueError(f'{name}必须为' + '、'.join(str(x) for x in choices))
    return candidate


def rgb_image(image):
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError('需要三通道RGB图像')
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError('图像宽高至少为2')
    if array.shape[0] * array.shape[1] > MAX_PIXELS:
        raise ValueError('图像最多支持419万像素')
    if not np.all(np.isfinite(array)):
        raise ValueError('图像包含非有限值')
    if np.min(array) < 0 or np.max(array) > 1:
        raise ValueError('RGB像素值必须在0到1之间')
    return np.ascontiguousarray(array)


def cmyk_image(image):
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError('需要四通道CMYK分色图')
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError('分色图宽高至少为2')
    if array.shape[0] * array.shape[1] > MAX_PIXELS:
        raise ValueError('分色图超过像素预算')
    if not np.all(np.isfinite(array)):
        raise ValueError('分色图包含非有限值')
    if np.min(array) < 0 or np.max(array) > 1:
        raise ValueError('分色覆盖率必须在0到1之间')
    return np.ascontiguousarray(array)


def gray_image(image):
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[:, :, 0]
        elif array.shape[2] == 3:
            array = luminance(array)
        else:
            raise ValueError('灰度输入不支持该通道数')
    if array.ndim != 2:
        raise ValueError('需要二维灰度图')
    if not np.all(np.isfinite(array)):
        raise ValueError('灰度图包含非有限值')
    if array.shape[0] * array.shape[1] > MAX_PIXELS:
        raise ValueError('图像超过像素预算')
    return np.clip(array, 0, 1)


def mask_image(mask, shape):
    if mask is None:
        return np.ones(shape[:2], dtype=bool)
    array = np.asarray(mask)
    if array.shape != tuple(shape[:2]):
        raise ValueError('区域掩码和图像大小不同')
    return array.astype(bool)


def read_image(data, matte=(1.0, 1.0, 1.0)):
    if isinstance(data, (str, Path)):
        source = Path(data)
        if source.stat().st_size > MAX_FILE_BYTES:
            raise ValueError('文件超过24 MiB')
        payload = source.read_bytes()
    else:
        payload = bytes(data)
    if not payload:
        raise ValueError('图片内容为空')
    if len(payload) > MAX_FILE_BYTES:
        raise ValueError('文件超过24 MiB')
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            if opened.width * opened.height > MAX_PIXELS:
                raise ValueError('图片最多支持419万像素')
            if max(opened.size) > MAX_DIMENSION:
                raise ValueError('图片单边最多4096像素')
            original_mode = opened.mode
            original_size = list(opened.size)
            original_format = opened.format
            frame_count = getattr(opened, 'n_frames', 1)
            if frame_count != 1:
                raise ValueError('请上传单帧图像')
            dpi = opened.info.get('dpi', (300, 300))
            icc = bool(opened.info.get('icc_profile'))
            oriented = ImageOps.exif_transpose(opened)
            rgba = np.asarray(oriented.convert('RGBA'), dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('无法读取图片，请使用PNG/JPEG/TIFF') from exc
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3] * alpha[:, :, None]
    matte_value = np.clip(np.asarray(matte, dtype=np.float32), 0, 1)
    if matte_value.shape != (3,):
        raise ValueError('底纸颜色需要三个分量')
    rgb += matte_value * (1 - alpha[:, :, None])
    try:
        resolution = [float(dpi[0]), float(dpi[1])]
    except (TypeError, ValueError, IndexError):
        resolution = [300.0, 300.0]
    resolution = [x if math.isfinite(x) and x > 0 else 300.0 for x in resolution]
    metadata = {
        'original_mode': original_mode,
        'original_size': original_size,
        'format': original_format,
        'width': int(rgb.shape[1]),
        'height': int(rgb.shape[0]),
        'dpi': resolution,
        'has_icc_profile': icc,
        'icc_handling': 'RGB数值工作流；未执行ICC转换',
        'alpha_pixels': int(np.count_nonzero(alpha < 1)),
        'matte': matte_value.tolist(),
        'exif_orientation_applied': True,
    }
    return np.ascontiguousarray(rgb), alpha, metadata


def png_bytes(array):
    values = np.asarray(array)
    if values.dtype == bool:
        values = values.astype(np.uint8) * 255
    elif np.issubdtype(values.dtype, np.floating):
        if not np.all(np.isfinite(values)):
            raise ValueError('不能导出非有限像素')
        values = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
    elif values.dtype != np.uint8:
        values = np.clip(values, 0, 255).astype(np.uint8)
    if values.ndim == 3 and values.shape[2] == 1:
        values = values[:, :, 0]
    if values.ndim not in (2, 3):
        raise ValueError('导出图像维度无效')
    if values.ndim == 3 and values.shape[2] not in (3, 4):
        raise ValueError('PNG只支持灰度、RGB或RGBA')
    buffer = io.BytesIO()
    Image.fromarray(values).save(buffer, format='PNG', compress_level=4)
    return buffer.getvalue()


def tiff_bytes(array, dpi=300):
    values = np.asarray(array)
    if values.ndim == 3 and values.shape[2] == 4:
        output = Image.fromarray(np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8), 'CMYK')
    else:
        output = Image.open(io.BytesIO(png_bytes(values)))
    resolution = number(dpi, '分辨率', 36, 2400)
    buffer = io.BytesIO()
    output.save(buffer, format='TIFF', compression='tiff_deflate', dpi=(resolution, resolution))
    return buffer.getvalue()


def thumbnail(array, maximum=720):
    values = np.asarray(array)
    height, width = values.shape[:2]
    scale = min(1.0, maximum / max(width, height))
    if scale == 1.0:
        return values.copy()
    return cv2.resize(values, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def luminance(rgb):
    values = np.asarray(rgb, dtype=np.float32)
    return values[..., 0] * 0.2126 + values[..., 1] * 0.7152 + values[..., 2] * 0.0722


def srgb_to_linear(rgb):
    values = np.asarray(rgb, dtype=np.float32)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    values = np.clip(np.asarray(rgb, dtype=np.float32), 0, 1)
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * values ** (1 / 2.4) - 0.055)


def rgb_to_lab(rgb):
    values = rgb_image(rgb)
    return cv2.cvtColor(values, cv2.COLOR_RGB2LAB)


def lab_to_rgb(lab):
    values = np.asarray(lab, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError('Lab需要三通道')
    return np.clip(cv2.cvtColor(values, cv2.COLOR_LAB2RGB), 0, 1)


def delta_e76(left, right):
    first = rgb_to_lab(left)
    second = rgb_to_lab(right)
    if first.shape != second.shape:
        raise ValueError('比较图像大小不同')
    return np.linalg.norm(first - second, axis=2)


def heatmap(values, maximum=None, minimum=0):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError('热图需要二维数值')
    finite = np.isfinite(array)
    if maximum is None:
        maximum = float(np.percentile(array[finite], 99)) if np.any(finite) else 1.0
    span = max(float(maximum) - float(minimum), 1e-9)
    normalized = np.clip((np.where(finite, array, minimum) - minimum) / span, 0, 1)
    mapped = cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    mapped = cv2.cvtColor(mapped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    mapped[~finite] = (0.4, 0.4, 0.4)
    return mapped


def overlay_mask(rgb, mask, color=(1, 0.1, 0.05), opacity=0.5):
    values = rgb_image(rgb).copy()
    selected = mask_image(mask, values.shape)
    alpha = number(opacity, '叠加透明度', 0, 1)
    tint = np.asarray(color, dtype=np.float32)
    if tint.shape != (3,):
        raise ValueError('叠加色需要RGB三分量')
    values[selected] = values[selected] * (1 - alpha) + tint * alpha
    return np.clip(values, 0, 1)


def percentile_statistics(values, mask=None):
    array = np.asarray(values, dtype=np.float64)
    if mask is not None:
        array = array[np.asarray(mask, dtype=bool)]
    array = array[np.isfinite(array)]
    if not array.size:
        return {'count': 0, 'mean': None, 'std': None, 'min': None, 'max': None, 'median': None, 'p05': None, 'p95': None}
    q = np.percentile(array, [5, 50, 95])
    return {
        'count': int(array.size),
        'mean': float(np.mean(array)),
        'std': float(np.std(array)),
        'min': float(np.min(array)),
        'max': float(np.max(array)),
        'median': float(q[1]),
        'p05': float(q[0]),
        'p95': float(q[2]),
    }


def histogram(values, bins=32, mask=None, range_min=0, range_max=1):
    count = integer(bins, '直方图区间数', 4, 256)
    array = np.asarray(values, dtype=np.float64)
    if mask is not None:
        array = array[np.asarray(mask, dtype=bool)]
    array = array[np.isfinite(array)]
    counts, edges = np.histogram(array, bins=count, range=(range_min, range_max))
    denominator = max(int(counts.sum()), 1)
    return [{'lower': float(edges[i]), 'upper': float(edges[i + 1]), 'count': int(counts[i]), 'fraction': float(counts[i] / denominator)} for i in range(count)]


def polygon_mask(shape, points):
    if not isinstance(points, list) or len(points) < 3 or len(points) > 256:
        raise ValueError('区域多边形需要3到256个点')
    height, width = shape[:2]
    converted = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError('区域点需要x和y')
        x = number(point[0], 'x', 0, width - 1)
        y = number(point[1], 'y', 0, height - 1)
        converted.append([round(x), round(y)])
    contour = np.asarray(converted, dtype=np.int32)
    if abs(cv2.contourArea(contour)) < 1:
        raise ValueError('区域没有有效面积')
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 1)
    return mask.astype(bool)


def rectangle_mask(shape, x, y, width, height):
    image_height, image_width = shape[:2]
    left = integer(x, '区域x', 0, image_width - 1)
    top = integer(y, '区域y', 0, image_height - 1)
    region_width = integer(width, '区域宽度', 1, image_width)
    region_height = integer(height, '区域高度', 1, image_height)
    if left + region_width > image_width or top + region_height > image_height:
        raise ValueError('区域不能超出图像')
    mask = np.zeros((image_height, image_width), dtype=bool)
    mask[top:top + region_height, left:left + region_width] = True
    return mask


def mask_bounds(mask):
    selected = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(selected)
    if not len(xs):
        return None
    return {'x': int(xs.min()), 'y': int(ys.min()), 'width': int(xs.max() - xs.min() + 1), 'height': int(ys.max() - ys.min() + 1)}


def mask_outline(rgb, mask, color=(1, 0.2, 0.1)):
    values = rgb_image(rgb).copy()
    selected = mask_image(mask, values.shape).astype(np.uint8)
    contours, _ = cv2.findContours(selected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(values, contours, -1, tuple(float(x) for x in color), 1)
    return values


def checkerboard(shape, cell=16):
    height, width = shape[:2]
    yy, xx = np.indices((height, width))
    grid = ((xx // cell + yy // cell) % 2).astype(np.float32)
    values = 0.75 + 0.2 * grid
    return np.repeat(values[:, :, None], 3, axis=2)


def alpha_preview(rgb, alpha):
    values = rgb_image(rgb)
    opacity = np.asarray(alpha, dtype=np.float32)
    if opacity.shape != values.shape[:2]:
        raise ValueError('透明度与图像大小不同')
    background = checkerboard(values.shape)
    return values * opacity[:, :, None] + background * (1 - opacity[:, :, None])


def channel_preview(plane, channel):
    coverage = gray_image(plane)
    ink_rgb = {'C': (0, 1, 1), 'M': (1, 0, 1), 'Y': (1, 1, 0), 'K': (0, 0, 0)}
    color = np.asarray(ink_rgb[choice(channel, '色版', CHANNELS)], dtype=np.float32)
    return 1 - coverage[:, :, None] * (1 - color)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def resize_rgb(rgb, width, height, method='area'):
    values = rgb_image(rgb)
    target_width = integer(width, '目标宽度', 2, MAX_DIMENSION)
    target_height = integer(height, '目标高度', 2, MAX_DIMENSION)
    if target_width * target_height > MAX_PIXELS:
        raise ValueError('缩放后超过像素预算')
    methods = {'area': cv2.INTER_AREA, 'linear': cv2.INTER_LINEAR, 'cubic': cv2.INTER_CUBIC, 'nearest': cv2.INTER_NEAREST}
    interpolation = methods[choice(method, '采样方式', methods)]
    return np.clip(cv2.resize(values, (target_width, target_height), interpolation=interpolation), 0, 1)


def image_facts(rgb, dpi=300):
    values = rgb_image(rgb)
    resolution = number(dpi, '分辨率', 36, 2400)
    height, width = values.shape[:2]
    light = luminance(values)
    saturated = (values.max(axis=2) - values.min(axis=2)) > 0.15
    return {
        'width': width,
        'height': height,
        'pixels': width * height,
        'dpi': resolution,
        'width_mm': width / resolution * 25.4,
        'height_mm': height / resolution * 25.4,
        'mean_rgb': values.mean(axis=(0, 1)).tolist(),
        'luminance': percentile_statistics(light),
        'colorful_fraction': float(np.mean(saturated)),
        'white_fraction': float(np.mean(np.all(values > 0.98, axis=2))),
        'black_fraction': float(np.mean(np.all(values < 0.02, axis=2))),
    }
