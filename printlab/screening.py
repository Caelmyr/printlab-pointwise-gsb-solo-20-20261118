"""Raster plate screening and spatial-frequency checks for a print proof."""

import math
from functools import lru_cache

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


CHANNELS = ("C", "M", "Y", "K")
DEFAULT_ANGLES = (15.0, 75.0, 0.0, 45.0)
INK_RGB = np.array([[0.10, 0.85, 0.91], [0.89, 0.12, 0.52], [0.98, 0.91, 0.08], [0.12, 0.12, 0.12]], dtype=np.float32)


def _unit_array(value, channels, name):
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != channels:
        raise ValueError(f"{name} 必须为 H×W×{channels} 数组")
    if min(arr.shape[:2]) < 2:
        raise ValueError("图像宽高至少为 2 像素")
    if arr.shape[0] * arr.shape[1] > 24_000_000:
        raise ValueError("本工作台一次处理不超过 2400 万像素")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} 含有非有限数值")
    if float(arr.min()) < -0.00001 or float(arr.max()) > 1.00001:
        raise ValueError(f"{name} 的值应在 0 到 1 之间")
    return np.clip(arr, 0.0, 1.0)


def _rgb(value):
    return _unit_array(value, 3, "RGB 图像")


def _cmyk(value):
    return _unit_array(value, 4, "CMYK 分色")


def _mask(value, shape):
    if value is None:
        return np.ones(shape, dtype=bool)
    result = np.asarray(value, dtype=bool)
    if result.shape != shape:
        raise ValueError("有效区域尺寸必须与图像相同")
    if not result.any():
        raise ValueError("有效区域不能全部为空")
    return result


def _positive(value, name, minimum, maximum):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须为数值") from exc
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise ValueError(f"{name} 应在 {minimum} 到 {maximum} 之间")
    return result


def _integer(value, name, minimum, maximum):
    number = _positive(value, name, minimum, maximum)
    if int(number) != number:
        raise ValueError(f"{name} 必须为整数")
    return int(number)


def _luminance(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _colorize_plates(plates):
    result = np.ones(plates.shape[:2] + (3,), dtype=np.float32)
    for index in range(4):
        result *= 1.0 - plates[..., index, None] * (1.0 - INK_RGB[index])
    return np.clip(result, 0.0, 1.0)


def _heatmap(data, valid=None):
    values = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(values)
    if valid is not None:
        finite &= valid
    scaled = np.zeros(values.shape, dtype=np.uint8)
    if finite.any():
        high = float(np.percentile(values[finite], 99))
        low = float(np.percentile(values[finite], 1))
        if high > low:
            scaled[finite] = np.clip((values[finite] - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    colored[~finite] = 0.8
    return colored


def _draw_text(image, text, position, color=(0.1, 0.1, 0.1)):
    output = np.ascontiguousarray(image.copy())
    _put_text(output, str(text), position, cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return output


def _put_text(image, text, position, font, scale, color, thickness=1, line_type=cv2.LINE_AA):
    (text_width, text_height), baseline = cv2.getTextSize(str(text), font, scale, thickness)
    glyph = np.zeros((text_height + baseline + 4, text_width + 4), dtype=np.uint8)
    cv2.putText(glyph, str(text), (2, text_height + 1), font, scale, 255, thickness, line_type)
    x = int(position[0]) - 2
    y = int(position[1]) - text_height - 1
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(image.shape[1], x + glyph.shape[1])
    y1 = min(image.shape[0], y + glyph.shape[0])
    if x1 <= x0 or y1 <= y0:
        return image
    alpha = glyph[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)[..., None] / 255.0
    image[y0:y1, x0:x1] = image[y0:y1, x0:x1] * (1.0 - alpha) + np.asarray(color, dtype=np.float32) * alpha
    return image


@lru_cache(maxsize=2)
def _rank_reference(shape):
    grid = np.linspace(0.0, 1.0, 384, dtype=np.float32)
    gx, gy = np.meshgrid(grid, grid)
    if shape == "round":
        return np.sort(np.sqrt(gx * gx + gy * gy).ravel())
    return np.sort(np.sqrt((gx / 1.35) ** 2 + (gy * 1.35) ** 2).ravel())


def _threshold_rank(u, v, shape):
    x = np.abs(u - np.floor(u) - 0.5) * 2.0
    y = np.abs(v - np.floor(v) - 0.5) * 2.0
    if shape == "round":
        radius = np.sqrt(x * x + y * y)
        reference = _rank_reference(shape)
        return np.searchsorted(reference, radius, side="right") / float(reference.size)
    if shape == "ellipse":
        radius = np.sqrt((x / 1.35) ** 2 + (y * 1.35) ** 2)
        reference = _rank_reference(shape)
        return np.searchsorted(reference, radius, side="right") / float(reference.size)
    if shape == "diamond":
        total = x + y
        return np.where(total <= 1.0, total * total / 2.0, 1.0 - (2.0 - total) ** 2 / 2.0)
    if shape == "line":
        return y
    if shape == "square":
        return np.maximum(x, y) ** 2
    raise ValueError("网点形状可选 round、ellipse、diamond、line 或 square")


def _angles(value):
    if value is None:
        return np.array(DEFAULT_ANGLES, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4,) or not np.isfinite(arr).all():
        raise ValueError("须为 C、M、Y、K 分别提供一个网角")
    return np.mod(arr, 180.0)


def _phase(value):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (2,):
        arr = np.repeat(arr[None, :], 4, axis=0)
    if arr.shape != (4, 2) or not np.isfinite(arr).all():
        raise ValueError("网点相位须为一个 XY 对或四个 XY 对")
    return np.mod(arr, 1.0)


def render_am(cmyk, lpi=45.0, dpi=300.0, angles=None, shape="ellipse", supersample=2, phase=(0.0, 0.0), mask=None):
    plates = _cmyk(cmyk)
    valid = _mask(mask, plates.shape[:2])
    dpi = _positive(dpi, "输出 DPI", 36.0, 4800.0)
    lpi = _positive(lpi, "加网线数", 1.0, 600.0)
    sample_count = _integer(supersample, "子采样数量", 1, 4)
    if lpi > dpi / 2.0:
        raise ValueError("线数不能高于输出像素频率的一半")
    plate_angles = _angles(angles)
    phases = _phase(phase)
    if shape not in {"round", "ellipse", "diamond", "line", "square"}:
        raise ValueError("不支持的网点形状")
    height, width = plates.shape[:2]
    period = dpi / lpi
    output = np.zeros_like(plates)
    chunk_rows = max(8, min(256, 1_000_000 // max(1, width)))
    offsets = (np.arange(sample_count, dtype=np.float32) + 0.5) / sample_count
    xbase = np.arange(width, dtype=np.float32)[None, :]
    for channel in range(4):
        theta = math.radians(float(plate_angles[channel]))
        cosine = math.cos(theta)
        sine = math.sin(theta)
        for top in range(0, height, chunk_rows):
            bottom = min(height, top + chunk_rows)
            ybase = np.arange(top, bottom, dtype=np.float32)[:, None]
            accumulated = np.zeros((bottom - top, width), dtype=np.float32)
            coverage = plates[top:bottom, :, channel]
            for dy in offsets:
                for dx in offsets:
                    u = ((xbase + dx) * cosine + (ybase + dy) * sine) / period + phases[channel, 0]
                    v = (-(xbase + dx) * sine + (ybase + dy) * cosine) / period + phases[channel, 1]
                    threshold = _threshold_rank(u, v, shape)
                    selected = threshold < coverage
                    selected[coverage <= 0.0] = False
                    selected[coverage >= 1.0] = True
                    accumulated += selected
            output[top:bottom, :, channel] = accumulated / (sample_count * sample_count)
    output[~valid] = 0.0
    records = []
    for channel, label in enumerate(CHANNELS):
        requested = float(plates[..., channel][valid].mean())
        actual = float(output[..., channel][valid].mean())
        records.append({"channel": label, "angle_deg": float(plate_angles[channel]), "requested_coverage": requested, "screened_coverage": actual, "coverage_error": actual - requested, "phase_x": float(phases[channel, 0]), "phase_y": float(phases[channel, 1])})
    images = {"composite": _colorize_plates(output), "coverage": output}
    for channel, label in enumerate(CHANNELS):
        images[f"plate_{label}"] = np.repeat((1.0 - output[..., channel, None]), 3, axis=2)
    return {"summary": {"method": "AM", "shape": shape, "dpi": dpi, "lpi": lpi, "period_px": period, "supersample": sample_count, "width": width, "height": height, "valid_pixels": int(valid.sum()), "approximation": "每像素子采样的旋转阈值网点；小网胞有离散面积误差"}, "images": images, "tables": {"plates": records}}


def _noise_threshold(height, width, seed, dot_size):
    cell_height = math.ceil(height / dot_size)
    cell_width = math.ceil(width / dot_size)
    random = np.random.default_rng(seed)
    count = cell_height * cell_width
    order = random.permutation(count)
    ranked = (order.astype(np.float64) + 0.5) / count
    ranked = ranked.reshape(cell_height, cell_width).astype(np.float32)
    if dot_size > 1:
        ranked = np.repeat(np.repeat(ranked, dot_size, axis=0), dot_size, axis=1)
    return ranked[:height, :width]


def render_fm(cmyk, seed=42, dot_size=1, mask=None, shared_noise=False):
    plates = _cmyk(cmyk)
    valid = _mask(mask, plates.shape[:2])
    dot_size = _integer(dot_size, "点块边长", 1, 16)
    seed = _integer(seed, "随机种子", 0, 2**32 - 1)
    height, width = plates.shape[:2]
    output = np.zeros_like(plates)
    records = []
    shared = None
    if shared_noise:
        shared = _noise_threshold(height, width, seed, dot_size)
    for channel, label in enumerate(CHANNELS):
        channel_seed = seed if shared_noise else (seed + channel * 104729) % (2**32)
        threshold = shared if shared is not None else _noise_threshold(height, width, channel_seed, dot_size)
        coverage = plates[..., channel]
        if dot_size > 1:
            small = cv2.resize(coverage, (math.ceil(width / dot_size), math.ceil(height / dot_size)), interpolation=cv2.INTER_AREA)
            coverage = np.repeat(np.repeat(small, dot_size, axis=0), dot_size, axis=1)[:height, :width]
        selected = threshold < coverage
        selected[plates[..., channel] <= 0.0] = False
        selected[plates[..., channel] >= 1.0] = True
        output[..., channel] = selected & valid
        requested = float(plates[..., channel][valid].mean())
        actual = float(output[..., channel][valid].mean())
        labels, count = ndimage.label(output[..., channel] > 0.5)
        sizes = np.bincount(labels.ravel())[1:]
        records.append({"channel": label, "requested_coverage": requested, "screened_coverage": actual, "coverage_error": actual - requested, "connected_ink_regions": int(count), "median_region_area_px": float(np.median(sizes)) if sizes.size else 0.0, "seed": channel_seed})
    images = {"composite": _colorize_plates(output), "coverage": output}
    for channel, label in enumerate(CHANNELS):
        images[f"plate_{label}"] = np.repeat(1.0 - output[..., channel, None], 3, axis=2)
    return {"summary": {"method": "FM_random_threshold", "dot_size_px": dot_size, "seed": seed, "shared_noise": bool(shared_noise), "valid_pixels": int(valid.sum()), "approximation": "等秩随机阈值；不是工业蓝噪声制版算法，点块边缘保留零墨/实地区域"}, "images": images, "tables": {"plates": records}}


def _frequency_peak_rows(power, dpi, max_peaks, exclusion_radius):
    height, width = power.shape
    center_y = height // 2
    center_x = width // 2
    yy, xx = np.indices(power.shape)
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    local_max = ndimage.maximum_filter(power, size=5, mode="nearest")
    candidate = (power == local_max) & (radius >= exclusion_radius)
    candidate &= ((yy > center_y) | ((yy == center_y) & (xx > center_x)))
    positions = np.argwhere(candidate)
    if positions.size == 0:
        return []
    values = power[candidate]
    indices = np.argsort(values)[::-1]
    output = []
    accepted = []
    total = float(power.sum())
    noise = float(np.median(power[radius >= exclusion_radius]))
    for index in indices:
        row, col = positions[index]
        if any((row - ay) ** 2 + (col - ax) ** 2 < 25 for ay, ax in accepted):
            continue
        value = float(power[row, col])
        if value <= noise * 2.0 or value <= 0.0:
            continue
        fy = (row - center_y) / height
        fx = (col - center_x) / width
        frequency = math.hypot(float(fx), float(fy))
        accepted.append((int(row), int(col)))
        output.append({"rank": len(output) + 1, "frequency_cycles_per_pixel": frequency, "frequency_lpi": frequency * dpi, "period_px": 1.0 / frequency, "angle_deg": math.degrees(math.atan2(fy, fx)) % 180.0, "power_fraction": value / total if total else 0.0, "power_over_median": value / max(noise, 1e-15), "spectrum_x": int(col), "spectrum_y": int(row)})
        if len(output) >= max_peaks:
            break
    return output


def spectrum_analysis(rgb, dpi=300.0, max_peaks=12, window=True, channel="luminance"):
    source = _rgb(rgb)
    dpi = _positive(dpi, "图像 DPI", 36.0, 4800.0)
    max_peaks = _integer(max_peaks, "峰值数量", 1, 40)
    if channel == "luminance":
        signal = _luminance(source)
    elif channel in ("R", "G", "B"):
        signal = source[..., ("R", "G", "B").index(channel)]
    else:
        raise ValueError("频域通道须为 luminance、R、G 或 B")
    original_height, original_width = signal.shape
    scale = min(1.0, 2048.0 / max(signal.shape))
    if scale < 1.0:
        signal = cv2.resize(signal, (max(2, round(original_width * scale)), max(2, round(original_height * scale))), interpolation=cv2.INTER_AREA)
    effective_dpi = dpi * scale
    height, width = signal.shape
    centered = signal.astype(np.float64) - float(signal.mean())
    if window:
        centered *= np.outer(np.hanning(height), np.hanning(width))
    transform = np.fft.fftshift(np.fft.fft2(centered))
    power = np.abs(transform) ** 2
    exclusion = max(3, round(min(height, width) / 150))
    yy, xx = np.indices(power.shape)
    radius = np.sqrt((yy - height // 2) ** 2 + (xx - width // 2) ** 2)
    power[radius < exclusion] = 0.0
    rows = _frequency_peak_rows(power, effective_dpi, max_peaks, exclusion)
    view = _heatmap(np.log1p(power))
    for item in rows:
        point = (item["spectrum_x"], item["spectrum_y"])
        cv2.circle(view, point, 5, (1.0, 1.0, 1.0), 1, cv2.LINE_AA)
    frequency_y = np.fft.fftshift(np.fft.fftfreq(height))[:, None]
    frequency_x = np.fft.fftshift(np.fft.fftfreq(width))[None, :]
    frequency_radius = np.sqrt(frequency_x**2 + frequency_y**2)
    edges = np.linspace(0.0, math.sqrt(0.5), 65)
    radial = []
    total_power = float(power.sum())
    for index in range(len(edges) - 1):
        selected = (frequency_radius >= edges[index]) & (frequency_radius < edges[index + 1])
        radial.append({"frequency_lpi": float((edges[index] + edges[index + 1]) * 0.5 * effective_dpi), "mean_power": float(power[selected].mean()) if selected.any() else 0.0, "power_fraction": float(power[selected].sum()) / max(total_power, 1e-15), "bins": int(selected.sum())})
    return {"summary": {"channel": channel, "dpi": dpi, "analysis_dpi": effective_dpi, "analysis_width": width, "analysis_height": height, "downsampled": scale < 1.0, "windowed": bool(window), "peak_count": len(rows), "signal_std": float(signal.std()), "limitation": "峰值仅说明周期能量，不自动判定网花或设备故障"}, "images": {"spectrum": view}, "tables": {"peaks": rows, "radial_power": radial}}


def _trend(values, sigma):
    if values.size < 3:
        return np.full(values.shape, float(values.mean()), dtype=np.float64)
    return ndimage.gaussian_filter1d(values.astype(np.float64), sigma=sigma, mode="reflect")


def _profile_peaks(residual, dpi, limit=10):
    if residual.size < 4 or float(np.max(np.abs(residual))) < 1e-10:
        return []
    windowed = residual * np.hanning(residual.size)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    frequencies = np.fft.rfftfreq(residual.size)
    power[0] = 0.0
    candidates = np.flatnonzero((power >= np.roll(power, 1)) & (power >= np.roll(power, -1)))
    candidates = candidates[candidates > 0]
    candidates = candidates[np.argsort(power[candidates])[::-1]]
    output = []
    total = max(float(power.sum()), 1e-15)
    for index in candidates[:limit]:
        frequency = float(frequencies[index])
        output.append({"rank": len(output) + 1, "period_px": 1.0 / frequency, "period_mm": 25.4 / (frequency * dpi), "frequency_lpi": frequency * dpi, "power_fraction": float(power[index]) / total})
    return output


def _profile_plot(profile, trend, residual, width=960, height=300):
    image = np.ones((height, width, 3), dtype=np.float32)
    cv2.line(image, (35, height - 25), (width - 15, height - 25), (0.7, 0.7, 0.7), 1)
    cv2.line(image, (35, 15), (35, height - 25), (0.7, 0.7, 0.7), 1)
    combined = np.concatenate([profile, trend])
    low = float(combined.min())
    high = float(combined.max())
    padding = max((high - low) * 0.1, 0.01)
    low -= padding
    high += padding
    for values, color in ((profile, (0.1, 0.4, 0.9)), (trend, (0.9, 0.35, 0.1))):
        x = np.linspace(35, width - 15, values.size)
        y = height - 25 - (values - low) / (high - low) * (height - 40)
        points = np.stack([x, y], axis=1).round().astype(np.int32)
        cv2.polylines(image, [points], False, color, 1, cv2.LINE_AA)
    image = _draw_text(image, f"mean profile / broad trend; residual RMS {np.sqrt(np.mean(residual**2)):.5f}", (45, 25))
    return image


def banding_analysis(rgb, axis="vertical", sigma=20.0, dpi=300.0, mask=None):
    source = _rgb(rgb)
    valid = _mask(mask, source.shape[:2])
    dpi = _positive(dpi, "图像 DPI", 36.0, 4800.0)
    sigma = _positive(sigma, "背景趋势尺度", 1.0, 500.0)
    if axis not in ("vertical", "horizontal"):
        raise ValueError("条带方向须为 vertical 或 horizontal")
    gray = _luminance(source)
    average_axis = 0 if axis == "vertical" else 1
    counts = valid.sum(axis=average_axis)
    profile = np.sum(gray * valid, axis=average_axis) / np.maximum(counts, 1)
    known = np.flatnonzero(counts > 0)
    if known.size < 4:
        raise ValueError("检测方向至少需要 4 个有数据的位置")
    profile = np.interp(np.arange(profile.size), known, profile[known])
    trend = _trend(profile, sigma)
    residual = profile - trend
    rms = float(np.sqrt(np.mean(residual[counts > 0] ** 2)))
    peak_to_peak = float(np.percentile(residual[counts > 0], 99) - np.percentile(residual[counts > 0], 1))
    records = []
    stride = max(1, math.ceil(profile.size / 4096))
    for index in range(0, profile.size, stride):
        records.append({"position_px": index, "position_mm": index * 25.4 / dpi, "mean_luminance": float(profile[index]), "trend": float(trend[index]), "residual": float(residual[index]), "valid_pixels": int(counts[index]), "interpolated": bool(counts[index] == 0)})
    if axis == "vertical":
        residual_map = np.broadcast_to(residual[None, :], gray.shape)
    else:
        residual_map = np.broadcast_to(residual[:, None], gray.shape)
    return {"summary": {"axis": axis, "residual_rms": rms, "residual_p99_p01": peak_to_peak, "profile_positions": int(profile.size), "missing_positions": int((counts == 0).sum()), "background_sigma_px": sigma, "limitation": "仅适合平坦色块；图案边缘也可产生周期信号"}, "images": {"residual": _heatmap(np.abs(residual_map), valid), "profile": _profile_plot(profile, trend, residual)}, "tables": {"profile": records, "periods": _profile_peaks(residual, dpi)}}


def _seam_direction(source, direction, band, tolerance):
    height, width = source.shape[:2]
    if direction == "horizontal":
        first = source[:, :band]
        last = source[:, -band:]
        direct = np.mean(np.abs(source[:, 0] - source[:, -1]), axis=1)
        gradient_first = source[:, 1] - source[:, 0]
        gradient_last = source[:, -1] - source[:, -2]
        jump = np.mean(np.abs(gradient_first - gradient_last), axis=1)
        band_error = np.mean(np.abs(first - last[:, ::-1]), axis=(1, 2))
        canvas = np.concatenate([last, first], axis=1)
    else:
        first = source[:band]
        last = source[-band:]
        direct = np.mean(np.abs(source[0] - source[-1]), axis=1)
        gradient_first = source[1] - source[0]
        gradient_last = source[-1] - source[-2]
        jump = np.mean(np.abs(gradient_first - gradient_last), axis=1)
        band_error = np.mean(np.abs(first - last[::-1]), axis=(0, 2))
        canvas = np.concatenate([last, first], axis=0)
    rows = []
    step = max(1, math.ceil(direct.size / 4096))
    for index in range(0, direct.size, step):
        rows.append({"direction": direction, "position_px": index, "edge_rgb_error": float(direct[index]), "slope_error": float(jump[index]), "mirrored_band_error": float(band_error[index]), "above_tolerance": bool(direct[index] > tolerance)})
    summary = {"direction": direction, "mean_edge_error": float(direct.mean()), "p95_edge_error": float(np.percentile(direct, 95)), "maximum_edge_error": float(direct.max()), "mean_slope_error": float(jump.mean()), "fraction_above_tolerance": float(np.mean(direct > tolerance)), "compared_positions": int(direct.size)}
    return summary, rows, canvas


def repeat_seam(rgb, axis="both", band=8, tolerance=0.04, repeat_count=2):
    source = _rgb(rgb)
    if axis not in ("both", "horizontal", "vertical"):
        raise ValueError("重复方向须为 both、horizontal 或 vertical")
    band = _integer(band, "接缝条带宽度", 1, min(source.shape[:2]) // 2)
    repeat_count = _integer(repeat_count, "预览重复次数", 2, 4)
    tolerance = _positive(tolerance, "边缘容差", 0.0, 1.0)
    summaries = []
    records = []
    images = {}
    for direction in ("horizontal", "vertical"):
        if axis != "both" and axis != direction:
            continue
        summary, rows, strip = _seam_direction(source, direction, band, tolerance)
        summaries.append(summary)
        records.extend(rows)
        images[f"seam_{direction}"] = strip
    height, width = source.shape[:2]
    max_side = 1800 // repeat_count
    scale = min(1.0, max_side / max(height, width))
    tile = source
    if scale < 1.0:
        tile = cv2.resize(source, (max(2, round(width * scale)), max(2, round(height * scale))), interpolation=cv2.INTER_AREA)
    repetitions = (repeat_count if axis != "horizontal" else 1, repeat_count if axis != "vertical" else 1, 1)
    repeated = np.tile(tile, repetitions)
    annotated = repeated.copy()
    for x in range(tile.shape[1], repeated.shape[1], tile.shape[1]):
        cv2.line(annotated, (x, 0), (x, repeated.shape[0] - 1), (0.95, 0.15, 0.2), 1)
    for y in range(tile.shape[0], repeated.shape[0], tile.shape[0]):
        cv2.line(annotated, (0, y), (repeated.shape[1] - 1, y), (0.95, 0.15, 0.2), 1)
    images["repeat_preview"] = repeated
    images["seam_locations"] = annotated
    return {"summary": {"axis": axis, "band_px": band, "tolerance": tolerance, "repeat_count": repeat_count, "preview_scale": scale, "directions": summaries, "limitation": "接缝检查不修改图像；纹理的有意突变仍需人工判断"}, "images": images, "tables": {"seam_profile": records, "directions": summaries}}


def screen_coverage(cmyk, screened, dpi=300.0, grid_rows=6, grid_columns=8, mask=None):
    target = _cmyk(cmyk)
    actual = _cmyk(screened)
    if target.shape != actual.shape:
        raise ValueError("目标分色与网版尺寸必须相同")
    valid = _mask(mask, target.shape[:2])
    dpi = _positive(dpi, "输出 DPI", 36.0, 4800.0)
    rows = _integer(grid_rows, "网格行数", 1, min(100, target.shape[0]))
    columns = _integer(grid_columns, "网格列数", 1, min(100, target.shape[1]))
    yedges = np.linspace(0, target.shape[0], rows + 1).astype(int)
    xedges = np.linspace(0, target.shape[1], columns + 1).astype(int)
    records = []
    errors = np.zeros_like(target)
    for row in range(rows):
        for column in range(columns):
            y0, y1 = yedges[row], yedges[row + 1]
            x0, x1 = xedges[column], xedges[column + 1]
            selected = valid[y0:y1, x0:x1]
            count = int(selected.sum())
            for channel, label in enumerate(CHANNELS):
                nominal = target[y0:y1, x0:x1, channel][selected]
                output = actual[y0:y1, x0:x1, channel][selected]
                error = float(output.mean() - nominal.mean()) if count else None
                if error is not None:
                    errors[y0:y1, x0:x1, channel] = abs(error)
                records.append({"row": row + 1, "column": column + 1, "channel": label, "x": int(x0), "y": int(y0), "width": int(x1 - x0), "height": int(y1 - y0), "valid_pixels": count, "target_coverage": float(nominal.mean()) if count else None, "screened_coverage": float(output.mean()) if count else None, "coverage_error": error})
    global_rows = []
    for channel, label in enumerate(CHANNELS):
        nominal = target[..., channel][valid]
        output = actual[..., channel][valid]
        difference = output - nominal
        binary_fraction = float(np.mean((output < 1e-6) | (output > 1.0 - 1e-6)))
        global_rows.append({"channel": label, "target_coverage": float(nominal.mean()), "screened_coverage": float(output.mean()), "coverage_error": float(difference.mean()), "binary_pixel_fraction": binary_fraction, "ink_area_mm2": float(output.sum() * (25.4 / dpi) ** 2)})
    return {"summary": {"valid_pixels": int(valid.sum()), "grid_rows": rows, "grid_columns": columns, "dpi": dpi, "maximum_local_coverage_error": float(errors.max())}, "images": {"local_coverage_error": _heatmap(errors.max(axis=2), valid)}, "tables": {"plates": global_rows, "cells": records}}


def dot_shape_analysis(coverage, channel="K", threshold=0.5, dpi=300.0, mask=None, min_area=2, exclude_boundary=True, max_dots=1000):
    plates = _cmyk(coverage)
    valid = _mask(mask, plates.shape[:2])
    if channel not in CHANNELS:
        raise ValueError("网点分析通道须为 C、M、Y 或 K")
    threshold = _positive(threshold, "网点阈值", 0.01, 0.99)
    dpi = _positive(dpi, "网版 DPI", 36.0, 4800.0)
    min_area = _integer(min_area, "最小网点面积", 1, 1_000_000)
    max_dots = _integer(max_dots, "最多测量网点", 1, 10000)
    plane = plates[..., CHANNELS.index(channel)]
    binary = (plane >= threshold) & valid
    component_count, labels, statistics, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    edge = valid & ~ndimage.binary_erosion(valid, iterations=1)
    boundary_labels = set(int(value) for value in np.unique(labels[edge]) if value)
    eligible = []
    small_count = 0
    boundary_count = 0
    for identifier in range(1, component_count):
        area = int(statistics[identifier, cv2.CC_STAT_AREA])
        if area < min_area:
            small_count += 1
            continue
        if exclude_boundary and identifier in boundary_labels:
            boundary_count += 1
            continue
        eligible.append(identifier)
    chosen = sorted(eligible, key=lambda identifier: (-int(statistics[identifier, cv2.CC_STAT_AREA]), identifier))[:max_dots]
    view = np.repeat((1.0 - plane[..., None]), 3, axis=2)
    table = []
    for identifier in chosen:
        x, y, width, height, area = map(int, statistics[identifier])
        local = labels[y:y + height, x:x + width] == identifier
        contours, hierarchy = cv2.findContours(local.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        external = [contour for index, contour in enumerate(contours) if hierarchy[0, index, 3] < 0]
        perimeter = sum(cv2.arcLength(contour, True) for contour in external)
        holes = int(np.sum(hierarchy[0, :, 3] >= 0)) if hierarchy is not None else 0
        ly, lx = np.nonzero(local)
        points = np.column_stack([lx, ly]).astype(np.float64)
        if len(points) >= 2:
            centered = points - points.mean(axis=0)
            covariance = centered.T @ centered / len(points)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 0.0)
            major = 4.0 * math.sqrt(float(eigenvalues[1]))
            minor = 4.0 * math.sqrt(float(eigenvalues[0]))
            vector = eigenvectors[:, 1]
            orientation = math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 180.0
        else:
            major = 1.0
            minor = 1.0
            orientation = 0.0
        distances = ndimage.distance_transform_edt(np.pad(local, 1))
        maximum_radius = float(distances.max())
        equivalent_diameter = 2.0 * math.sqrt(area / math.pi)
        antialias_area = float(plane[y:y + height, x:x + width][local].sum())
        table.append({
            "dot": len(table) + 1,
            "component_label": identifier,
            "x": x,
            "y": y,
            "width_px": width,
            "height_px": height,
            "centroid_x": float(centroids[identifier, 0]),
            "centroid_y": float(centroids[identifier, 1]),
            "threshold_area_px": area,
            "coverage_area_px": antialias_area,
            "equivalent_diameter_px": equivalent_diameter,
            "equivalent_diameter_um": equivalent_diameter * 25400.0 / dpi,
            "major_axis_px": major,
            "minor_axis_px": minor,
            "aspect_ratio": major / max(minor, 1e-6),
            "orientation_deg": orientation,
            "perimeter_px": float(perimeter),
            "circularity": min(1.0, 4.0 * math.pi * area / (perimeter * perimeter)) if perimeter else None,
            "maximum_inside_radius_px": maximum_radius,
            "holes": holes,
            "fill_fraction": area / (width * height),
            "touches_valid_boundary": identifier in boundary_labels,
        })
        color = (0.95, 0.15, 0.1) if holes else (0.1, 0.65, 0.2)
        for contour in external:
            shifted = contour + np.array([[[x, y]]], dtype=np.int32)
            cv2.drawContours(view, [shifted], -1, color, 1)
    if len(table) > 1:
        centers = np.array([[row["centroid_x"], row["centroid_y"]] for row in table])
        neighbors = cKDTree(centers).query(centers, k=2)[0][:, 1]
        for row, distance in zip(table, neighbors):
            row["nearest_reported_centroid_px"] = float(distance)
    else:
        for row in table:
            row["nearest_reported_centroid_px"] = None
    area_values = np.array([row["threshold_area_px"] for row in table], dtype=np.float64)
    size_rows = []
    if area_values.size:
        upper = max(2.0, float(area_values.max()) + 1.0)
        bins = np.geomspace(1.0, upper, min(25, len(area_values) + 1))
        counts, edges = np.histogram(area_values, bins=bins)
        for index, count in enumerate(counts):
            size_rows.append({"area_from_px": float(edges[index]), "area_to_px": float(edges[index + 1]), "dots": int(count)})
    summary = {
        "channel": channel,
        "threshold": threshold,
        "dpi": dpi,
        "connected_ink_regions": int(component_count - 1),
        "eligible_dots": len(eligible),
        "reported_dots": len(table),
        "omitted_for_limit": max(0, len(eligible) - max_dots),
        "excluded_small": small_count,
        "excluded_boundary": boundary_count,
        "mean_threshold_area_px": float(area_values.mean()) if area_values.size else None,
        "area_coefficient_of_variation": float(area_values.std() / area_values.mean()) if area_values.size else None,
        "threshold_coverage": float(binary.sum() / valid.sum()),
        "limitation": "测量阈值后的连通墨点；深色连片不是单个网点，最大数量截断时近邻仅在已报告网点中计算",
    }
    return {"summary": summary, "images": {"dot_outlines": view, "thresholded_plate": np.repeat((~binary)[..., None].astype(np.float32), 3, axis=2)}, "tables": {"dots": table, "area_histogram": size_rows}}
