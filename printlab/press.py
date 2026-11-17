"""Practical manual press controls and approximate subtractive ink previews."""

import math

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from .screening import _put_text


CHANNELS = ("C", "M", "Y", "K")
DEFAULT_INKS = ((0.10, 0.85, 0.91), (0.89, 0.12, 0.52), (0.98, 0.91, 0.08), (0.12, 0.12, 0.12))


def _image(value, channels):
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != channels:
        raise ValueError(f"图像应为 H×W×{channels} 数组")
    if min(arr.shape[:2]) < 2:
        raise ValueError("图像宽高至少为 2")
    if arr.shape[0] * arr.shape[1] > 24_000_000:
        raise ValueError("单张图像不超过 2400 万像素")
    if not np.isfinite(arr).all():
        raise ValueError("图像不能含非有限数值")
    if float(arr.min()) < -1e-5 or float(arr.max()) > 1.00001:
        raise ValueError("图像数值须在 0 到 1 之间")
    return np.clip(arr, 0.0, 1.0)


def _number(value, name, low, high):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须为数值") from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} 须在 {low} 到 {high} 之间")
    return number


def _integer(value, name, low, high):
    number = _number(value, name, low, high)
    if int(number) != number:
        raise ValueError(f"{name} 必须为整数")
    return int(number)


def _vector(value, count, name, low=0.0, high=1.0):
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(count, float(arr))
    if arr.shape != (count,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} 需要 {count} 个有限数值")
    if np.any(arr < low) or np.any(arr > high):
        raise ValueError(f"{name} 应在 {low} 到 {high} 之间")
    return arr


def _valid_mask(value, shape):
    if value is None:
        return np.ones(shape, dtype=bool)
    selected = np.asarray(value, dtype=bool)
    if selected.shape != shape:
        raise ValueError("有效区域尺寸不匹配")
    if not selected.any():
        raise ValueError("有效区域为空")
    return selected


def _simple_preview(cmyk, paper=(1.0, 1.0, 1.0), inks=DEFAULT_INKS):
    result = np.empty(cmyk.shape[:2] + (3,), dtype=np.float32)
    result[:] = paper
    for index in range(4):
        result *= 1.0 - cmyk[..., index, None] * (1.0 - np.asarray(inks[index]))
    return np.clip(result, 0.0, 1.0)


def _heatmap(values, maximum=None):
    data = np.asarray(values, dtype=np.float32)
    high = float(np.percentile(data, 99)) if maximum is None else float(maximum)
    if high <= 0.0:
        high = 1.0
    scaled = np.clip(data * 255.0 / high, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _curve_points(points, label):
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or not 2 <= len(arr) <= 64:
        raise ValueError(f"{label} 曲线需要 2 至 64 个输入/输出点")
    if not np.isfinite(arr).all() or np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{label} 曲线坐标须在 0 到 1 之间")
    order = np.argsort(arr[:, 0], kind="stable")
    arr = arr[order]
    if np.any(np.diff(arr[:, 0]) <= 0.0):
        raise ValueError(f"{label} 曲线输入坐标不能重复")
    if arr[0, 0] != 0.0 or arr[-1, 0] != 1.0:
        raise ValueError(f"{label} 曲线必须包含输入 0 和 1")
    if np.any(np.diff(arr[:, 1]) < 0.0):
        raise ValueError(f"{label} 曲线输出必须单调不减")
    return arr


def _curves(value):
    if value is None:
        return {label: np.array([[0.0, 0.0], [1.0, 1.0]]) for label in CHANNELS}
    if isinstance(value, dict):
        unknown = set(value) - set(CHANNELS)
        if unknown:
            raise ValueError("曲线通道只接受 C、M、Y、K")
        return {label: _curve_points(value.get(label, [[0.0, 0.0], [1.0, 1.0]]), label) for label in CHANNELS}
    if len(value) != 4:
        raise ValueError("需要提供四条通道曲线")
    return {label: _curve_points(value[index], label) for index, label in enumerate(CHANNELS)}


def _curve_plot(samples, labels=CHANNELS):
    height, width = 420, 600
    canvas = np.ones((height, width, 3), dtype=np.float32)
    left, top, right, bottom = 45, 25, width - 25, height - 35
    for fraction in np.linspace(0.0, 1.0, 5):
        x = round(left + fraction * (right - left))
        y = round(bottom - fraction * (bottom - top))
        cv2.line(canvas, (x, top), (x, bottom), (0.88, 0.88, 0.88), 1)
        cv2.line(canvas, (left, y), (right, y), (0.88, 0.88, 0.88), 1)
        _put_text(canvas, f"{fraction:.2f}", (x - 12, bottom + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0.25, 0.25, 0.25), 1)
    cv2.line(canvas, (left, bottom), (right, top), (0.7, 0.7, 0.7), 1)
    colors = [(0.0, 0.65, 0.85), (0.85, 0.1, 0.5), (0.8, 0.65, 0.0), (0.15, 0.15, 0.15)]
    for index, values in enumerate(samples):
        x = np.linspace(left, right, len(values))
        y = bottom - np.asarray(values) * (bottom - top)
        points = np.stack([x, y], axis=1).round().astype(np.int32)
        color = colors[index % len(colors)]
        cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
        _put_text(canvas, str(labels[index]), (left + 20 + index * 75, top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return canvas


def apply_ink_curves(cmyk, curves=None, interpolation="linear", mask=None, preserve_endpoints=True):
    source = _image(cmyk, 4)
    valid = _valid_mask(mask, source.shape[:2])
    channel_curves = _curves(curves)
    if interpolation not in ("linear", "smooth"):
        raise ValueError("曲线插值可选 linear 或 smooth")
    if preserve_endpoints:
        for label, points in channel_curves.items():
            if points[0, 1] != 0.0 or points[-1, 1] != 1.0:
                raise ValueError(f"保留端点时 {label} 曲线须映射 0→0、1→1")
    output = source.copy()
    samples = []
    records = []
    sample_x = np.linspace(0.0, 1.0, 101)
    for index, label in enumerate(CHANNELS):
        points = channel_curves[label]
        if interpolation == "linear":
            transform = lambda values: np.interp(values, points[:, 0], points[:, 1])
        else:
            transform = PchipInterpolator(points[:, 0], points[:, 1], extrapolate=False)
        mapped = np.clip(transform(source[..., index]), 0.0, 1.0).astype(np.float32)
        output[..., index][valid] = mapped[valid]
        sample_y = np.clip(transform(sample_x), 0.0, 1.0)
        samples.append(sample_y)
        change = output[..., index][valid] - source[..., index][valid]
        records.append({"channel": label, "input_mean": float(source[..., index][valid].mean()), "output_mean": float(output[..., index][valid].mean()), "mean_change": float(change.mean()), "maximum_absolute_change": float(np.max(np.abs(change))), "control_points": points.tolist()})
    curve_table = []
    for index, x in enumerate(sample_x):
        row = {"input": float(x)}
        row.update({label: float(samples[channel][index]) for channel, label in enumerate(CHANNELS)})
        curve_table.append(row)
    return {"summary": {"interpolation": interpolation, "preserve_endpoints": bool(preserve_endpoints), "valid_pixels": int(valid.sum()), "operation": "manual_ink_curves"}, "images": {"coverage": output, "preview": _simple_preview(output), "curves": _curve_plot(samples)}, "tables": {"channels": records, "curve_samples": curve_table}}


def dot_gain(cmyk, gains=(0.15, 0.15, 0.12, 0.18), mask=None, inverse=False):
    source = _image(cmyk, 4)
    valid = _valid_mask(mask, source.shape[:2])
    gain_values = _vector(gains, 4, "50% 网点扩大", -0.24, 0.24)
    output = source.copy()
    sample_x = np.linspace(0.0, 1.0, 1001)
    samples = []
    rows = []
    for index, label in enumerate(CHANNELS):
        gain = gain_values[index]
        forward = sample_x + 4.0 * gain * sample_x * (1.0 - sample_x)
        if inverse:
            mapped = np.interp(source[..., index], forward, sample_x).astype(np.float32)
            display = np.interp(sample_x, forward, sample_x)
        else:
            mapped = source[..., index] + 4.0 * gain * source[..., index] * (1.0 - source[..., index])
            display = forward
        output[..., index][valid] = np.clip(mapped[valid], 0.0, 1.0)
        samples.append(display)
        rows.append({"channel": label, "gain_at_half": float(gain), "input_mean": float(source[..., index][valid].mean()), "output_mean": float(output[..., index][valid].mean()), "half_tone_output": float(np.interp(0.5, sample_x, display))})
    return {"summary": {"mode": "inverse_compensation" if inverse else "forward_simulation", "valid_pixels": int(valid.sum()), "model": "单调二次网点扩大曲线；不是实测印机特性"}, "images": {"coverage": output, "preview": _simple_preview(output), "curves": _curve_plot(samples)}, "tables": {"channels": rows}}


def _ink_values(value):
    inks = np.asarray(DEFAULT_INKS if value is None else value, dtype=np.float64)
    if inks.shape != (4, 3) or not np.isfinite(inks).all():
        raise ValueError("油墨颜色须为四个 RGB 三元组")
    if np.any(inks < 0.0) or np.any(inks > 1.0):
        raise ValueError("油墨 RGB 值应在 0 到 1 之间")
    return inks


def _print_order(value):
    if value is None:
        return [0, 1, 2, 3]
    values = list(value)
    if len(values) != 4:
        raise ValueError("印刷顺序必须包含四个通道")
    if all(isinstance(item, str) for item in values):
        if set(values) != set(CHANNELS):
            raise ValueError("印刷顺序必须恰好包含 C、M、Y、K")
        return [CHANNELS.index(item) for item in values]
    if set(values) != {0, 1, 2, 3}:
        raise ValueError("印刷顺序必须恰好包含 0、1、2、3")
    return [int(item) for item in values]


def overprint(cmyk, ink_rgb=None, paper_rgb=(1.0, 1.0, 1.0), opacity=(0.0, 0.0, 0.0, 0.0), order=None, mask=None, trapping=(1.0, 1.0, 1.0, 1.0)):
    source = _image(cmyk, 4)
    valid = _valid_mask(mask, source.shape[:2])
    inks = _ink_values(ink_rgb)
    paper = _vector(paper_rgb, 3, "纸张颜色", 0.01, 1.0)
    opacities = _vector(opacity, 4, "油墨不透明度", 0.0, 1.0)
    efficiencies = _vector(trapping, 4, "湿叠印转移率", 0.0, 1.0)
    sequence = _print_order(order)
    result = np.empty(source.shape[:2] + (3,), dtype=np.float32)
    result[:] = paper
    previous_coverage = np.zeros(source.shape[:2], dtype=np.float32)
    stages = {}
    rows = []
    for position, index in enumerate(sequence):
        coverage = source[..., index] * valid
        effective = coverage * (1.0 - previous_coverage * (1.0 - efficiencies[index]))
        transmission = 1.0 - effective[..., None] * (1.0 - inks[index])
        transparent = result * transmission
        opaque = result * (1.0 - effective[..., None]) + (paper * inks[index]) * effective[..., None]
        result = transparent * (1.0 - opacities[index]) + opaque * opacities[index]
        result = np.clip(result, 0.0, 1.0).astype(np.float32)
        previous_coverage = 1.0 - (1.0 - previous_coverage) * (1.0 - coverage)
        stages[f"after_{position + 1}_{CHANNELS[index]}"] = result.copy()
        rows.append({"sequence": position + 1, "channel": CHANNELS[index], "ink_rgb": inks[index].tolist(), "opacity": float(opacities[index]), "trapping": float(efficiencies[index]), "nominal_mean_coverage": float(coverage[valid].mean()), "effective_mean_coverage": float(effective[valid].mean())})
    total = source.sum(axis=2)
    overlap = np.maximum(0, (source > 0.01).sum(axis=2) - 1).astype(np.float32)
    stages["preview"] = result
    stages["ink_overlap"] = _heatmap(overlap, 3.0)
    stages["total_ink"] = _heatmap(total, 4.0)
    return {"summary": {"print_order": [CHANNELS[index] for index in sequence], "paper_rgb": paper.tolist(), "mean_total_coverage": float(total[valid].mean()), "maximum_total_coverage": float(total[valid].max()), "overprinted_fraction": float(np.mean(overlap[valid] > 0)), "model": "RGB 透射率/遮盖混合的近似预览，不是 ICC 或光谱合同打样"}, "images": stages, "tables": {"inks": rows}}


def ink_estimate(cmyk, width_mm=210.0, height_mm=297.0, copies=1000, waste_percent=5.0, film_microns=(1.0, 1.0, 1.0, 1.0), density_g_ml=(1.0, 1.0, 1.0, 1.0), efficiency=(0.85, 0.85, 0.85, 0.85), price_per_kg=(0.0, 0.0, 0.0, 0.0), mask=None, zones=8):
    source = _image(cmyk, 4)
    valid = _valid_mask(mask, source.shape[:2])
    width_mm = _number(width_mm, "印刷宽度", 1.0, 10000.0)
    height_mm = _number(height_mm, "印刷高度", 1.0, 10000.0)
    copies = _integer(copies, "成品份数", 1, 100_000_000)
    waste_percent = _number(waste_percent, "损耗百分比", 0.0, 100.0)
    films = _vector(film_microns, 4, "实地墨膜厚度", 0.01, 100.0)
    densities = _vector(density_g_ml, 4, "油墨密度", 0.1, 10.0)
    efficiencies = _vector(efficiency, 4, "利用率", 0.01, 1.0)
    prices = _vector(price_per_kg, 4, "每千克价格", 0.0, 1_000_000.0)
    zones = _integer(zones, "墨键分区数", 1, min(128, source.shape[1]))
    printed_copies = math.ceil(copies * (1.0 + waste_percent / 100.0))
    pixel_area = width_mm * height_mm / (source.shape[0] * source.shape[1])
    masked = source * valid[..., None]
    records = []
    total_grams = 0.0
    total_price = 0.0
    for index, label in enumerate(CHANNELS):
        area = float(masked[..., index].sum() * pixel_area)
        volume_per_copy_ml = area * films[index] / 1_000_000.0
        grams = volume_per_copy_ml * printed_copies * densities[index] / efficiencies[index]
        price = grams * prices[index] / 1000.0
        total_grams += grams
        total_price += price
        records.append({"channel": label, "mean_coverage_full_sheet": float(masked[..., index].mean()), "mean_coverage_valid_area": float(source[..., index][valid].mean()), "inked_area_mm2_per_copy": area, "film_microns": float(films[index]), "volume_ml_per_copy": volume_per_copy_ml, "density_g_ml": float(densities[index]), "efficiency": float(efficiencies[index]), "required_grams": grams, "price": price})
    edges = np.linspace(0, source.shape[1], zones + 1).astype(int)
    zone_rows = []
    zone_image = _simple_preview(source)
    for zone in range(zones):
        x0, x1 = int(edges[zone]), int(edges[zone + 1])
        for index, label in enumerate(CHANNELS):
            values = masked[:, x0:x1, index]
            zone_rows.append({"zone": zone + 1, "channel": label, "left_mm": width_mm * x0 / source.shape[1], "right_mm": width_mm * x1 / source.shape[1], "mean_coverage": float(values.mean()), "inked_area_mm2": float(values.sum() * pixel_area)})
        if zone:
            cv2.line(zone_image, (x0, 0), (x0, zone_image.shape[0] - 1), (0.85, 0.05, 0.1), 1)
        if x1 - x0 > 20:
            _put_text(zone_image, str(zone + 1), (x0 + 4, min(18, zone_image.shape[0] - 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0.05, 0.05, 0.05), 1)
    return {"summary": {"width_mm": width_mm, "height_mm": height_mm, "finished_copies": copies, "printed_copies": printed_copies, "waste_percent": waste_percent, "valid_area_mm2": float(valid.sum() * pixel_area), "total_grams": total_grams, "estimated_price": total_price, "zones": zones, "limitation": "估算由用户给定墨膜和利用率决定，不含洗机、残墨或设备最小装墨量"}, "images": {"ink_zones": zone_image, "total_ink": _heatmap(masked.sum(axis=2), 4.0)}, "tables": {"inks": records, "zones": zone_rows}}


def _patch_rectangles(patches, shape):
    if not isinstance(patches, (list, tuple)) or not 1 <= len(patches) <= 512:
        raise ValueError("需提供 1 至 512 个测量色块")
    output = []
    height, width = shape
    for index, item in enumerate(patches):
        if isinstance(item, dict):
            coordinates = [item.get(key) for key in ("x", "y", "width", "height")]
            label = str(item.get("label", index + 1))[:120]
        else:
            coordinates = item
            label = str(index + 1)
        rect = np.asarray(coordinates, dtype=np.float64)
        if rect.shape != (4,) or not np.isfinite(rect).all():
            raise ValueError("色块需要 x、y、width、height 四个数值")
        if np.any(rect != np.round(rect)):
            raise ValueError("色块坐标须为整数像素")
        x, y, patch_width, patch_height = map(int, rect)
        if x < 0 or y < 0 or patch_width < 1 or patch_height < 1 or x + patch_width > width or y + patch_height > height:
            raise ValueError(f"色块 {label} 超出图像范围")
        output.append({"label": label, "x": x, "y": y, "width": patch_width, "height": patch_height})
    return output


def _sample_patch(source, patch, inset):
    x, y, width, height = (patch[key] for key in ("x", "y", "width", "height"))
    margin_x = min((width - 1) // 2, int(round(width * inset)))
    margin_y = min((height - 1) // 2, int(round(height * inset)))
    values = source[y + margin_y:y + height - margin_y, x + margin_x:x + width - margin_x]
    flat = values.reshape(-1, 3)
    median = np.median(flat, axis=0)
    return {"median": median, "mean": flat.mean(axis=0), "std": flat.std(axis=0), "count": int(len(flat)), "p05": np.percentile(flat, 5, axis=0), "p95": np.percentile(flat, 95, axis=0)}


def _density(rgb, paper_rgb, channel):
    ratio = np.clip(np.asarray(rgb, dtype=np.float64) / paper_rgb, 1e-6, 1.0)
    if channel == "C":
        return -np.log10(ratio[..., 0])
    if channel == "M":
        return -np.log10(ratio[..., 1])
    if channel == "Y":
        return -np.log10(ratio[..., 2])
    if channel == "K":
        luminance = ratio[..., 0] * 0.2126 + ratio[..., 1] * 0.7152 + ratio[..., 2] * 0.0722
        return -np.log10(luminance)
    raise ValueError("测量通道须为 C、M、Y 或 K")


def _patch_overlay(source, patches, values=None):
    overlay = source.copy()
    for index, patch in enumerate(patches):
        x, y, width, height = (patch[key] for key in ("x", "y", "width", "height"))
        cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), (0.95, 0.15, 0.1), 1)
        label = str(index + 1)
        if values is not None:
            label += f":{values[index]:.2f}"
        _put_text(overlay, label, (x + 2, min(overlay.shape[0] - 2, y + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0.9, 0.1, 0.1), 1, cv2.LINE_AA)
    return overlay


def analyse_step_wedge(rgb, patches, nominal, channel="K", paper_rgb=None, solid_density=None, inset=0.15):
    source = _image(rgb, 3)
    rectangles = _patch_rectangles(patches, source.shape[:2])
    nominal_values = _vector(nominal, len(rectangles), "名义网点覆盖率", 0.0, 1.0)
    inset = _number(inset, "色块内缩比例", 0.0, 0.45)
    if channel not in CHANNELS:
        raise ValueError("测量通道须为 C、M、Y 或 K")
    sampled = [_sample_patch(source, patch, inset) for patch in rectangles]
    medians = np.array([item["median"] for item in sampled])
    if paper_rgb is None:
        paper_candidates = np.flatnonzero(nominal_values <= 1e-6)
        if not paper_candidates.size:
            raise ValueError("未给定纸白颜色时，色块中须包含 0% 纸白")
        paper = np.median(medians[paper_candidates], axis=0)
        paper = np.clip(paper, 1e-4, 1.0)
    else:
        paper = _vector(paper_rgb, 3, "纸白颜色", 1e-4, 1.0)
    densities = _density(medians, paper, channel)
    if solid_density is None:
        solid_candidates = np.flatnonzero(nominal_values >= 1.0 - 1e-6)
        if not solid_candidates.size:
            raise ValueError("未给定实地密度时，色块中须包含 100% 实地")
        solid = float(np.median(densities[solid_candidates]))
    else:
        solid = _number(solid_density, "实地密度", 0.001, 6.0)
    if solid <= 0.001:
        raise ValueError("实地与纸白差异太小，无法估算网点面积")
    effective = np.clip((1.0 - np.power(10.0, -densities)) / (1.0 - 10.0**(-solid)), 0.0, 1.0)
    records = []
    for index, patch in enumerate(rectangles):
        sample = sampled[index]
        records.append({"patch": index + 1, "label": patch["label"], "nominal": float(nominal_values[index]), "density": float(densities[index]), "effective_coverage": float(effective[index]), "tone_value_increase": float(effective[index] - nominal_values[index]), "median_rgb": medians[index].tolist(), "rgb_std": sample["std"].tolist(), "sampled_pixels": sample["count"], "x": patch["x"], "y": patch["y"], "width": patch["width"], "height": patch["height"]})
    order = np.argsort(nominal_values)
    unique_nominal = np.unique(nominal_values)
    mean_effective = np.array([effective[nominal_values == value].mean() for value in unique_nominal])
    monotone_violations = int(np.sum(np.diff(mean_effective) < -0.01))
    sample_x = np.linspace(0.0, 1.0, 101)
    measured_curve = np.interp(sample_x, unique_nominal, mean_effective)
    plot = _curve_plot([measured_curve], [channel])
    return {"summary": {"channel": channel, "paper_rgb": paper.tolist(), "solid_density": solid, "patch_count": len(records), "mean_absolute_tone_increase": float(np.mean(np.abs(effective - nominal_values))), "monotone_violations": monotone_violations, "model": "RGB 反射比估算密度与 Murray-Davies 面积；未经仪器标定，不等同密度计读数"}, "images": {"patches": _patch_overlay(source, rectangles, densities), "measured_curve": plot}, "tables": {"patches": records, "ordered_response": [records[index] for index in order]}}


def measure_color_bars(rgb, patches, channels=None, target_densities=None, paper_rgb=(1.0, 1.0, 1.0), tolerance=0.1, inset=0.15):
    source = _image(rgb, 3)
    rectangles = _patch_rectangles(patches, source.shape[:2])
    paper = _vector(paper_rgb, 3, "纸白颜色", 1e-4, 1.0)
    tolerance = _number(tolerance, "密度容差", 0.0, 2.0)
    inset = _number(inset, "色块内缩比例", 0.0, 0.45)
    if channels is None:
        channels = [CHANNELS[index % 4] for index in range(len(rectangles))]
    if len(channels) != len(rectangles) or any(channel not in CHANNELS for channel in channels):
        raise ValueError("每个色块必须指定 C、M、Y 或 K 测量通道")
    targets = None
    if target_densities is not None:
        targets = _vector(target_densities, len(rectangles), "目标密度", 0.0, 6.0)
    records = []
    densities = []
    for index, patch in enumerate(rectangles):
        sampled = _sample_patch(source, patch, inset)
        measured = float(_density(sampled["median"], paper, channels[index]))
        densities.append(measured)
        target = float(targets[index]) if targets is not None else None
        error = measured - target if target is not None else None
        records.append({"patch": index + 1, "label": patch["label"], "channel": channels[index], "density": measured, "target_density": target, "error": error, "within_tolerance": abs(error) <= tolerance if error is not None else None, "median_rgb": sampled["median"].tolist(), "rgb_std": sampled["std"].tolist(), "sampled_pixels": sampled["count"]})
    summaries = []
    for channel in CHANNELS:
        selected = [row for row in records if row["channel"] == channel]
        if not selected:
            continue
        values = np.array([row["density"] for row in selected])
        summaries.append({"channel": channel, "patch_count": len(selected), "mean_density": float(values.mean()), "minimum_density": float(values.min()), "maximum_density": float(values.max()), "range": float(np.ptp(values)), "out_of_tolerance": sum(row["within_tolerance"] is False for row in selected) if targets is not None else None})
    return {"summary": {"patch_count": len(records), "paper_rgb": paper.tolist(), "tolerance": tolerance, "has_targets": targets is not None, "out_of_tolerance": sum(row["within_tolerance"] is False for row in records) if targets is not None else None, "limitation": "相机 RGB 推算的相对密度；色块位置由用户指定"}, "images": {"patches": _patch_overlay(source, rectangles, densities)}, "tables": {"patches": records, "channels": summaries}}


def build_step_wedge(width=1000, height=520, steps=11, channels=("C", "M", "Y", "K"), margin=24, gap=6, paper_rgb=(1.0, 1.0, 1.0)):
    width = _integer(width, "样张宽度", 160, 6000)
    height = _integer(height, "样张高度", 120, 6000)
    steps = _integer(steps, "灰阶数量", 3, 33)
    margin = _integer(margin, "页边距", 0, 200)
    gap = _integer(gap, "色块间距", 1, 50)
    paper = _vector(paper_rgb, 3, "纸张颜色", 0.01, 1.0)
    channels = list(channels)
    if not channels or len(channels) > 4 or len(set(channels)) != len(channels) or any(label not in CHANNELS for label in channels):
        raise ValueError("样张通道须为 C、M、Y、K 中不重复的选择")
    label_height = 20
    patch_width = (width - 2 * margin - (steps - 1) * gap) // steps
    row_height = (height - 2 * margin - (len(channels) - 1) * gap) // len(channels)
    patch_height = row_height - label_height
    if patch_width < 4 or patch_height < 4:
        raise ValueError("画布不足以容纳指定色块；请减少级数或增大画布")
    coverage = np.zeros((height, width, 4), dtype=np.float32)
    records = []
    nominal = np.linspace(0.0, 1.0, steps)
    for row, label in enumerate(channels):
        channel = CHANNELS.index(label)
        y = margin + row * (row_height + gap)
        for column, value in enumerate(nominal):
            x = margin + column * (patch_width + gap)
            coverage[y:y + patch_height, x:x + patch_width, channel] = value
            records.append({"label": f"{label} {round(value * 100)}%", "channel": label, "nominal": float(value), "x": x, "y": y, "width": patch_width, "height": patch_height})
    preview = _simple_preview(coverage, paper=paper)
    for item in records:
        x, y, patch_width, patch_height = (item[key] for key in ("x", "y", "width", "height"))
        cv2.rectangle(preview, (x, y), (x + patch_width - 1, y + patch_height - 1), (0.65, 0.65, 0.65), 1)
        _put_text(preview, f"{item['channel']} {round(item['nominal'] * 100)}", (x, y + patch_height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0.15, 0.15, 0.15), 1, cv2.LINE_AA)
    return {"summary": {"synthetic": True, "target": "single_channel_step_wedge", "width": width, "height": height, "steps": steps, "channels": channels, "patch_count": len(records), "instruction": "测量时按 patches 表选择色块内部；标签及边框不属于测量区"}, "images": {"coverage": coverage, "target": preview}, "tables": {"patches": records}}


def build_color_control_strip(width=1200, height=240, repeats=3, margin=16, gap=4, paper_rgb=(1.0, 1.0, 1.0)):
    width = _integer(width, "控制条宽度", 240, 8000)
    height = _integer(height, "控制条高度", 80, 2000)
    repeats = _integer(repeats, "控制组重复数", 1, 16)
    margin = _integer(margin, "页边距", 0, 100)
    gap = _integer(gap, "色块间距", 1, 30)
    paper = _vector(paper_rgb, 3, "纸张颜色", 0.01, 1.0)
    recipes = [
        ("C100", [1.0, 0.0, 0.0, 0.0], "C"),
        ("M100", [0.0, 1.0, 0.0, 0.0], "M"),
        ("Y100", [0.0, 0.0, 1.0, 0.0], "Y"),
        ("K100", [0.0, 0.0, 0.0, 1.0], "K"),
        ("C50", [0.5, 0.0, 0.0, 0.0], "C"),
        ("M50", [0.0, 0.5, 0.0, 0.0], "M"),
        ("Y50", [0.0, 0.0, 0.5, 0.0], "Y"),
        ("K50", [0.0, 0.0, 0.0, 0.5], "K"),
        ("CM", [1.0, 1.0, 0.0, 0.0], "K"),
        ("MY", [0.0, 1.0, 1.0, 0.0], "K"),
        ("CY", [1.0, 0.0, 1.0, 0.0], "K"),
        ("GRAY", [0.5, 0.4, 0.4, 0.0], "K"),
    ]
    count = repeats * len(recipes)
    patch_width = (width - 2 * margin - (count - 1) * gap) // count
    patch_height = height - 2 * margin - 20
    if patch_width < 3 or patch_height < 8:
        raise ValueError("控制条画布太小，请增宽或减少控制组")
    coverage = np.zeros((height, width, 4), dtype=np.float32)
    records = []
    for group in range(repeats):
        for offset, (label, recipe, channel) in enumerate(recipes):
            index = group * len(recipes) + offset
            x = margin + index * (patch_width + gap)
            y = margin
            coverage[y:y + patch_height, x:x + patch_width] = recipe
            records.append({"label": f"{group + 1}:{label}", "group": group + 1, "kind": label, "channel": channel, "recipe": recipe, "x": x, "y": y, "width": patch_width, "height": patch_height})
    preview = _simple_preview(coverage, paper=paper)
    for item in records:
        x, y, patch_width, patch_height = (item[key] for key in ("x", "y", "width", "height"))
        cv2.rectangle(preview, (x, y), (x + patch_width - 1, y + patch_height - 1), (0.55, 0.55, 0.55), 1)
        if patch_width >= 18:
            _put_text(preview, item["kind"][:2], (x, y + patch_height + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0.15, 0.15, 0.15), 1)
    return {"summary": {"synthetic": True, "target": "print_control_strip", "width": width, "height": height, "groups": repeats, "patch_count": len(records), "limitation": "通用演示控制条，非标准认证测量标板"}, "images": {"coverage": coverage, "target": preview}, "tables": {"patches": records}}
