"""Image-based measurements for reviewing scanned print sheets."""

import math

import cv2
import numpy as np
from scipy import ndimage

from .screening import _put_text


def _rgb(value):
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("图像须为 H×W×3 的 RGB 数组")
    if min(arr.shape[:2]) < 2:
        raise ValueError("图像的宽高须至少为 2")
    if arr.shape[0] * arr.shape[1] > 24_000_000:
        raise ValueError("单张图像不能超过 2400 万像素")
    if not np.isfinite(arr).all():
        raise ValueError("图像含有无效数值")
    if np.any(arr < -1e-5) or np.any(arr > 1.00001):
        raise ValueError("RGB 值应在 0 到 1 之间")
    return np.clip(arr, 0.0, 1.0)


def _number(value, name, low, high):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须为数值") from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} 应在 {low} 到 {high} 之间")
    return number


def _integer(value, name, low, high):
    number = _number(value, name, low, high)
    if int(number) != number:
        raise ValueError(f"{name} 必须为整数")
    return int(number)


def _mask(value, shape):
    if value is None:
        return np.ones(shape, dtype=bool)
    selected = np.asarray(value, dtype=bool)
    if selected.shape != shape:
        raise ValueError("有效区域与图像尺寸不同")
    if not selected.any():
        raise ValueError("有效区域为空")
    return selected


def _paper(value):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.isfinite(arr).all() or np.any(arr < 0.0001) or np.any(arr > 1.0):
        raise ValueError("纸白须为三个 0.0001 到 1 之间的 RGB 数值")
    return arr


def _gray(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _rgb_lab(rgb):
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2LAB)


def _heatmap(values, mask=None, maximum=None):
    data = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(data)
    if mask is not None:
        valid &= mask
    selected = data[valid]
    high = float(np.percentile(selected, 99)) if selected.size and maximum is None else maximum
    if high is None or high <= 0.0:
        high = 1.0
    scaled = np.zeros(data.shape, dtype=np.uint8)
    scaled[valid] = np.clip(data[valid] * 255.0 / high, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    image = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image[~valid] = (0.8, 0.8, 0.8)
    return image


def _overlay_mask(source, selected, color=(1.0, 0.1, 0.1), alpha=0.5):
    output = source.copy()
    output[selected] = output[selected] * (1.0 - alpha) + np.asarray(color) * alpha
    return np.clip(output, 0.0, 1.0)


def _sample_summary(values):
    arr = np.asarray(values, dtype=np.float64)
    if not arr.size:
        return {"count": 0, "mean": None, "std": None, "minimum": None, "maximum": None, "median": None, "p05": None, "p95": None}
    return {"count": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std()), "minimum": float(arr.min()), "maximum": float(arr.max()), "median": float(np.median(arr)), "p05": float(np.percentile(arr, 5)), "p95": float(np.percentile(arr, 95))}


def _region_mask(region, shape):
    if not isinstance(region, dict):
        raise ValueError("区域须包含矩形坐标或 polygon 顶点")
    height, width = shape
    selected = np.zeros(shape, dtype=np.uint8)
    if "polygon" in region:
        points = np.asarray(region["polygon"], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not 3 <= len(points) <= 1000:
            raise ValueError("多边形区域须有 3 至 1000 个 XY 顶点")
        if not np.isfinite(points).all():
            raise ValueError("多边形含无效坐标")
        if np.any(points[:, 0] < 0) or np.any(points[:, 0] > width - 1) or np.any(points[:, 1] < 0) or np.any(points[:, 1] > height - 1):
            raise ValueError("多边形顶点超出图像")
        points = np.round(points).astype(np.int32)
        if abs(cv2.contourArea(points)) < 0.5:
            raise ValueError("多边形面积为空")
        cv2.fillPoly(selected, [points], 1)
        x, y, region_width, region_height = cv2.boundingRect(points)
        outline = points
    else:
        raw = [region.get(key) for key in ("x", "y", "width", "height")]
        values = np.asarray(raw, dtype=np.float64)
        if values.shape != (4,) or not np.isfinite(values).all() or np.any(values != np.round(values)):
            raise ValueError("矩形区域需要整数 x、y、width、height")
        x, y, region_width, region_height = map(int, values)
        if x < 0 or y < 0 or region_width < 1 or region_height < 1 or x + region_width > width or y + region_height > height:
            raise ValueError("矩形区域超出图像")
        selected[y:y + region_height, x:x + region_width] = 1
        outline = np.array([[x, y], [x + region_width - 1, y], [x + region_width - 1, y + region_height - 1], [x, y + region_height - 1]], dtype=np.int32)
    return selected.astype(bool), (x, y, region_width, region_height), outline


def _component_table(binary, score, min_area, max_regions, connectivity=8):
    component_count, labels, statistics, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=connectivity)
    candidates = np.flatnonzero(statistics[1:, cv2.CC_STAT_AREA] >= min_area) + 1
    ordered = candidates[np.argsort(statistics[candidates, cv2.CC_STAT_AREA])[::-1]]
    retained = ordered[:max_regions]
    retained_mask = np.isin(labels, retained)
    records = []
    for rank, identifier in enumerate(retained, start=1):
        x = int(statistics[identifier, cv2.CC_STAT_LEFT])
        y = int(statistics[identifier, cv2.CC_STAT_TOP])
        width = int(statistics[identifier, cv2.CC_STAT_WIDTH])
        height = int(statistics[identifier, cv2.CC_STAT_HEIGHT])
        local = labels[y:y + height, x:x + width] == identifier
        values = score[y:y + height, x:x + width][local]
        area = int(statistics[identifier, cv2.CC_STAT_AREA])
        contours, hierarchy = cv2.findContours(local.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
        holes = 0
        if hierarchy is not None:
            holes = int(np.sum(hierarchy[0, :, 3] >= 0))
        records.append({"region": rank, "label": int(identifier), "x": x, "y": y, "width": width, "height": height, "area_px": area, "centroid_x": float(centroids[identifier, 0]), "centroid_y": float(centroids[identifier, 1]), "mean_score": float(values.mean()), "maximum_score": float(values.max()), "p95_score": float(np.percentile(values, 95)), "fill_fraction": area / (width * height), "perimeter_px": float(perimeter), "holes": holes, "touches_boundary": x == 0 or y == 0 or x + width == binary.shape[1] or y + height == binary.shape[0]})
    metadata = {"raw_components": int(component_count - 1), "eligible_components": int(len(ordered)), "reported_components": int(len(retained)), "omitted_components": max(0, len(ordered) - max_regions), "filtered_small_components": int(component_count - 1 - len(ordered))}
    return records, retained_mask, labels, metadata


def connected_defects(score, threshold=0.05, min_area=8, max_regions=200, mask=None, source_rgb=None, connectivity=8, opening=0, closing=0):
    values = np.asarray(score, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("差异分数须为二维图像")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("差异分数须为非负有限数值")
    if values.size > 24_000_000:
        raise ValueError("分数图不能超过 2400 万像素")
    valid = _mask(mask, values.shape)
    threshold = _number(threshold, "差异阈值", 0.0, 1_000_000.0)
    min_area = _integer(min_area, "最小区域面积", 1, 24_000_000)
    max_regions = _integer(max_regions, "最大区域数", 1, 2000)
    opening = _integer(opening, "开运算半径", 0, 15)
    closing = _integer(closing, "闭运算半径", 0, 15)
    if connectivity not in (4, 8):
        raise ValueError("连通邻域须为 4 或 8")
    binary = (values > threshold) & valid
    if opening:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening * 2 + 1, opening * 2 + 1))
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool) & valid
    if closing:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing * 2 + 1, closing * 2 + 1))
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool) & valid
    rows, retained, labels, metadata = _component_table(binary, values, min_area, max_regions, connectivity)
    if source_rgb is None:
        base = np.repeat(np.full(values.shape + (1,), 0.9, dtype=np.float32), 3, axis=2)
    else:
        base = _rgb(source_rgb)
        if base.shape[:2] != values.shape:
            raise ValueError("背景图像与分数图尺寸不同")
    overlay = _overlay_mask(base, retained)
    for row in rows:
        x, y, width, height = (row[key] for key in ("x", "y", "width", "height"))
        cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), (1.0, 0.05, 0.1), 1)
        _put_text(overlay, str(row["region"]), (x, max(12, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0.9, 0.05, 0.1), 1, cv2.LINE_AA)
    summary = {"threshold": threshold, "minimum_area_px": min_area, "valid_pixels": int(valid.sum()), "thresholded_pixels": int(binary.sum()), "reported_region_pixels": int(retained.sum()), "opening_radius": opening, "closing_radius": closing, "connectivity": connectivity}
    summary.update(metadata)
    mask_image = np.repeat(retained[..., None].astype(np.float32), 3, axis=2)
    return {"summary": summary, "images": {"overlay": overlay, "mask": mask_image, "score": _heatmap(values, valid)}, "tables": {"regions": rows}}


def _ssim_map(first, second, valid):
    kernel = 1.5
    weight = ndimage.gaussian_filter(valid.astype(np.float64), kernel, mode="reflect")
    mean_a = ndimage.gaussian_filter(first * valid, kernel, mode="reflect") / np.maximum(weight, 1e-12)
    mean_b = ndimage.gaussian_filter(second * valid, kernel, mode="reflect") / np.maximum(weight, 1e-12)
    second_a = ndimage.gaussian_filter(first * first * valid, kernel, mode="reflect") / np.maximum(weight, 1e-12)
    second_b = ndimage.gaussian_filter(second * second * valid, kernel, mode="reflect") / np.maximum(weight, 1e-12)
    cross = ndimage.gaussian_filter(first * second * valid, kernel, mode="reflect") / np.maximum(weight, 1e-12)
    variance_a = np.maximum(0.0, second_a - mean_a * mean_a)
    variance_b = np.maximum(0.0, second_b - mean_b * mean_b)
    covariance = cross - mean_a * mean_b
    numerator = (2.0 * mean_a * mean_b + 0.01**2) * (2.0 * covariance + 0.03**2)
    denominator = (mean_a**2 + mean_b**2 + 0.01**2) * (variance_a + variance_b + 0.03**2)
    return np.clip(numerator / denominator, -1.0, 1.0).astype(np.float32)


def compare_images(reference, candidate, mask=None, threshold=0.05, min_area=8, metric="rgb", max_regions=200, blur_sigma=0.0):
    first = _rgb(reference)
    second = _rgb(candidate)
    if first.shape != second.shape:
        raise ValueError("比较图像尺寸须相同；请先完成套准或裁切")
    valid = _mask(mask, first.shape[:2])
    blur_sigma = _number(blur_sigma, "比较前平滑尺度", 0.0, 8.0)
    if metric not in ("rgb", "luminance", "lab", "structure"):
        raise ValueError("比较方式须为 rgb、luminance、lab 或 structure")
    threshold = _number(threshold, "差异阈值", 0.0, 200.0 if metric == "lab" else 1.0)
    measured_a = first
    measured_b = second
    if blur_sigma:
        weights = ndimage.gaussian_filter(valid.astype(np.float32), blur_sigma, mode="reflect")
        measured_a = np.empty_like(first)
        measured_b = np.empty_like(second)
        for channel in range(3):
            measured_a[..., channel] = ndimage.gaussian_filter(first[..., channel] * valid, blur_sigma, mode="reflect") / np.maximum(weights, 1e-10)
            measured_b[..., channel] = ndimage.gaussian_filter(second[..., channel] * valid, blur_sigma, mode="reflect") / np.maximum(weights, 1e-10)
    gray_a = _gray(measured_a)
    gray_b = _gray(measured_b)
    absolute = np.abs(measured_a - measured_b)
    ssim = _ssim_map(gray_a, gray_b, valid)
    if metric == "rgb":
        score = absolute.max(axis=2)
        units = "最大 RGB 通道差"
    elif metric == "luminance":
        score = np.abs(gray_a - gray_b)
        units = "相对亮度差"
    elif metric == "lab":
        score = np.linalg.norm(_rgb_lab(measured_a) - _rgb_lab(measured_b), axis=2)
        units = "CIE Lab 欧氏距离，基于 sRGB 假设"
    else:
        score = (1.0 - ssim) / 2.0
        units = "局部结构差 (1-SSIM)/2"
    result = connected_defects(score, threshold, min_area, max_regions, valid, second)
    mse = float(np.mean((measured_a[valid] - measured_b[valid]) ** 2))
    result["summary"].update({"metric": metric, "units": units, "mean_score": float(score[valid].mean()), "p95_score": float(np.percentile(score[valid], 95)), "maximum_score": float(score[valid].max()), "rgb_mean_absolute_error": float(absolute[valid].mean()), "rgb_mse": mse, "psnr_db": -10.0 * math.log10(mse) if mse > 0.0 else None, "identical": mse == 0.0, "mean_ssim": float(ssim[valid].mean()), "blur_sigma_px": blur_sigma, "limitation": "直接比较已对齐图像，不自动补偿形变、曝光或可变印字"})
    result["images"]["absolute_rgb"] = np.clip(absolute * 4.0, 0.0, 1.0)
    result["images"]["blend"] = (first + second) / 2.0
    result["images"]["structure_difference"] = _heatmap((1.0 - ssim) / 2.0, valid)
    channel_rows = []
    for index, label in enumerate(("R", "G", "B")):
        values = measured_b[..., index][valid] - measured_a[..., index][valid]
        channel_rows.append({"channel": label, "mean_signed_difference": float(values.mean()), "mean_absolute_difference": float(np.abs(values).mean()), "maximum_absolute_difference": float(np.abs(values).max()), "reference_mean": float(measured_a[..., index][valid].mean()), "candidate_mean": float(measured_b[..., index][valid].mean())})
    result["tables"]["channels"] = channel_rows
    return result


def _density_maps(source, paper):
    ratio = np.clip(source / paper, 1e-6, 1.0)
    density = -np.log10(ratio)
    gray_ratio = ratio[..., 0] * 0.2126 + ratio[..., 1] * 0.7152 + ratio[..., 2] * 0.0722
    return {"C": density[..., 0], "M": density[..., 1], "Y": density[..., 2], "K": -np.log10(gray_ratio)}


def density_grid(rgb, rows=8, columns=8, paper_rgb=(1.0, 1.0, 1.0), channel="K", mask=None, inset=0.1):
    source = _rgb(rgb)
    valid = _mask(mask, source.shape[:2])
    paper = _paper(paper_rgb)
    rows = _integer(rows, "网格行数", 1, min(source.shape[0], 128))
    columns = _integer(columns, "网格列数", 1, min(source.shape[1], 128))
    inset = _number(inset, "单元内缩比例", 0.0, 0.4)
    if channel not in ("C", "M", "Y", "K"):
        raise ValueError("测量通道须为 C、M、Y 或 K")
    maps = _density_maps(source, paper)
    yedges = np.linspace(0, source.shape[0], rows + 1).astype(int)
    xedges = np.linspace(0, source.shape[1], columns + 1).astype(int)
    records = []
    cell_map = np.zeros(source.shape[:2], dtype=np.float32)
    measured = np.zeros(source.shape[:2], dtype=bool)
    overlay = source.copy()
    for row in range(rows):
        for column in range(columns):
            y0, y1 = int(yedges[row]), int(yedges[row + 1])
            x0, x1 = int(xedges[column]), int(xedges[column + 1])
            dx = min((x1 - x0 - 1) // 2, round((x1 - x0) * inset))
            dy = min((y1 - y0 - 1) // 2, round((y1 - y0) * inset))
            selected = valid[y0 + dy:y1 - dy, x0 + dx:x1 - dx]
            count = int(selected.sum())
            item = {"row": row + 1, "column": column + 1, "x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0, "sampled_pixels": count}
            for label, values in maps.items():
                sample = values[y0 + dy:y1 - dy, x0 + dx:x1 - dx][selected]
                item[f"density_{label}"] = float(np.median(sample)) if count else None
            current = maps[channel][y0 + dy:y1 - dy, x0 + dx:x1 - dx][selected]
            item["density_std"] = float(current.std()) if count else None
            item["density_range_p95_p05"] = float(np.percentile(current, 95) - np.percentile(current, 5)) if count else None
            records.append(item)
            if count:
                cell_map[y0:y1, x0:x1] = item[f"density_{channel}"]
                measured[y0:y1, x0:x1] = True
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0.9, 0.1, 0.1), 1)
            if x1 - x0 > 35 and y1 - y0 > 16 and count:
                _put_text(overlay, f"{item[f'density_{channel}']:.2f}", (x0 + 2, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0.9, 0.1, 0.1), 1, cv2.LINE_AA)
    cell_values = [row[f"density_{channel}"] for row in records if row["sampled_pixels"]]
    return {"summary": {"rows": rows, "columns": columns, "channel": channel, "paper_rgb": paper.tolist(), "measured_cells": len(cell_values), "empty_cells": rows * columns - len(cell_values), "cell_density": _sample_summary(cell_values), "limitation": "相机相对密度仅用于同扫描条件比较，图案边缘与纸白混合会改变读数"}, "images": {"grid": overlay, "density": _heatmap(cell_map, measured), "pixel_density": _heatmap(maps[channel], valid)}, "tables": {"cells": records}}


def roi_statistics(rgb, regions, mask=None, paper_rgb=(1.0, 1.0, 1.0), dpi=300.0):
    source = _rgb(rgb)
    valid = _mask(mask, source.shape[:2])
    paper = _paper(paper_rgb)
    dpi = _number(dpi, "图像 DPI", 36.0, 4800.0)
    if not isinstance(regions, (list, tuple)) or not 1 <= len(regions) <= 256:
        raise ValueError("需要 1 至 256 个测量区域")
    lab = _rgb_lab(source)
    gray = _gray(source)
    density = _density_maps(source, paper)
    overlay = source.copy()
    rows = []
    density_rows = []
    for index, region in enumerate(regions):
        selected, bounds, outline = _region_mask(region, source.shape[:2])
        selected &= valid
        label = str(region.get("label", f"区域 {index + 1}"))[:120]
        count = int(selected.sum())
        if not count:
            raise ValueError(f"{label} 中没有有效像素")
        pixels = source[selected]
        lab_pixels = lab[selected]
        means = pixels.mean(axis=0)
        median = np.median(pixels, axis=0)
        deviation = pixels.std(axis=0)
        lab_mean = lab_pixels.mean(axis=0)
        color_distances = np.linalg.norm(lab_pixels - lab_mean, axis=1)
        x, y, width, height = bounds
        rows.append({"region": index + 1, "label": label, "x": x, "y": y, "width": width, "height": height, "valid_pixels": count, "area_mm2": count * (25.4 / dpi) ** 2, "mean_rgb": means.tolist(), "median_rgb": median.tolist(), "rgb_std": deviation.tolist(), "mean_lab": lab_mean.tolist(), "luminance_mean": float(gray[selected].mean()), "luminance_std": float(gray[selected].std()), "color_spread_mean": float(color_distances.mean()), "color_spread_p95": float(np.percentile(color_distances, 95)), "near_white_fraction": float(np.mean(np.min(pixels, axis=1) > 0.98)), "near_black_fraction": float(np.mean(np.max(pixels, axis=1) < 0.02))})
        for channel, values in density.items():
            sample = values[selected]
            density_rows.append({"region": index + 1, "label": label, "channel": channel, "density_mean": float(sample.mean()), "density_median": float(np.median(sample)), "density_std": float(sample.std()), "density_p05": float(np.percentile(sample, 5)), "density_p95": float(np.percentile(sample, 95))})
        cv2.polylines(overlay, [outline], True, (0.95, 0.1, 0.15), 1, cv2.LINE_AA)
        _put_text(overlay, str(index + 1), (x + 2, min(source.shape[0] - 2, y + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0.9, 0.05, 0.1), 1, cv2.LINE_AA)
    return {"summary": {"region_count": len(rows), "dpi": dpi, "paper_rgb": paper.tolist(), "total_region_pixels": sum(row["valid_pixels"] for row in rows), "overlap_policy": "区域分别测量，交叠像素会计入各自区域"}, "images": {"regions": overlay}, "tables": {"regions": rows, "densities": density_rows}}


def _plot_profiles(records, keys, colors, title, width=960, height=320):
    canvas = np.ones((height, width, 3), dtype=np.float32)
    left, right, top, bottom = 40, width - 20, 35, height - 30
    all_values = np.array([[row[key] for key in keys] for row in records], dtype=np.float64)
    finite = all_values[np.isfinite(all_values)]
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 1.0
    if high - low < 1e-8:
        low -= 0.05
        high += 0.05
    padding = (high - low) * 0.1
    low -= padding
    high += padding
    for fraction in np.linspace(0, 1, 5):
        y = round(bottom - fraction * (bottom - top))
        cv2.line(canvas, (left, y), (right, y), (0.9, 0.9, 0.9), 1)
        _put_text(canvas, f"{low + fraction * (high-low):.2f}", (2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0.3, 0.3, 0.3), 1)
    for index, key in enumerate(keys):
        values = all_values[:, index]
        x = np.linspace(left, right, len(values))
        y = bottom - (values - low) * (bottom - top) / (high - low)
        good = np.isfinite(y)
        for run in np.split(np.flatnonzero(good), np.flatnonzero(np.diff(np.flatnonzero(good)) > 1) + 1):
            if len(run) >= 2:
                points = np.stack([x[run], y[run]], axis=1).round().astype(np.int32)
                cv2.polylines(canvas, [points], False, colors[index], 1, cv2.LINE_AA)
        _put_text(canvas, key, (left + index * 130, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[index], 1)
    _put_text(canvas, title, (left, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0.15, 0.15, 0.15), 1, cv2.LINE_AA)
    return canvas


def line_profile(rgb, start, end, width=1, samples=None, mask=None, dpi=300.0):
    source = _rgb(rgb)
    valid = _mask(mask, source.shape[:2])
    dpi = _number(dpi, "图像 DPI", 36.0, 4800.0)
    line_width = _integer(width, "采样线宽", 1, 101)
    points = np.asarray([start, end], dtype=np.float64)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError("起点和终点须为有限 XY 坐标")
    if np.any(points[:, 0] < 0) or np.any(points[:, 0] > source.shape[1] - 1) or np.any(points[:, 1] < 0) or np.any(points[:, 1] > source.shape[0] - 1):
        raise ValueError("采样线端点超出图像")
    delta = points[1] - points[0]
    length = float(np.linalg.norm(delta))
    if length < 1.0:
        raise ValueError("采样线长度至少为 1 像素")
    sample_count = min(8192, math.ceil(length) + 1) if samples is None else _integer(samples, "采样数量", 2, 8192)
    fractions = np.linspace(0.0, 1.0, sample_count)
    centers = points[0] + fractions[:, None] * delta
    normal = np.array([-delta[1], delta[0]]) / length
    offsets = np.arange(line_width) - (line_width - 1) / 2.0
    coordinates = centers[:, None, :] + offsets[None, :, None] * normal
    x = coordinates[..., 0].astype(np.float32)
    y = coordinates[..., 1].astype(np.float32)
    inside = (x >= 0) & (x <= source.shape[1] - 1) & (y >= 0) & (y <= source.shape[0] - 1)
    weights = cv2.remap(valid.astype(np.float32), x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    weights *= inside
    sampled = cv2.remap(source * valid[..., None], x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    denominators = weights.sum(axis=1)
    mean = sampled.sum(axis=1) / np.maximum(denominators[:, None], 1e-12)
    records = []
    for index in range(sample_count):
        good = denominators[index] > 1e-6
        values = mean[index] if good else [None, None, None]
        luminance = float(np.dot(mean[index], [0.2126, 0.7152, 0.0722])) if good else None
        records.append({"sample": index, "distance_px": float(fractions[index] * length), "distance_mm": float(fractions[index] * length * 25.4 / dpi), "x": float(centers[index, 0]), "y": float(centers[index, 1]), "R": float(values[0]) if good else None, "G": float(values[1]) if good else None, "B": float(values[2]) if good else None, "luminance": luminance, "effective_samples": float(denominators[index]), "valid": bool(good)})
    plot_rows = [{key: (float("nan") if item[key] is None else item[key]) for key in ("R", "G", "B")} for item in records]
    plot = _plot_profiles(plot_rows, ["R", "G", "B"], [(0.9, 0.1, 0.1), (0.1, 0.65, 0.2), (0.1, 0.3, 0.9)], "RGB line profile")
    overlay = source.copy()
    point_a = tuple(np.round(points[0]).astype(int))
    point_b = tuple(np.round(points[1]).astype(int))
    cv2.line(overlay, point_a, point_b, (1.0, 0.05, 0.1), max(1, min(3, line_width)), cv2.LINE_AA)
    cv2.circle(overlay, point_a, 4, (0.1, 0.9, 0.2), 1)
    cv2.circle(overlay, point_b, 4, (0.1, 0.3, 1.0), 1)
    return {"summary": {"length_px": length, "length_mm": length * 25.4 / dpi, "width_px": line_width, "samples": sample_count, "valid_samples": int(np.sum(denominators > 1e-6)), "dpi": dpi, "sampling": "双线性采样，边缘与无效像素按有效权重归一化"}, "images": {"line": overlay, "profile": plot}, "tables": {"profile": records}}


def _threshold_crossing(coordinates, values, target, pivot):
    transitions = np.flatnonzero((values[:-1] - target) * (values[1:] - target) <= 0)
    if not transitions.size:
        return None
    candidates = []
    for index in transitions:
        first = float(values[index])
        second = float(values[index + 1])
        if abs(second - first) < 1e-12:
            crossing = float((coordinates[index] + coordinates[index + 1]) / 2.0)
        else:
            crossing = float(coordinates[index] + (coordinates[index + 1] - coordinates[index]) * (target - first) / (second - first))
        candidates.append(crossing)
    return min(candidates, key=lambda value: abs(value - pivot))


def sharpness_analysis(rgb, region=None, direction="auto", mask=None, dpi=300.0):
    source = _rgb(rgb)
    valid = _mask(mask, source.shape[:2])
    dpi = _number(dpi, "图像 DPI", 36.0, 4800.0)
    if direction not in ("auto", "horizontal", "vertical"):
        raise ValueError("边缘法向须为 auto、horizontal 或 vertical")
    selected = valid.copy()
    bounds = (0, 0, source.shape[1], source.shape[0])
    outline = None
    if region is not None:
        selected_region, bounds, outline = _region_mask(region, source.shape[:2])
        selected &= selected_region
    if selected.sum() < 16:
        raise ValueError("清晰度区域至少需要 16 个有效像素")
    gray = _gray(source)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0)
    magnitude = np.hypot(gx, gy)
    inner = ndimage.binary_erosion(selected, iterations=1)
    if not inner.any():
        inner = selected
    gx_energy = float(np.mean(gx[inner] ** 2))
    gy_energy = float(np.mean(gy[inner] ** 2))
    chosen = direction
    if chosen == "auto":
        chosen = "horizontal" if gx_energy >= gy_energy else "vertical"
    x, y, width, height = bounds
    cropped = gray[y:y + height, x:x + width]
    crop_mask = selected[y:y + height, x:x + width]
    reduce_axis = 0 if chosen == "horizontal" else 1
    counts = crop_mask.sum(axis=reduce_axis)
    positions = np.flatnonzero(counts > 0)
    if positions.size < 4:
        raise ValueError("边缘法向至少需要 4 个有数据的位置")
    profile = (cropped * crop_mask).sum(axis=reduce_axis) / np.maximum(counts, 1)
    profile = np.interp(np.arange(profile.size), positions, profile[positions])
    smoothed = ndimage.gaussian_filter1d(profile.astype(np.float64), 0.6, mode="reflect")
    gradient = np.gradient(smoothed)
    peak = int(np.argmax(np.abs(gradient)))
    edge_polarity = 1.0 if gradient[peak] >= 0 else -1.0
    local_radius = max(3, min(40, len(profile) // 4))
    local_left = max(0, peak - local_radius)
    local_right = min(len(profile), peak + local_radius + 1)
    low = float(np.percentile(smoothed[local_left:local_right], 10))
    high = float(np.percentile(smoothed[local_left:local_right], 90))
    contrast = high - low
    crossing_10 = None
    crossing_90 = None
    transition_width = None
    edge_rows = []
    if contrast > 0.01:
        normalized = (smoothed - low) / contrast
        if edge_polarity < 0:
            normalized = 1.0 - normalized
        local_values = normalized[local_left:local_right]
        local_positions = np.arange(local_left, local_right, dtype=np.float64)
        crossing_10 = _threshold_crossing(local_positions, local_values, 0.1, peak)
        crossing_90 = _threshold_crossing(local_positions, local_values, 0.9, peak)
        if crossing_10 is not None and crossing_90 is not None and crossing_90 >= crossing_10:
            transition_width = crossing_90 - crossing_10
        for index in range(len(profile)):
            edge_rows.append({"position_px": index, "luminance": float(profile[index]), "smoothed": float(smoothed[index]), "normalized_edge": float(normalized[index]), "derivative": float(gradient[index]), "valid_pixels": int(counts[index])})
    else:
        for index in range(len(profile)):
            edge_rows.append({"position_px": index, "luminance": float(profile[index]), "smoothed": float(smoothed[index]), "normalized_edge": None, "derivative": float(gradient[index]), "valid_pixels": int(counts[index])})
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    overlay = source.copy()
    if outline is not None:
        cv2.polylines(overlay, [outline], True, (0.95, 0.15, 0.1), 1)
    if chosen == "horizontal":
        cv2.line(overlay, (x + peak, y), (x + peak, y + height - 1), (0.95, 0.15, 0.1), 1)
    else:
        cv2.line(overlay, (x, y + peak), (x + width - 1, y + peak), (0.95, 0.15, 0.1), 1)
    summary = {"direction": chosen, "valid_pixels": int(selected.sum()), "tenengrad_energy": gx_energy + gy_energy, "horizontal_gradient_energy": gx_energy, "vertical_gradient_energy": gy_energy, "laplacian_variance": float(laplacian[inner].var()), "edge_contrast": contrast, "edge_position_px": peak, "transition_10_90_px": transition_width, "transition_10_90_mm": transition_width * 25.4 / dpi if transition_width is not None else None, "crossing_10_px": crossing_10, "crossing_90_px": crossing_90, "limitation": "轴向平均边缘测量，区域应只有一条近直线边缘；不能作为标准斜边 MTF"}
    plot = _plot_profiles(edge_rows, ["luminance", "smoothed"], [(0.1, 0.35, 0.9), (0.9, 0.3, 0.1)], "Edge transition profile")
    return {"summary": summary, "images": {"edge": overlay, "gradient": _heatmap(magnitude, selected), "profile": plot}, "tables": {"edge_profile": edge_rows}}


def uniformity_inspection(rgb, mask=None, scale=12.0, threshold=0.03, min_area=8, paper_rgb=(1.0, 1.0, 1.0), channel="luminance", max_regions=200):
    source = _rgb(rgb)
    valid = _mask(mask, source.shape[:2])
    scale = _number(scale, "背景尺度", 1.0, 300.0)
    threshold = _number(threshold, "局部差异阈值", 0.0, 6.0)
    paper = _paper(paper_rgb)
    if channel == "luminance":
        signal = _gray(source)
    elif channel in ("C", "M", "Y", "K"):
        signal = _density_maps(source, paper)[channel].astype(np.float32)
    else:
        raise ValueError("通道须为 luminance、C、M、Y 或 K")
    weights = ndimage.gaussian_filter(valid.astype(np.float32), scale, mode="reflect")
    background = ndimage.gaussian_filter(signal * valid, scale, mode="reflect") / np.maximum(weights, 1e-10)
    residual = signal - background
    score = np.abs(residual)
    result = connected_defects(score, threshold, min_area, max_regions, valid, source)
    low = (residual < -threshold) & valid
    high = (residual > threshold) & valid
    sign_overlay = _overlay_mask(source, low, (0.1, 0.25, 1.0))
    sign_overlay = _overlay_mask(sign_overlay, high, (1.0, 0.15, 0.1))
    result["summary"].update({"channel": channel, "background_scale_px": scale, "residual_mean": float(residual[valid].mean()), "residual_rms": float(np.sqrt(np.mean(residual[valid] ** 2))), "below_background_pixels": int(low.sum()), "above_background_pixels": int(high.sum()), "limitation": "检查平坦色块的局部不匀，图案边界会被当作差异；背景为掩膜归一化高斯估计"})
    result["images"]["signed_deviations"] = sign_overlay
    result["images"]["background"] = _heatmap(background, valid)
    _, component_labels = cv2.connectedComponents(((score > threshold) & valid).astype(np.uint8), connectivity=8)
    signed_rows = []
    for row in result["tables"]["regions"]:
        x, y, width, height = (row[key] for key in ("x", "y", "width", "height"))
        crop = residual[y:y + height, x:x + width]
        selected = component_labels[y:y + height, x:x + width] == row["label"]
        signed_rows.append({"region": row["region"], "mean_signed_residual": float(crop[selected].mean()) if selected.any() else 0.0, "negative_pixels": int(np.sum((crop < -threshold) & selected)), "positive_pixels": int(np.sum((crop > threshold) & selected))})
    result["tables"]["region_signs"] = signed_rows
    return result


def histogram_analysis(rgb, bins=64, mask=None, clip_low=0.01, clip_high=0.99):
    source = _rgb(rgb)
    valid = _mask(mask, source.shape[:2])
    bins = _integer(bins, "直方图分箱数", 8, 1024)
    clip_low = _number(clip_low, "低端裁切阈值", 0.0, 0.25)
    clip_high = _number(clip_high, "高端裁切阈值", 0.75, 1.0)
    values = {"R": source[..., 0], "G": source[..., 1], "B": source[..., 2], "luminance": _gray(source)}
    edges = np.linspace(0.0, 1.0, bins + 1)
    histograms = {}
    summaries = []
    for label, plane in values.items():
        sampled = plane[valid]
        counts, _ = np.histogram(sampled, bins=edges)
        histograms[label] = counts
        probabilities = counts / float(counts.sum())
        nonzero = probabilities > 0
        entropy = float(-np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero])))
        row = {"channel": label, "entropy_bits": entropy, "low_clipped_fraction": float(np.mean(sampled <= clip_low)), "high_clipped_fraction": float(np.mean(sampled >= clip_high)), "occupied_bins": int(np.sum(counts > 0))}
        row.update(_sample_summary(sampled))
        summaries.append(row)
    records = []
    for index in range(bins):
        row = {"bin": index, "low": float(edges[index]), "high": float(edges[index + 1])}
        for label in values:
            row[label] = int(histograms[label][index])
        records.append(row)
    plot_rows = []
    maximum = max(1, max(int(histogram.max()) for histogram in histograms.values()))
    for index in range(bins):
        plot_rows.append({label: math.log1p(int(histograms[label][index])) / math.log1p(maximum) for label in ("R", "G", "B")})
    plot = _plot_profiles(plot_rows, ["R", "G", "B"], [(0.9, 0.1, 0.1), (0.1, 0.6, 0.2), (0.15, 0.3, 0.9)], "RGB histogram (log counts)")
    low_mask = np.any(source <= clip_low, axis=2) & valid
    high_mask = np.any(source >= clip_high, axis=2) & valid
    overlay = _overlay_mask(source, low_mask, (0.15, 0.25, 1.0), 0.6)
    overlay = _overlay_mask(overlay, high_mask, (1.0, 0.15, 0.1), 0.6)
    return {"summary": {"bins": bins, "valid_pixels": int(valid.sum()), "clip_low": clip_low, "clip_high": clip_high, "any_channel_low_fraction": float(low_mask.sum() / valid.sum()), "any_channel_high_fraction": float(high_mask.sum() / valid.sum()), "limitation": "裁切提示只依图像数值，纸白或设计实黑不自动判为错误"}, "images": {"histogram": plot, "clipping": overlay}, "tables": {"histogram": records, "channels": summaries}}


def _morphological_skeleton(binary):
    current = binary.astype(np.uint8)
    skeleton = np.zeros_like(current)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    maximum_steps = min(binary.shape) // 2 + 2
    for _ in range(maximum_steps):
        eroded = cv2.erode(current, element, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        opened = cv2.dilate(eroded, element, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        skeleton |= current & ~opened
        current = eroded
        if not current.any():
            break
    return skeleton.astype(bool)


def stroke_integrity(rgb, threshold=0.5, minimum_width_px=2.0, polarity="dark", mask=None, region=None, dpi=300.0, min_component_area=2, max_regions=200):
    source = _rgb(rgb)
    valid = _mask(mask, source.shape[:2])
    threshold = _number(threshold, "前景阈值", 0.01, 0.99)
    minimum_width_px = _number(minimum_width_px, "最小笔画宽度", 0.5, 200.0)
    dpi = _number(dpi, "图像 DPI", 36.0, 4800.0)
    min_component_area = _integer(min_component_area, "最小前景区域面积", 1, 1_000_000)
    max_regions = _integer(max_regions, "最大区域数", 1, 2000)
    if polarity not in ("dark", "light"):
        raise ValueError("前景极性须为 dark 或 light")
    if region is not None:
        region_mask, _, _ = _region_mask(region, source.shape[:2])
        valid &= region_mask
        if not valid.any():
            raise ValueError("检测区域内没有有效像素")
    gray = _gray(source)
    foreground = (gray < threshold if polarity == "dark" else gray > threshold) & valid
    count, labels, statistics, centroids = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
    candidates = [identifier for identifier in range(1, count) if statistics[identifier, cv2.CC_STAT_AREA] >= min_component_area]
    candidates.sort(key=lambda identifier: (-int(statistics[identifier, cv2.CC_STAT_AREA]), identifier))
    chosen = candidates[:max_regions]
    distance = ndimage.distance_transform_edt(np.pad(foreground, 1))[1:-1, 1:-1]
    valid_inner = ndimage.binary_erosion(valid, iterations=1)
    boundary = valid & ~valid_inner
    skeleton = np.zeros(foreground.shape, dtype=bool)
    width_map = np.zeros(foreground.shape, dtype=np.float32)
    narrow = np.zeros(foreground.shape, dtype=bool)
    records = []
    for identifier in chosen:
        x, y, width, height, area = map(int, statistics[identifier])
        component = labels[y:y + height, x:x + width] == identifier
        local_skeleton = _morphological_skeleton(component)
        skeleton[y:y + height, x:x + width] |= local_skeleton
        radius = distance[y:y + height, x:x + width]
        local_widths = np.maximum(1.0, 2.0 * radius - 1.0)
        reliable = local_skeleton & ~boundary[y:y + height, x:x + width]
        values = local_widths[reliable]
        local_narrow = reliable & (local_widths < minimum_width_px)
        narrow[y:y + height, x:x + width] |= local_narrow
        width_map[y:y + height, x:x + width][local_skeleton] = local_widths[local_skeleton]
        contours, hierarchy = cv2.findContours(component.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = int(np.sum(hierarchy[0, :, 3] >= 0)) if hierarchy is not None else 0
        perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
        component_boundary = component & boundary[y:y + height, x:x + width]
        records.append({
            "region": len(records) + 1,
            "component_label": identifier,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "area_px": area,
            "area_mm2": area * (25.4 / dpi) ** 2,
            "centroid_x": float(centroids[identifier, 0]),
            "centroid_y": float(centroids[identifier, 1]),
            "skeleton_pixels": int(local_skeleton.sum()),
            "measured_skeleton_pixels": int(reliable.sum()),
            "minimum_width_px": float(values.min()) if values.size else None,
            "median_width_px": float(np.median(values)) if values.size else None,
            "p10_width_px": float(np.percentile(values, 10)) if values.size else None,
            "narrow_skeleton_pixels": int(local_narrow.sum()),
            "narrow_fraction": float(local_narrow.sum() / values.size) if values.size else None,
            "holes": holes,
            "perimeter_px": float(perimeter),
            "touches_valid_boundary": bool(component_boundary.any()),
        })
    overlay = source.copy()
    expanded_narrow = ndimage.binary_dilation(narrow, iterations=1) & valid
    overlay = _overlay_mask(overlay, expanded_narrow, (1.0, 0.1, 0.1), 0.7)
    skeleton_view = source.copy()
    skeleton_view[skeleton] = (0.1, 0.8, 0.2)
    skeleton_view[narrow] = (1.0, 0.1, 0.1)
    for item in records:
        if item["narrow_skeleton_pixels"]:
            x, y, width, height = (item[key] for key in ("x", "y", "width", "height"))
            cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), (0.9, 0.1, 0.1), 1)
            _put_text(overlay, str(item["region"]), (x + 1, max(12, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0.9, 0.1, 0.1), 1)
    measured_widths = width_map[skeleton & valid_inner]
    histogram = []
    if measured_widths.size:
        edges = np.linspace(0.0, max(2.0, float(np.percentile(measured_widths, 99)) + 1.0), 33)
        counts, _ = np.histogram(measured_widths, edges)
        for index, amount in enumerate(counts):
            histogram.append({"width_from_px": float(edges[index]), "width_to_px": float(edges[index + 1]), "skeleton_pixels": int(amount)})
    summary = {
        "polarity": polarity,
        "threshold": threshold,
        "required_width_px": minimum_width_px,
        "required_width_mm": minimum_width_px * 25.4 / dpi,
        "foreground_pixels": int(foreground.sum()),
        "foreground_components": int(count - 1),
        "eligible_components": len(candidates),
        "reported_components": len(records),
        "omitted_for_limit": max(0, len(candidates) - max_regions),
        "regions_with_narrow_strokes": sum(item["narrow_skeleton_pixels"] > 0 for item in records),
        "narrow_skeleton_pixels": int(narrow.sum()),
        "minimum_measured_width_px": float(measured_widths.min()) if measured_widths.size else None,
        "dpi": dpi,
        "limitation": "阈值前景的形态骨架与距离宽度近似；不识别文字语义，不判断原稿中本应相连的笔画；边界骨架不计宽度",
    }
    return {"summary": summary, "images": {"narrow_strokes": overlay, "skeleton": skeleton_view, "stroke_width": _heatmap(width_map, skeleton)}, "tables": {"regions": records, "width_histogram": histogram}}
