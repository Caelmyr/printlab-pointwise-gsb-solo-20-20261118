"""Shared spot-ink batch separation for label sets.

A batch registers 2-8 spot inks (name, solid RGB, usage cost), picks RGB
source images on one paper, and marks key regions with Delta E tolerances.
Per-ink coverages are solved under the same multiplicative ink/paper model
used by the overprint preview (paper times per-ink transmittance factors),
with coverages constrained to [0, 1] and a total ink limit. Two plans
(ink-saving vs. color-accurate) are compared before plating; confirming one
locks plates, previews, error maps and usage statistics to that recipe
snapshot. Later ink-library edits or other batches never change a confirmed
batch.
"""

import itertools

import cv2
import numpy as np

from .pixels import heatmap, integer, mask_outline, number, png_bytes, rgb_image
from .regions import geometry_mask, validate_geometry

SOLVER_VERSION = 'spotcolor-v1'
MAX_IMAGES = 8
MAX_KEY_REGIONS = 8
PALETTE_SIZE = 256
KEY_SAMPLES = 256
COARSE_ITERATIONS = 28
REFINE_ITERATIONS = 110
FULL_ITERATIONS = 24
FULL_CHUNK = 500_000
LAM_ACCURATE = 1e-4
LAM_SAVING_CANDIDATES = (2e-3, 8e-3, 3e-2, 1e-1)
IDENTICAL_COVERAGE = 0.01
IDENTICAL_DELTA_E = 0.05
IDENTICAL_COST_RELATIVE = 0.01
MODEL_NOTE = '与叠印预览一致的 RGB 透射率相乘模型（纸白 × 各墨 (1-覆盖率×(1-墨色透射率))）；非 ICC 或光谱打样'
DELTA_E_NOTE = 'CIE L*a*b* 1976 欧氏距离（sRGB 假设，L 0-100 刻度）'


def validate_spot_inks(payload):
    if not isinstance(payload, dict):
        raise ValueError('油墨配置必须为对象')
    inks = payload.get('inks')
    if not isinstance(inks, list) or not 2 <= len(inks) <= 8:
        raise ValueError('油墨库需要2到8种可用油墨')
    result = []
    names = set()
    for item in inks:
        if not isinstance(item, dict):
            raise ValueError('油墨项必须为对象')
        name = str(item.get('name') or '').strip()
        if not name or len(name) > 40:
            raise ValueError('油墨名称需要1到40个字符')
        if name in names:
            raise ValueError('油墨名称不能重复：' + name)
        names.add(name)
        rgb = item.get('rgb')
        if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
            raise ValueError('油墨' + name + '的实色需要三个RGB分量')
        solid = [number(x, '实色RGB', 0, 1) for x in rgb]
        result.append({
            'name': name,
            'rgb': solid,
            'cost_per_kg': number(item.get('cost_per_kg'), '每公斤用量成本', 0, 1_000_000, 100),
            'density': number(item.get('density'), '油墨密度', 0.1, 10, 1.0),
        })
    paper = payload.get('paper_rgb')
    checked_paper = None
    if paper is not None:
        if not isinstance(paper, (list, tuple)) or len(paper) != 3:
            raise ValueError('纸白需要三个RGB分量')
        checked_paper = [number(x, '纸白RGB', 0.01, 1) for x in paper]
    return {'inks': result, 'paper_rgb': checked_paper, 'model': 'multiplicative_rgb_transmittance', 'description': str(payload.get('description', ''))[:1000]}


def mix_colors(coverages, inks, paper):
    cov = np.asarray(coverages, dtype=np.float32)
    ink_arr = np.asarray(inks, dtype=np.float32)
    factors = 1.0 - cov[..., :, None] * (1.0 - ink_arr)
    return np.clip(np.asarray(paper, dtype=np.float32) * np.prod(factors, axis=-2), 0.0, 1.0)


def _lab(colors):
    flat = np.ascontiguousarray(colors, dtype=np.float32).reshape(-1, 1, 3)
    lab = cv2.cvtColor(flat, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    lab[:, 0] *= 100.0 / 255.0
    lab[:, 1:] -= 128.0
    return lab


def delta_e(first, second):
    return np.linalg.norm(_lab(first) - _lab(second), axis=1)


def _project_capped(coverages, limit):
    result = np.clip(coverages, 0.0, 1.0)
    over = result.sum(axis=1) > limit
    if not np.any(over):
        return result
    selected = result[over]
    low = selected.min(axis=1) - 1.0
    high = selected.max(axis=1)
    for _ in range(50):
        mid = 0.5 * (low + high)
        total = np.clip(selected - mid[:, None], 0.0, 1.0).sum(axis=1)
        too_high = total > limit
        low = np.where(too_high, mid, low)
        high = np.where(too_high, high, mid)
    result[over] = np.clip(selected - high[:, None], 0.0, 1.0)
    return result


def _objective(coverages, target, inks, paper, cost_weight, lam):
    factors = 1.0 - coverages[:, :, None] * (1.0 - inks)[None, :, :]
    result = paper[None, :] * np.prod(factors, axis=1)
    diff = result - target
    energy = 0.5 * np.einsum('mc,mc->m', diff, diff) + lam * (coverages * cost_weight[None, :]).sum(axis=1)
    ratio = (inks - 1.0)[None, :, :] / np.maximum(factors, 1e-6)
    weighted = diff * result
    gradient = lam * np.broadcast_to(cost_weight[None, :], coverages.shape).copy()
    for channel in range(3):
        gradient += weighted[:, channel][:, None] * ratio[:, :, channel]
    return energy, gradient


def _linear_init(target, inks, paper, limit):
    paper = np.asarray(paper, dtype=np.float32)
    matrix = 1.0 - inks.T
    pseudo = np.linalg.pinv(matrix)
    shift = np.log(paper[None, :] / np.clip(target, 1e-4, 1.0))
    coverages = np.clip(shift @ pseudo.T, 0.0, 1.0).astype(np.float32)
    return _project_capped(coverages, limit)


def _solve_chunk(target, inks, paper, limit, cost_weight, lam, iterations, init=None):
    paper = np.asarray(paper, dtype=np.float32)
    coverages = _linear_init(target, inks, paper, limit) if init is None else np.asarray(init, dtype=np.float32).copy()
    step = np.full(len(target), 0.4, dtype=np.float32)
    energy, gradient = _objective(coverages, target, inks, paper, cost_weight, lam)
    for _ in range(iterations):
        trial = _project_capped(coverages - step[:, None] * gradient, limit)
        trial_energy, trial_gradient = _objective(trial, target, inks, paper, cost_weight, lam)
        better = trial_energy < energy - 1e-7
        coverages = np.where(better[:, None], trial, coverages)
        energy = np.where(better, trial_energy, energy)
        gradient = np.where(better[:, None], trial_gradient, gradient)
        step = np.clip(np.where(better, step * 1.3, step * 0.5), 1e-3, 4.0)
    return coverages


def _warm_init(target, inks, paper, limit, cost_weight, lam):
    quantized = (np.clip(target, 0, 1) * 255).astype(np.uint8).astype(np.int32)
    bins = (quantized[:, 0] >> 3) * 1024 + (quantized[:, 1] >> 3) * 32 + (quantized[:, 2] >> 3)
    counts = np.bincount(bins, minlength=32768)
    order = np.argsort(-counts, kind='stable')
    chosen = order[counts[order] > 0][:PALETTE_SIZE]
    red = chosen // 1024
    green = (chosen // 32) % 32
    blue = chosen % 32
    colors = (np.stack([red, green, blue], axis=1) * 8 + 4).astype(np.float32) / 255.0
    palette_cov = _solve_chunk(colors, inks, paper, limit, cost_weight, lam, REFINE_ITERATIONS)
    rank = np.full(32768, -1, dtype=np.int64)
    rank[chosen] = np.arange(len(chosen))
    mapping = rank[bins]
    init = _linear_init(target, inks, paper, limit)
    known = mapping >= 0
    init[known] = palette_cov[mapping[known]]
    return init


def solve_coverages(image, inks, paper, total_limit, cost_weight, lam, iterations=FULL_ITERATIONS):
    target = rgb_image(image).reshape(-1, 3)
    ink_arr = np.asarray(inks, dtype=np.float32)
    limit = min(float(total_limit), float(len(ink_arr)))
    init = _warm_init(target, ink_arr, paper, limit, cost_weight, lam)
    output = np.empty((len(target), len(ink_arr)), dtype=np.float32)
    for start in range(0, len(target), FULL_CHUNK):
        end = start + FULL_CHUNK
        output[start:end] = _solve_chunk(target[start:end], ink_arr, paper, limit, cost_weight, lam, iterations, init[start:end])
    return output.reshape(image.shape[0], image.shape[1], len(ink_arr))


def _palette(images):
    indices = []
    for image in images:
        quantized = (np.clip(image, 0, 1) * 255).astype(np.uint8).astype(np.int32)
        indices.append(((quantized[..., 0] >> 3) * 1024 + (quantized[..., 1] >> 3) * 32 + (quantized[..., 2] >> 3)).ravel())
    counts = np.bincount(np.concatenate(indices), minlength=32768)
    order = np.argsort(-counts, kind='stable')
    chosen = order[counts[order] > 0][:PALETTE_SIZE]
    red = chosen // 1024
    green = (chosen // 32) % 32
    blue = chosen % 32
    colors = (np.stack([red, green, blue], axis=1) * 8 + 4).astype(np.float32) / 255.0
    return colors, counts[chosen].astype(np.float64)


def _key_samples(images, key_regions):
    colors = []
    weights = []
    slices = []
    cursor = 0
    for image, regions in zip(images, key_regions):
        for region in regions:
            mask = geometry_mask(region['geometry'], image.shape)
            pixels = image[mask]
            if len(pixels) > KEY_SAMPLES:
                pick = np.linspace(0, len(pixels) - 1, KEY_SAMPLES).round().astype(np.int64)
                pixels = pixels[pick]
            colors.append(pixels)
            weights.append(np.full(len(pixels), mask.sum() / len(pixels), dtype=np.float64))
            slices.append({'start': cursor, 'end': cursor + len(pixels), 'tolerance': region['tolerance'], 'name': region['name']})
            cursor += len(pixels)
    return np.concatenate(colors), np.concatenate(weights), slices


def _usage_grams(coverage_pixels, density, pixel_area_mm2, film_microns, copies):
    return coverage_pixels * pixel_area_mm2 * film_microns * 1e-6 * density * copies


def _evaluate_subset(subset, ink_arr, cost_weight, colors, weights, palette_count, slices, paper, limit, lam, iterations, usage_scale):
    inks = ink_arr[list(subset)]
    coverages = _solve_chunk(colors, inks, paper, min(limit, len(subset)), cost_weight[list(subset)], lam, iterations)
    preview = mix_colors(coverages, inks, paper)
    errors = delta_e(colors, preview)
    ratios = []
    region_means = []
    for item in slices:
        mean_error = float(errors[item['start']:item['end']].mean())
        region_means.append(mean_error)
        ratios.append(mean_error / item['tolerance'])
    palette_mean = float(np.average(errors[:palette_count], weights=weights[:palette_count])) if palette_count else 0.0
    coverage_pixels = (coverages * weights[:, None]).sum(axis=0)
    grams = coverage_pixels * usage_scale['pixel_area'] * usage_scale['film'] * 1e-6 * usage_scale['densities'][list(subset)] * usage_scale['copies']
    cost = float((grams / 1000.0 * usage_scale['prices'][list(subset)]).sum())
    return {'max_ratio': max(ratios), 'key_mean': float(np.mean(region_means)), 'palette_mean': palette_mean, 'cost': cost}


def _worst_colors(image, mask, errors, tolerance, limit=5):
    values = errors[mask]
    pixels = image[mask]
    failing = values > tolerance
    if not np.any(failing):
        return []
    values = values[failing]
    pixels = pixels[failing]
    if len(values) > 4096:
        pick = np.linspace(0, len(values) - 1, 4096).round().astype(np.int64)
        values = values[pick]
        pixels = pixels[pick]
    order = np.argsort(-values, kind='stable')
    labs = _lab(pixels)
    chosen = []
    chosen_lab = []
    for index in order:
        if chosen and float(np.min(np.linalg.norm(np.asarray(chosen_lab) - labs[index], axis=1))) < 4.0:
            continue
        chosen.append({'rgb': [float(x) for x in pixels[index]], 'delta_e': float(values[index])})
        chosen_lab.append(labs[index])
        if len(chosen) >= limit:
            break
    return chosen


def _region_metrics(image, errors, regions):
    rows = []
    union = np.zeros(image.shape[:2], dtype=bool)
    for region in regions:
        mask = geometry_mask(region['geometry'], image.shape)
        union |= mask
        values = errors[mask]
        rows.append({
            'name': region['name'],
            'tolerance': region['tolerance'],
            'pixels': int(mask.sum()),
            'mean': float(values.mean()),
            'p95': float(np.percentile(values, 95)),
            'max': float(values.max()),
            'pass': bool(values.mean() <= region['tolerance']),
        })
    other = ~union
    other_mean = float(errors[other].mean()) if np.any(other) else None
    return rows, other_mean


def _full_plan(store, job, assets, images, key_regions, inks, subset, paper, constraints, lam, cost_weight):
    ink_arr = np.asarray([ink['rgb'] for ink in inks], dtype=np.float32)
    sub_inks = ink_arr[list(subset)]
    sub_weight = cost_weight[list(subset)]
    pixel_area = (25.4 / job['dpi']) ** 2
    entries = []
    coverages_all = []
    totals = {'grams': 0.0, 'cost': 0.0}
    failures = []
    max_ratio = 0.0
    for asset, image, regions in zip(assets, images, key_regions):
        coverages = solve_coverages(image, sub_inks, paper, constraints['total_ink_limit'], sub_weight, lam)
        quantized = coverages.astype(np.float16).astype(np.float32)
        limit = min(constraints['total_ink_limit'], float(len(subset)))
        coverages = _project_capped(quantized.reshape(-1, len(subset)), limit).reshape(image.shape[0], image.shape[1], len(subset))
        blob = store.write_array(coverages)
        coverages_all.append(coverages)
        preview = mix_colors(coverages, sub_inks, paper)
        errors = delta_e(image.reshape(-1, 3), preview.reshape(-1, 3)).reshape(image.shape[:2])
        region_rows, other_mean = _region_metrics(image, errors, regions)
        usage = []
        for position, index in enumerate(subset):
            ink = inks[index]
            coverage_pixels = float(coverages[..., position].sum())
            grams = _usage_grams(coverage_pixels, ink['density'], pixel_area, constraints['film_microns'], constraints['copies'])
            cost = grams / 1000.0 * ink['cost_per_kg']
            totals['grams'] += grams
            totals['cost'] += cost
            usage.append({'ink': ink['name'], 'mean_coverage_percent': float(coverages[..., position].mean() * 100), 'coverage_pixels': coverage_pixels, 'grams': grams, 'cost': cost})
        for region, row in zip(regions, region_rows):
            max_ratio = max(max_ratio, row['mean'] / region['tolerance'])
            if not row['pass']:
                mask = geometry_mask(region['geometry'], image.shape)
                failures.append({'asset_id': asset['id'], 'image': asset['name'], 'region': region['name'], 'tolerance': region['tolerance'], 'achieved_mean': row['mean'], 'achieved_p95': row['p95'], 'worst_colors': _worst_colors(image, mask, errors, region['tolerance'])})
        entries.append({'asset_id': asset['id'], 'name': asset['name'], 'coverage_blob': blob, 'key_regions': region_rows, 'other_mean_delta_e': other_mean, 'whole_mean_delta_e': float(errors.mean()), 'usage': usage})
    plan = {'ink_indexes': list(subset), 'inks': [inks[index] for index in subset], 'lambda_cost': lam, 'qualified': not failures, 'key_region_max_ratio': max_ratio, 'images': entries, 'totals': totals, 'failures': failures}
    return plan, coverages_all


def _compare_identical(first, second, first_cov, second_cov):
    if first['ink_indexes'] != second['ink_indexes']:
        return False
    for left, right in zip(first_cov, second_cov):
        if left.shape != right.shape or float(np.abs(left - right).max()) > IDENTICAL_COVERAGE:
            return False
    if abs(first['key_region_max_ratio'] - second['key_region_max_ratio']) > IDENTICAL_DELTA_E:
        return False
    cost_a = first['totals']['cost']
    cost_b = second['totals']['cost']
    return abs(cost_a - cost_b) <= IDENTICAL_COST_RELATIVE * max(cost_a, cost_b, 1e-9)


def compare_plans(store, job, assets, images, key_regions, inks, paper, constraints):
    ink_arr = np.asarray([ink['rgb'] for ink in inks], dtype=np.float32)
    cost_weight = np.asarray([ink['cost_per_kg'] * ink['density'] for ink in inks], dtype=np.float32)
    cost_weight = cost_weight / max(float(cost_weight.max()), 1e-9)
    palette_colors, palette_weights = _palette(images)
    sample_colors, sample_weights, slices = _key_samples(images, key_regions)
    colors = np.concatenate([palette_colors, sample_colors])
    weights = np.concatenate([palette_weights, sample_weights])
    usage_scale = {'pixel_area': (25.4 / job['dpi']) ** 2, 'film': constraints['film_microns'], 'copies': constraints['copies'], 'densities': np.asarray([ink['density'] for ink in inks]), 'prices': np.asarray([ink['cost_per_kg'] for ink in inks])}
    limit = constraints['total_ink_limit']
    subsets = [combo for size in range(1, constraints['max_inks'] + 1) for combo in itertools.combinations(range(len(inks)), size)]
    coarse = [_evaluate_subset(subset, ink_arr, cost_weight, colors, weights, len(palette_colors), slices, paper, limit, LAM_ACCURATE, COARSE_ITERATIONS, usage_scale) for subset in subsets]
    feasible = [i for i, row in enumerate(coarse) if row['max_ratio'] <= 1.0]
    by_error = sorted(range(len(subsets)), key=lambda i: (round(coarse[i]['max_ratio'], 4), round(coarse[i]['key_mean'], 4), coarse[i]['cost']))[:8]
    by_cost = sorted(feasible, key=lambda i: (coarse[i]['cost'], coarse[i]['max_ratio']))[:8]
    shortlist = sorted(set(by_error) | set(by_cost))
    refined = {i: _evaluate_subset(subsets[i], ink_arr, cost_weight, colors, weights, len(palette_colors), slices, paper, limit, LAM_ACCURATE, REFINE_ITERATIONS, usage_scale) for i in shortlist}
    accurate_idx = min(shortlist, key=lambda i: (round(refined[i]['max_ratio'], 4), round(refined[i]['key_mean'], 4), refined[i]['cost']))
    refined_feasible = [i for i in shortlist if refined[i]['max_ratio'] <= 1.0]
    if refined_feasible:
        saving_idx = min(refined_feasible, key=lambda i: (refined[i]['cost'], refined[i]['max_ratio']))
    else:
        saving_idx = min(shortlist, key=lambda i: (refined[i]['cost'], refined[i]['max_ratio']))
    lam_saving = LAM_ACCURATE
    for lam in LAM_SAVING_CANDIDATES:
        trial = _evaluate_subset(subsets[saving_idx], ink_arr, cost_weight, colors, weights, len(palette_colors), slices, paper, limit, lam, REFINE_ITERATIONS, usage_scale)
        if trial['max_ratio'] <= 0.98:
            lam_saving = lam
            refined[saving_idx] = trial
        else:
            break
    combos = {'accurate': (subsets[accurate_idx], LAM_ACCURATE), 'saving': (subsets[saving_idx], lam_saving)}
    cache = {}
    plans = []
    for plan_id, title in (('accurate', '更接近重点颜色'), ('saving', '更省油墨')):
        combo = combos[plan_id]
        if combo not in cache:
            cache[combo] = _full_plan(store, job, assets, images, key_regions, inks, combo[0], paper, constraints, combo[1], cost_weight)
        plan, plan_coverages = cache[combo]
        plan = dict(plan)
        plan['plan_id'] = plan_id
        plan['title'] = title
        plan['_coverages'] = plan_coverages
        plans.append(plan)
    identical = _compare_identical(plans[0], plans[1], plans[0]['_coverages'], plans[1]['_coverages'])
    identical_note = '两套方案使用相同油墨组合，且各图覆盖率、重点区域误差与用量成本实质相同；确认任意一套得到的结果一致。' if identical else None
    for plan in plans:
        del plan['_coverages']
        if not plan['qualified']:
            plan['note'] = '现有油墨或版数上限下，以下重点区域达不到容差，不能作为合格方案开版。'
    return {
        'solver': SOLVER_VERSION,
        'model': MODEL_NOTE,
        'delta_e_unit': DELTA_E_NOTE,
        'subsets_evaluated': len(subsets),
        'feasible_subsets': len(feasible),
        'plans': plans,
        'identical': identical,
        'identical_note': identical_note,
        'limitations': ['分色在 RGB 透射率相乘模型下求解，非光谱或 ICC 流程', '误差按 CIE76 在 sRGB 假设下度量', '用量按覆盖率面积 × 墨膜 × 密度估算，不含洗机与残墨'],
    }


def _validate_key_regions(asset, entries):
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_KEY_REGIONS:
        raise ValueError('每张图需要1到%d个重点区域' % MAX_KEY_REGIONS)
    known = {region['id']: region for region in asset.get('regions', [])}
    result = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('重点区域条目必须为对象')
        tolerance = number(entry.get('tolerance'), '重点区域可接受误差(ΔE)', 0.1, 200, 3.0)
        if entry.get('region_id'):
            region = known.get(entry['region_id'])
            if region is None:
                raise ValueError('重点区域不属于图像' + asset['name'])
            geometry = region['geometry']
            name = region['name']
        else:
            geometry = validate_geometry(entry.get('geometry'), (asset['height'], asset['width']))
            name = str(entry.get('name') or '重点区域').strip()[:80] or '重点区域'
        result.append({'name': name, 'geometry': geometry, 'tolerance': tolerance})
    return result


def create_batch(store, job_id, payload):
    job = store.get_job(job_id)
    source_ids = payload.get('source_ids')
    if not isinstance(source_ids, list) or not 1 <= len(source_ids) <= MAX_IMAGES or len(set(source_ids)) != len(source_ids):
        raise ValueError('一批需要1到%d张不同的原稿' % MAX_IMAGES)
    assets = []
    for asset_id in source_ids:
        asset = store.get_asset(asset_id)
        if asset['job_id'] != job_id:
            raise ValueError('原稿不属于当前作业')
        if asset['channels'] != 3:
            raise ValueError('专色批分色需要RGB原稿：' + asset['name'])
        assets.append(asset)
    profile_id = payload.get('ink_profile_id') or None
    if payload.get('inks') is not None:
        library = validate_spot_inks({'inks': payload.get('inks')})
    elif profile_id:
        profile = store.get_profile(profile_id)
        if profile['job_id'] != job_id or profile['kind'] != 'spot_inks':
            raise ValueError('所选配置不是本作业的专色油墨库')
        library = validate_spot_inks(profile['payload'])
    else:
        raise ValueError('请登记2到8种可用油墨，或选择已保存的专色油墨库')
    inks = library['inks']
    paper_value = payload.get('paper_rgb', library.get('paper_rgb') or [0.97, 0.96, 0.92])
    if not isinstance(paper_value, (list, tuple)) or len(paper_value) != 3:
        raise ValueError('纸白需要三个RGB分量')
    paper = [number(x, '纸白RGB', 0.01, 1) for x in paper_value]
    max_inks = integer(payload.get('max_inks'), '最多使用油墨种数', 1, len(inks), min(4, len(inks)))
    total_limit = number(payload.get('total_ink_limit'), '总墨量上限', 0.1, float(max_inks), min(3.0, float(max_inks)))
    constraints = {
        'max_inks': max_inks,
        'total_ink_limit': total_limit,
        'copies': integer(payload.get('copies'), '印刷份数', 1, 10_000_000, 1),
        'film_microns': number(payload.get('film_microns'), '实地墨膜厚度', 0.01, 100, 1.0),
    }
    region_payload = payload.get('key_regions') or {}
    if not isinstance(region_payload, dict):
        raise ValueError('重点区域必须按图像ID组织')
    unknown = set(region_payload) - set(source_ids)
    if unknown:
        raise ValueError('重点区域指向未选择的图像')
    key_regions = []
    for asset in assets:
        entries = region_payload.get(asset['id'])
        if entries is None:
            raise ValueError('请为每张原稿圈出至少一个重点区域：' + asset['name'])
        key_regions.append(_validate_key_regions(asset, entries))
    images = [rgb_image(store.load_array(asset['id'])) for asset in assets]
    plans = compare_plans(store, job, assets, images, key_regions, inks, paper, constraints)
    status = 'compared' if any(plan['qualified'] for plan in plans['plans']) else 'unfeasible'
    spec = {
        'solver': plans['solver'],
        'inks': inks,
        'ink_profile_id': profile_id,
        'paper_rgb': paper,
        'constraints': constraints,
        'images': [{'asset_id': asset['id'], 'name': asset['name'], 'width': asset['width'], 'height': asset['height'], 'key_regions': regions} for asset, regions in zip(assets, key_regions)],
    }
    return store.create_spot_batch(job_id, payload.get('name') or '专色油墨批', spec, plans, status)


def _load_coverages(store, entry, height, width, ink_count):
    array = store.read_array(entry['coverage_blob'])
    if array.shape != (height, width, ink_count):
        raise RuntimeError('批次覆盖率数据与记录不符')
    return array


def _error_map_image(errors, regions, shape):
    scale = max(6.0, max(region['tolerance'] for region in regions))
    mapped = heatmap(errors, maximum=scale)
    for region in regions:
        mapped = mask_outline(mapped, geometry_mask(region['geometry'], shape), color=(1, 1, 1))
    return mapped


def plan_preview_png(store, batch_id, plan_id, asset_id, kind):
    batch = store.get_spot_batch(batch_id)
    plan = next((item for item in batch['plans']['plans'] if item['plan_id'] == plan_id), None)
    if plan is None:
        raise KeyError('方案不存在')
    entry = next((item for item in plan['images'] if item['asset_id'] == asset_id), None)
    if entry is None:
        raise KeyError('该图像不在此方案中')
    spec_image = next(item for item in batch['spec']['images'] if item['asset_id'] == asset_id)
    coverages = _load_coverages(store, entry, spec_image['height'], spec_image['width'], len(plan['inks']))
    ink_arr = [ink['rgb'] for ink in plan['inks']]
    if kind == 'preview':
        return png_bytes(mix_colors(coverages, ink_arr, batch['spec']['paper_rgb']))
    if kind == 'error':
        source = rgb_image(store.load_array(asset_id))
        preview = mix_colors(coverages, ink_arr, batch['spec']['paper_rgb'])
        errors = delta_e(source.reshape(-1, 3), preview.reshape(-1, 3)).reshape(source.shape[:2])
        return png_bytes(_error_map_image(errors, spec_image['key_regions'], source.shape))
    raise ValueError('预览类型支持 preview 或 error')


def confirm_batch(store, batch_id, plan_id):
    batch = store.get_spot_batch(batch_id)
    if batch['status'] == 'confirmed':
        raise ValueError('该批次已确认开版，配方与结果已锁定')
    plan = next((item for item in batch['plans']['plans'] if item['plan_id'] == plan_id), None)
    if plan is None:
        raise KeyError('方案不存在')
    if not plan['qualified']:
        reasons = '；'.join('%s/%s 平均ΔE %.2f > 容差 %.2f' % (item['image'], item['region'], item['achieved_mean'], item['tolerance']) for item in plan['failures'])
        raise ValueError('该方案未满足重点区域容差，不能作为合格方案开版：' + reasons)
    spec = batch['spec']
    paper = spec['paper_rgb']
    ink_arr = [ink['rgb'] for ink in plan['inks']]
    outputs = []
    usage_rows = []
    for entry, spec_image in zip(plan['images'], spec['images']):
        source = store.get_asset(entry['asset_id'])
        coverages = _load_coverages(store, entry, spec_image['height'], spec_image['width'], len(plan['inks']))
        plates = []
        for position, ink in enumerate(plan['inks']):
            plate = coverages[..., position]
            tinted = 1.0 - plate[..., None] * (1.0 - np.asarray(ink['rgb'], dtype=np.float32))
            metadata = {'spot_batch_id': batch_id, 'plan_id': plan_id, 'role': 'spot_plate', 'ink_name': ink['name'], 'ink_rgb': ink['rgb'], 'source_id': source['id']}
            asset = store.add_asset(batch['job_id'], '%s / %s版' % (entry['name'], ink['name']), 'gray', plate, metadata, source['id'], preview=np.clip(tinted, 0, 1))
            plates.append({'ink': ink['name'], 'asset_id': asset['id']})
        for row in entry['usage']:
            usage_rows.append({'image': entry['name'], 'ink': row['ink'], 'mean_coverage_percent': row['mean_coverage_percent'], 'grams': row['grams'], 'cost': row['cost']})
        preview = mix_colors(coverages, ink_arr, paper)
        preview_asset = store.add_asset(batch['job_id'], entry['name'] + ' / 专色叠印预览', 'rgb', preview, {'spot_batch_id': batch_id, 'plan_id': plan_id, 'role': 'spot_preview', 'source_id': source['id']}, source['id'])
        source_rgb = rgb_image(store.load_array(entry['asset_id']))
        errors = delta_e(source_rgb.reshape(-1, 3), preview.reshape(-1, 3)).reshape(source_rgb.shape[:2])
        error_asset = store.add_asset(batch['job_id'], entry['name'] + ' / 专色误差图', 'rgb', _error_map_image(errors, spec_image['key_regions'], source_rgb.shape), {'spot_batch_id': batch_id, 'plan_id': plan_id, 'role': 'spot_error_map', 'source_id': source['id']}, source['id'])
        outputs.append({'source_asset_id': entry['asset_id'], 'name': entry['name'], 'plates': plates, 'preview_asset_id': preview_asset['id'], 'error_map_asset_id': error_asset['id']})
    confirmed = {'plan_id': plan_id, 'plan_title': plan['title'], 'outputs': outputs, 'usage_rows': usage_rows, 'totals': plan['totals'], 'note': '已按比较时的配方快照开版；后续修改油墨库或重算其他标签不会改变本批结果。'}
    return store.confirm_spot_batch(batch_id, confirmed)


def usage_rows(batch):
    if batch['status'] != 'confirmed' or not batch.get('confirmed'):
        raise ValueError('批次尚未确认开版，没有已锁定的用量统计')
    rows = list(batch['confirmed']['usage_rows'])
    totals = batch['confirmed']['totals']
    rows.append({'image': '合计', 'ink': '全部油墨', 'mean_coverage_percent': None, 'grams': totals['grams'], 'cost': totals['cost']})
    return rows
