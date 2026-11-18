"""Statistical print-sheet screening.

The screening engine never blurs the candidate image and never widens a single
global pixel threshold.  Each pixel keeps its own robust fluctuation envelope
learned from a group of already-corrected normal sheets (per-pixel median plus a
MAD scale with an absolute noise floor).  Candidate deviations become robust
z-scores; an optional *local* phase allowance only relaxes pixels whose
neighbourhood already contains the same deviation, so small broken strokes on
small type remain separable from ordinary halftone phase drift.

Whole sheets, never cropped patches, are the unit of learning and validation.
"""

import math

import cv2
import numpy as np

from .pixels import integer, number, rgb_image

POLICIES = {
    'low_false_alarm': {
        'label': '更少误报（复核优先）',
        'z_strong': 8.0,
        'z_weak': 5.0,
        'min_area_px': 12,
        'phase_allowance_px': 1,
        'morphology_gate': True,
        'min_component_mean_z': 10.0,
        'target_sheet_false_alarm_rate': 0.05,
    },
    'high_sensitivity': {
        'label': '更少漏检（敏感优先）',
        'z_strong': 5.0,
        'z_weak': 3.5,
        'min_area_px': 4,
        'phase_allowance_px': 0,
        'morphology_gate': False,
        'min_component_mean_z': 0.0,
        'target_sheet_false_alarm_rate': 0.10,
    },
}

DEFECT_KINDS = ('missing_print', 'spot', 'broken_stroke', 'other')
SHEET_ROLES = ('normal', 'defect', 'candidate')
REGION_ROLES = ('exclude', 'known_defect')
# MAD -> consistent standard-deviation estimator for normally distributed data.
MAD_NORMAL_FACTOR = 1.4826


def _rgb_stack(sheets, name):
    if not isinstance(sheets, (list, tuple)) or not sheets:
        raise ValueError(f'{name}至少需要一张图像')
    arrays = []
    shape = None
    for index, sheet in enumerate(sheets):
        image = rgb_image(sheet)
        if shape is None:
            shape = image.shape
        elif image.shape != shape:
            raise ValueError(f'{name}中第{index + 1}张尺寸与其他图像不同；请先完成几何与颜色校正')
        arrays.append(np.ascontiguousarray(image, dtype=np.float32))
    return arrays


def _finite_mask_from_annotations(shape, annotations):
    """Return (valid mask, exclude mask, defect annotations) from plan regions."""
    height, width = shape
    valid = np.ones((height, width), dtype=bool)
    excluded = np.zeros((height, width), dtype=bool)
    defects = []
    for item in annotations or []:
        role = item.get('role')
        if role not in REGION_ROLES:
            continue
        current = _annotation_mask(item.get('geometry'), shape)
        if role == 'exclude':
            valid &= ~current
            excluded |= current
        else:
            defects.append(item)
    return valid, excluded, defects


def _annotation_mask(geometry, shape):
    if not isinstance(geometry, dict):
        raise ValueError('标注区域必须是几何对象')
    height, width = shape
    kind = geometry.get('type')
    if kind == 'rectangle':
        x = integer(geometry.get('x'), '标注x', 0, width - 1)
        y = integer(geometry.get('y'), '标注y', 0, height - 1)
        region_width = integer(geometry.get('width'), '标注宽度', 1, width - x)
        region_height = integer(geometry.get('height'), '标注高度', 1, height - y)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y:y + region_height, x:x + region_width] = 1
        return mask.astype(bool)
    if kind == 'polygon':
        points = geometry.get('points')
        if not isinstance(points, list) or len(points) < 3 or len(points) > 256:
            raise ValueError('多边形标注需要3到256个点')
        contour = np.asarray([[integer(p[0], '点x', 0, width - 1), integer(p[1], '点y', 0, height - 1)] for p in points], dtype=np.int32)
        if abs(cv2.contourArea(contour)) < 1:
            raise ValueError('多边形标注没有有效面积')
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 1)
        return mask.astype(bool)
    raise ValueError('标注区域支持矩形或多边形')


def _resolve_policy(policy):
    if isinstance(policy, str):
        if policy not in POLICIES:
            raise ValueError('判定取向须为 low_false_alarm 或 high_sensitivity')
        return dict(POLICIES[policy])
    if not isinstance(policy, dict):
        raise ValueError('判定策略必须是对象或预设名称')
    selected = dict(POLICIES['low_false_alarm'])
    for key in ('z_strong', 'z_weak', 'min_area_px', 'phase_allowance_px'):
        if key in policy and policy[key] is not None:
            selected[key] = policy[key]
    selected['z_strong'] = number(selected['z_strong'], '强缺陷z阈值', 2.0, 50.0)
    selected['z_weak'] = number(selected['z_weak'], '弱差异z阈值', 1.5, 50.0)
    if selected['z_weak'] >= selected['z_strong']:
        raise ValueError('弱差异z阈值必须低于强缺陷z阈值')
    selected['min_area_px'] = integer(selected['min_area_px'], '最小缺陷面积', 1, 1_000_000)
    selected['phase_allowance_px'] = integer(selected['phase_allowance_px'], '局部相位容差', 0, 6)
    selected['morphology_gate'] = bool(selected.get('morphology_gate', True))
    selected['min_component_mean_z'] = number(selected.get('min_component_mean_z', 0.0), '组件平均z下限', 0.0, 50.0)
    return selected


def build_envelope(normal_sheets, noise_floor=0.02, min_support=3):
    """Per-pixel robust fluctuation envelope from corrected normal sheets.

    Returns float32 planes: median (HxWx3), scale (HxWx3), support (HxW),
    plus a summary dictionary.  ``scale`` is the MAD-based dispersion floored
    by ``noise_floor`` so that dead-stable paper areas keep a finite envelope
    without becoming a blanket tolerance for the whole sheet.
    """
    arrays = _rgb_stack(normal_sheets, '正常印张组')
    noise_floor = number(noise_floor, '噪声下限', 0.001, 0.25)
    min_support = integer(min_support, '最少支撑样本', 2, max(2, len(arrays)))
    stack = np.stack(arrays, axis=0)
    count = stack.shape[0]
    median = np.median(stack, axis=0).astype(np.float32)
    mad = np.median(np.abs(stack - median[None, ...]), axis=0).astype(np.float32)
    scale = np.maximum(mad * MAD_NORMAL_FACTOR, np.float32(noise_floor))
    support = np.full(median.shape[:2], count, dtype=np.uint16)
    valid_pixels = support >= min_support
    # Small-sample MAD under-estimates dispersion; apply a conservative factor.
    if count < 6:
        small_sample_factor = 1.0 + (6 - count) * 0.18
        scale = scale * np.float32(small_sample_factor)
    flat_scale = scale.reshape(-1, 3)
    summary = {
        'normal_sheet_count': count,
        'noise_floor': noise_floor,
        'min_support': min_support,
        'small_sample_factor_applied': float(small_sample_factor) if count < 6 else 1.0,
        'median_scale': float(np.median(flat_scale)),
        'scale_p95': float(np.percentile(flat_scale, 95)),
        'scale_p05': float(np.percentile(flat_scale, 5)),
        'pixels_with_envelope': int(valid_pixels.sum()),
        'pixels_without_envelope': int((~valid_pixels).sum()),
        'limitation': '逐像素中位数/MAD包络仅覆盖正常印张出现过的位置与阶调；新图案、未覆盖区域按证据不足处理',
    }
    return {'median': median.astype(np.float32), 'scale': scale.astype(np.float32),
            'support': support, 'summary': summary}


def deviation_planes(candidate, envelope):
    """Candidate versus envelope: robust z and signed channel deviation."""
    image = rgb_image(candidate)
    median = envelope['median']
    scale = envelope['scale']
    if median.shape != image.shape:
        raise ValueError('候选印张与正常包络尺寸不同')
    signed = (image - median).astype(np.float32)
    z = np.abs(signed) / np.maximum(scale, np.float32(1e-6))
    z_max = z.max(axis=2).astype(np.float32)
    direction = np.zeros(image.shape[:2], dtype=np.uint8)
    direction[signed.mean(axis=2) < 0] = 1   # 偏浅/缺墨
    direction[signed.mean(axis=2) > 0] = 2   # 偏深/污点
    return {'z': z.astype(np.float32), 'z_max': z_max, 'signed': signed, 'direction': direction}


def _apply_phase_allowance(z_max, radius):
    """Local phase allowance against small halftone/registration drift.

    A greyvalue erosion (local minimum over a small disk) lets a pixel match
    the smallest deviation already present nearby: a dot shifted by one pixel
    is normal somewhere in its own neighbourhood.  Unlike Gaussian blur or a
    global threshold widening, this cannot erase an isolated small break,
    whose whole neighbourhood stays high.  Radius 0 disables the allowance
    entirely (sensitive policy).
    """
    if radius <= 0:
        return z_max
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.erode(z_max, kernel).astype(np.float32)


def _coherent_strong(raw_z, threshold):
    """Dense-cluster mask of raw deviations above ``threshold``.

    Phase drift produces scattered edge pixels that rarely touch each other;
    a real missing/printed feature (e.g. a broken glyph stroke) fills a solid
    region whose pixels almost all have a strong neighbour.  We keep a strong
    pixel only when a strong neighbour exists, then close one-pixel gaps.
    Isolated noise -- including single-pixel halftone edge excursions -- is
    removed without any global blur.
    """
    raw_strong = (raw_z >= threshold).astype(np.uint8)
    neighbour = cv2.dilate(raw_strong, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    coherent = ((raw_strong & neighbour).astype(np.uint8))
    coherent = cv2.morphologyEx(coherent, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return coherent.astype(bool)


def _component_rows(binary, score, direction, max_regions=400):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    rows = []
    retained = np.zeros(binary.shape, dtype=bool)
    for identifier in range(1, count):
        area = int(stats[identifier, cv2.CC_STAT_AREA])
        if identifier > max_regions:
            break
        x = int(stats[identifier, cv2.CC_STAT_LEFT])
        y = int(stats[identifier, cv2.CC_STAT_TOP])
        width = int(stats[identifier, cv2.CC_STAT_WIDTH])
        height = int(stats[identifier, cv2.CC_STAT_HEIGHT])
        local_labels = labels[y:y + height, x:x + width] == identifier
        values = score[y:y + height, x:x + width][local_labels]
        local_direction = direction[y:y + height, x:x + width][local_labels]
        light = int(np.sum(local_direction == 1))
        dark = int(np.sum(local_direction == 2))
        rows.append({
            'label': int(identifier), 'x': x, 'y': y, 'width': width, 'height': height,
            'area_px': area, 'centroid_x': float(centroids[identifier, 0]),
            'centroid_y': float(centroids[identifier, 1]),
            'mean_z': float(values.mean()), 'max_z': float(values.max()),
            'p95_z': float(np.percentile(values, 95)),
            'light_pixels': light, 'dark_pixels': dark,
            'polarity': 'lighter_than_normal' if light >= dark else 'darker_than_normal',
            'fill_fraction': area / max(width * height, 1),
        })
        retained |= labels == identifier
    rows.sort(key=lambda row: (-row['max_z'], -row['area_px']))
    for rank, row in enumerate(rows, start=1):
        row['region'] = rank
    return rows, retained, labels


def _morphology_gate_failed(component_mask):
    """True when a strong component only looks like drift, not a printed defect.

    Phase/registration drift leaves either one-pixel-wide edge lines or sparse
    halftone-edge halos.  A real missing dot, spot or broken stroke keeps a
    solid core that survives a 3x3 opening, or fills a compact footprint.
    """
    area = int(component_mask.sum())
    if area == 0:
        return True
    ys, xs = np.nonzero(component_mask)
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    fill = area / max(width * height, 1)
    if fill < 0.10:
        return True
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(component_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return int(opened.sum()) < 0.4 * area


def _annotation_overlap(row, annotation, shape):
    mask = _annotation_mask(annotation['geometry'], shape)
    x, y, width, height = row['x'], row['y'], row['width'], row['height']
    local = mask[y:y + height, x:x + width]
    return float(local.mean()), int(local.sum())


def _heatmap(values, valid):
    finite = np.isfinite(values) & valid
    high = float(np.percentile(values[finite], 99)) if finite.any() else 1.0
    high = max(high, 1.0)
    scaled = np.zeros(values.shape, dtype=np.uint8)
    scaled[finite] = np.clip(values[finite] * 255.0 / high, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    image = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image[~finite] = (0.78, 0.78, 0.78)
    return image


def inspect_sheet(candidate, envelope, policy='low_false_alarm', annotations=None,
                  baseline=None, max_regions=400):
    """Classify one whole sheet into defect / normal / insufficient-evidence.

    ``annotations`` holds plan-level regions (excludes and known defects).
    ``baseline`` is the validation summary used to judge sheet-level drift.
    Returns component rows (with triage classes), summary, evidence images and
    per-pixel arrays used by the evidence viewer.
    """
    image = rgb_image(candidate)
    shape = image.shape[:2]
    settings = _resolve_policy(policy)
    valid, excluded, defect_notes = _finite_mask_from_annotations(shape, annotations)
    planes = deviation_planes(image, envelope)
    support = envelope['support'] >= envelope['summary'].get('min_support', 3)
    supported = valid & support
    unsupported = valid & ~support
    raw_z = planes['z_max']
    z_field = _apply_phase_allowance(raw_z, settings['phase_allowance_px'])
    # Policy-independent reference used only for sheet-level drift baselines.
    reference_strong = (raw_z >= 5.0) & supported
    if settings['phase_allowance_px'] > 0:
        # Tolerate scattered phase drift (erosion field) while retaining dense
        # fine-defect clusters such as broken glyph strokes via coherence.
        coherent = _coherent_strong(raw_z, settings['z_strong']) & supported
        strong = ((z_field >= settings['z_strong']) | coherent) & supported
        weak = ((z_field >= settings['z_weak']) | (_coherent_strong(raw_z, settings['z_weak']) & supported)) & supported
    else:
        strong = (raw_z >= settings['z_strong']) & supported
        weak = (raw_z >= settings['z_weak']) & supported
    # The displayed/evidence score keeps raw values so reviews see actual
    # local deviations rather than the eroded helper field.
    z_field = raw_z
    rows, retained, labels = _component_rows(strong | weak, z_field, planes['direction'], max_regions)

    # Per-component triage.
    for row in rows:
        component_mask = labels == row['label']
        strong_pixels = int((strong & component_mask).sum())
        weak_pixels = int((weak & component_mask).sum())
        unsupported_pixels = int((unsupported & component_mask).sum())
        gate_failed = bool(settings['morphology_gate'] and _morphology_gate_failed(component_mask))
        row['strong_pixels'] = strong_pixels
        row['weak_pixels'] = weak_pixels
        row['unsupported_pixels'] = unsupported_pixels
        row['morphology_gate_failed'] = gate_failed
        overlap_notes = []
        for note in defect_notes:
            fraction, pixels = _annotation_overlap(row, note, shape)
            if pixels > 0:
                overlap_notes.append({'annotation_id': note.get('id'), 'label': note.get('label'),
                                      'kind': note.get('kind'), 'overlap_fraction': fraction,
                                      'overlap_pixels': pixels})
        row['known_defect_matches'] = overlap_notes
        if unsupported_pixels > 0 and unsupported_pixels >= 0.25 * row['area_px']:
            row['class'] = 'insufficient_evidence'
            row['reason'] = '组件有相当部分位于正常样本未覆盖像素，支撑不足'
        elif strong_pixels >= settings['min_area_px'] and not gate_failed and row['mean_z'] >= settings['min_component_mean_z']:
            row['class'] = 'defect_review'
            row['reason'] = f"{strong_pixels}个像素超过强缺陷z阈值{settings['z_strong']:g}，形态紧凑且组件平均z={row['mean_z']:.1f}"
        elif strong_pixels >= settings['min_area_px'] and (gate_failed or row['mean_z'] < settings['min_component_mean_z']):
            row['class'] = 'insufficient_evidence'
            row['reason'] = '强差异仅为单像素边缘线、稀疏网纹边缘或平均对比不足，按正常波动/待人工复核处理'
        elif weak_pixels >= settings['min_area_px']:
            row['class'] = 'insufficient_evidence'
            row['reason'] = f"仅{weak_pixels}个像素达到弱差异z阈值{settings['z_weak']:g}，未达强缺陷判据"
        else:
            row['class'] = 'normal'
            row['reason'] = '差异像素少于最小缺陷面积，按正常网纹波动处理'

    # Sheet-level drift relative to validation baseline (whole-sheet statistics).
    sheet_stats = {
        'median_z': float(np.median(z_field[supported])) if supported.any() else None,
        'p95_z': float(np.percentile(z_field[supported], 95)) if supported.any() else None,
        'mean_signed_rgb': [float(v) for v in planes['signed'][supported].mean(axis=0)] if supported.any() else None,
        'strong_pixel_fraction': float(strong.sum() / max(supported.sum(), 1)),
        'reference_strong_fraction': float(reference_strong.sum() / max(supported.sum(), 1)),
        'unsupported_pixel_fraction': float(unsupported.sum() / max(valid.sum(), 1)),
    }
    out_of_range = False
    drift_reasons = []
    if baseline:
        limit = baseline.get('normal_p95_z_ceiling')
        if limit is not None and sheet_stats['p95_z'] is not None and sheet_stats['p95_z'] > limit:
            out_of_range = True
            drift_reasons.append(f"整张p95 z={sheet_stats['p95_z']:.2f} 超出正常验证上限{limit:.2f}")
        ceiling = baseline.get('normal_reference_strong_ceiling')
        if ceiling is not None and sheet_stats['reference_strong_fraction'] > ceiling:
            out_of_range = True
            drift_reasons.append(f"强差异像素占比{sheet_stats['reference_strong_fraction']:.4%}超出验证上限{ceiling:.4%}")
    if unsupported.sum() > 0.02 * max(valid.sum(), 1):
        out_of_range = True
        drift_reasons.append('超过2%的检查像素缺少足够正常样本支撑')

    classified = {'defect_review': 0, 'insufficient_evidence': 0, 'normal': 0}
    for row in rows:
        classified[row['class']] = classified.get(row['class'], 0) + 1
    summary = {
        'policy': settings.get('label', settings),
        'settings': settings,
        'valid_pixels': int(valid.sum()),
        'excluded_pixels': int(excluded.sum()),
        'unsupported_pixels': int(unsupported.sum()),
        'components': len(rows),
        **{f'{key}_components': value for key, value in classified.items()},
        'sheet_out_of_envelope_range': out_of_range,
        'drift_reasons': drift_reasons,
        'sheet_statistics': sheet_stats,
        'verdict': 'review_required' if (classified['defect_review'] or out_of_range) else
                   ('evidence_gap' if classified['insufficient_evidence'] or unsupported.any() else 'normal'),
        'limitation': '结果是筛查分流而非整张合格判定；像素级阈值不能直接解释为印张可靠合格',
    }

    # Evidence images.
    overlay = image.copy()
    defect_mask = np.isin(labels, [row['label'] for row in rows if row['class'] == 'defect_review'])
    weak_mask = np.isin(labels, [row['label'] for row in rows if row['class'] == 'insufficient_evidence'])
    overlay[defect_mask] = overlay[defect_mask] * 0.35 + np.array([0.95, 0.15, 0.1]) * 0.65
    overlay[weak_mask] = overlay[weak_mask] * 0.45 + np.array([0.95, 0.72, 0.15]) * 0.55
    overlay[excluded] = overlay[excluded] * 0.35 + np.array([0.55, 0.55, 0.6]) * 0.65
    overlay[unsupported & ~excluded] = overlay[unsupported & ~excluded] * 0.5 + np.array([0.45, 0.55, 0.95]) * 0.5
    for row in rows:
        if row['class'] == 'normal':
            continue
        color = (0.9, 0.05, 0.1) if row['class'] == 'defect_review' else (0.85, 0.55, 0.05)
        cv2.rectangle(overlay, (row['x'], row['y']), (row['x'] + row['width'] - 1, row['y'] + row['height'] - 1), color, 1)
    images = {
        '筛查叠图': np.clip(overlay, 0, 1),
        '稳健z热图': _heatmap(z_field, valid),
        '正常中位参考': envelope['median'],
        '局部尺度图': _heatmap(np.clip(envelope['scale'].mean(axis=2), 0, None), np.ones(shape, dtype=bool)),
        '排除与支撑图': _support_image(shape, valid, excluded, unsupported),
    }
    arrays = {'z_field': z_field, 'valid': valid, 'excluded': excluded,
              'unsupported': unsupported, 'labels': labels, 'median': envelope['median'],
              'scale': envelope['scale'], 'signed': planes['signed']}
    return {'summary': summary, 'images': images, 'tables': {'缺陷组件': rows}, 'arrays': arrays}


def _support_image(shape, valid, excluded, unsupported):
    image = np.ones(shape + (3,), dtype=np.float32)
    image[valid] = (0.96, 0.98, 0.95)
    image[unsupported & valid] = (0.55, 0.65, 0.95)
    image[excluded] = (0.5, 0.5, 0.55)
    return image


def match_known_defects(rows, annotations, shape):
    """Best detected component for each known-defect annotation (recall truth)."""
    notes = [item for item in (annotations or []) if item.get('role') == 'known_defect']
    matches = []
    for note in notes:
        best = None
        for row in rows:
            fraction, pixels = _annotation_overlap(row, note, shape)
            if pixels <= 0:
                continue
            candidate = {'annotation_id': note.get('id'), 'label': note.get('label'),
                         'kind': note.get('kind'), 'matched_region': row['region'],
                         'matched_class': row['class'], 'overlap_fraction': fraction,
                         'overlap_pixels': pixels, 'max_z': row['max_z'], 'area_px': row['area_px']}
            if best is None or candidate['overlap_pixels'] > best['overlap_pixels']:
                best = candidate
        if best is None:
            matches.append({'annotation_id': note.get('id'), 'label': note.get('label'),
                            'kind': note.get('kind'), 'matched_region': None,
                            'detected': False, 'reason': '没有任何差异组件与标注相交'})
        else:
            best['detected'] = best['matched_class'] in ('defect_review', 'insufficient_evidence')
            matches.append(best)
    return matches


def validate_plan(normal_sheets, defect_sheets, envelope_kwargs, annotations_by_sheet,
                  policies=None, min_support=3, normal_annotations=None):
    """Whole-sheet hold-out validation.

    Normal sheets are scored by leave-one-sheet-out envelopes so a sheet is
    never validated against an envelope it helped to learn.  Defect sheets use
    the full normal envelope and are matched against their known-defect
    annotations.  Both policy presets are evaluated on the same frozen groups.
    ``normal_annotations`` maps normal-sheet index to its exclude regions.
    """
    normals = _rgb_stack(normal_sheets, '正常印张组')
    defects = _rgb_stack(defect_sheets, '缺陷印张组') if defect_sheets else []
    if len(normals) < 2:
        raise ValueError('整张验证至少需要2张正常印张')
    normal_annotations = normal_annotations or {}
    selected_policies = policies or list(POLICIES.keys())
    if isinstance(selected_policies, str):
        selected_policies = [selected_policies]

    def held_envelope_for(index):
        training = normals[:index] + normals[index + 1:]
        kwargs = {k: v for k, v in envelope_kwargs.items() if k != 'min_support'}
        return build_envelope(training, min_support=min(2, min_support), **kwargs)

    # Leave-one-sheet-out normal validation (policy-independent baseline).
    holdout_p95 = []
    holdout_reference_strong = []
    for index in range(len(normals)):
        held_envelope = held_envelope_for(index)
        run = inspect_sheet(normals[index], held_envelope, policy=POLICIES[selected_policies[0]],
                            annotations=normal_annotations.get(index, []))
        stats = run['summary']['sheet_statistics']
        if stats['p95_z'] is not None:
            holdout_p95.append(stats['p95_z'])
        holdout_reference_strong.append(stats['reference_strong_fraction'])
    baseline = {
        'normal_p95_z_ceiling': float(np.max(holdout_p95)) if holdout_p95 else None,
        'normal_reference_strong_ceiling': float(np.max(holdout_reference_strong)) if holdout_reference_strong else None,
        'normal_sheet_count': len(normals),
    }

    full_envelope = build_envelope(normals, **envelope_kwargs)
    policy_results = {}
    for policy_name in selected_policies:
        settings = _resolve_policy(policy_name)
        # Normal-sheet false alarms per sheet (re-score with each policy).
        sheet_false_alarms = []
        for index in range(len(normals)):
            held_envelope = held_envelope_for(index)
            run = inspect_sheet(normals[index], held_envelope, policy=settings,
                                annotations=normal_annotations.get(index, []))
            false_components = [row for row in run['tables']['缺陷组件'] if row['class'] == 'defect_review']
            sheet_false_alarms.append({'sheet_index': index,
                                       'false_alarm_regions': len(false_components),
                                       'false_alarm_pixels': sum(row['area_px'] for row in false_components),
                                       'verdict': run['summary']['verdict']})
        false_alarm_sheets = sum(1 for item in sheet_false_alarms if item['false_alarm_regions'] > 0)
        # Rule of three: one-sided 95% upper bound when zero events observed.
        if false_alarm_sheets == 0:
            fa_upper = 3.0 / len(normals)
        else:
            fa_upper = min(1.0, (false_alarm_sheets / len(normals)) + 1.645 *
                           math.sqrt((false_alarm_sheets / len(normals)) * (1 - false_alarm_sheets / len(normals)) / len(normals)))
        # Defect-sheet detection per known defect.
        defect_runs = []
        all_matches = []
        for index, sheet in enumerate(defects):
            notes = annotations_by_sheet.get(index, [])
            run = inspect_sheet(sheet, full_envelope, policy=settings, annotations=notes, baseline=baseline)
            matches = match_known_defects(run['tables']['缺陷组件'], notes, sheet.shape[:2])
            defect_runs.append({'sheet_index': index, 'verdict': run['summary']['verdict'],
                                'defect_components': run['summary']['defect_review_components'],
                                'matches': matches})
            all_matches.extend(matches)
        detected = [m for m in all_matches if m.get('detected')]
        kinds = {}
        for match in all_matches:
            kinds.setdefault(match.get('kind') or 'other', []).append(match)
        kind_recall = {}
        for kind, items in kinds.items():
            hits = sum(1 for item in items if item.get('detected'))
            kind_recall[kind] = {'known': len(items), 'detected': hits,
                                 'recall': hits / len(items),
                                 'independent_validation': len(items) >= 2}
        if all_matches:
            recall = len(detected) / len(all_matches)
            # One-sided Clopper-Pearson-ish lower bound (normal approximation).
            if 0 < recall < 1:
                lower = max(0.0, recall - 1.645 * math.sqrt(recall * (1 - recall) / len(all_matches)))
            else:
                lower = recall
        else:
            recall = None
            lower = None
        target = settings['target_sheet_false_alarm_rate']
        policy_results[policy_name] = {
            'label': settings['label'],
            'settings': {key: settings[key] for key in ('z_strong', 'z_weak', 'min_area_px', 'phase_allowance_px', 'morphology_gate', 'min_component_mean_z')},
            'normal_sheet_false_alarms': sheet_false_alarms,
            'false_alarm_sheets': false_alarm_sheets,
            'false_alarm_rate_observed': false_alarm_sheets / len(normals),
            'false_alarm_rate_upper95': float(fa_upper),
            'meets_target_false_alarm': fa_upper <= target,
            'target_sheet_false_alarm_rate': target,
            'sample_sufficiency': _sample_sufficiency(len(normals), target),
            'defect_sheet_runs': defect_runs,
            'known_defect_total': len(all_matches),
            'known_defect_detected': len(detected),
            'recall_observed': recall,
            'recall_lower95': lower,
            'recall_by_kind': kind_recall,
            'unvalidated_kinds': [kind for kind, record in kind_recall.items() if not record['independent_validation']],
        }
    return {
        'baseline': baseline,
        'envelope_summary': full_envelope['summary'],
        'policies': policy_results,
        'normal_count': len(normals),
        'defect_count': len(defects),
        'holdout_protocol': '正常印张逐张留一（留一印张不参与该轮包络学习）；缺陷印张使用完整正常包络；整印张为最小单位，不裁块混用',
        'limitations': [
            '留一验证给出的是有限样本下的误报/检出区间，不是连续生产的保证',
            '只有1个标注实例的缺陷类别没有独立验证，按待复检处理',
            '新印张整张统计超出验证基线时整张降级为证据不足/待复检',
        ],
    }


def _sample_sufficiency(normal_count, target_rate):
    """Rule-of-three based check for claiming a target false-alarm rate."""
    needed = math.ceil(3.0 / target_rate)
    return {
        'normal_sheets': normal_count,
        'minimum_sheets_for_target': needed,
        'sufficient': normal_count >= needed,
        'note': f'按零误报三倍经验法则，声称单张误报率≤{target_rate:.0%}约需{needed}张正常印张；当前{normal_count}张',
    }


def local_evidence(candidate, envelope, component_row, padding=12):
    """Crop candidate / median / z / scale around one component for review."""
    image = rgb_image(candidate)
    height, width = image.shape[:2]
    x0 = max(0, component_row['x'] - padding)
    y0 = max(0, component_row['y'] - padding)
    x1 = min(width, component_row['x'] + component_row['width'] + padding)
    y1 = min(height, component_row['y'] + component_row['height'] + padding)
    planes = deviation_planes(image, envelope)
    crop_candidate = image[y0:y1, x0:x1]
    crop_median = envelope['median'][y0:y1, x0:x1]
    crop_z = planes['z_max'][y0:y1, x0:x1]
    crop_scale = envelope['scale'].mean(axis=2)[y0:y1, x0:x1]
    return {
        'box': {'x': x0, 'y': y0, 'width': x1 - x0, 'height': y1 - y0},
        'images': {
            '候选局部': crop_candidate,
            '正常中位局部': crop_median,
            'z分数局部': _heatmap(crop_z, np.ones(crop_z.shape, dtype=bool)),
            '正常尺度局部': _heatmap(crop_scale, np.ones(crop_z.shape, dtype=bool)),
        },
        'stats': {
            'candidate_mean_rgb': [float(v) for v in crop_candidate.reshape(-1, 3).mean(axis=0)],
            'normal_mean_rgb': [float(v) for v in crop_median.reshape(-1, 3).mean(axis=0)],
            'local_max_z': float(crop_z.max()),
            'local_median_scale': float(np.median(crop_scale)),
        },
    }
