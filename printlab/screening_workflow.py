"""Workflow service for reusable statistical screening plans."""

import hashlib
import json

import cv2
import numpy as np

from . import sheet_screening as screening
from .operations import as_rgb
from .pixels import json_safe
from .screening import _put_text
from .storage import identifier, safe_label, utc_now

PLAN_STATUSES = ('draft', 'validated', 'confirmed', 'superseded')
RUN_STATUSES = ('validation', 'inspection')


def _digest(value):
    serialized = json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:16]


def _require_job_asset(store, job_id, asset_id):
    asset = store.get_asset(asset_id)
    if asset['job_id'] != job_id:
        raise ValueError('所选图像不属于当前作业')
    return asset


def _load_rgb(store, asset_id):
    return as_rgb(store.load_array(asset_id))


def _normalize_groups(payload):
    groups = payload.get('groups')
    if not isinstance(groups, dict):
        raise ValueError('需要分组对象，包含 normal、defect、candidate 图像ID列表')
    result = {}
    for role in screening.SHEET_ROLES:
        items = groups.get(role, [])
        if not isinstance(items, list):
            raise ValueError(f'{role} 分组必须是图像ID列表')
        result[role] = [str(item) for item in items]
    overlap = set(result['normal']) & set(result['defect'])
    if overlap:
        raise ValueError('同一张印张不能同时属于正常组和缺陷组')
    if not result['normal']:
        raise ValueError('建立筛查方案至少需要一组正常印张')
    return result


def _normalize_annotations(payload, groups):
    annotations = payload.get('annotations', [])
    if not isinstance(annotations, list):
        raise ValueError('标注必须是列表')
    known_defect_allowed = set(groups['defect']) | set(groups['candidate'])
    all_sheets = set(sum(groups.values(), []))
    normalized = []
    for index, item in enumerate(annotations):
        if not isinstance(item, dict) or not item.get('asset_id'):
            raise ValueError(f'第{index + 1}条标注缺少印张ID')
        role = item.get('role')
        if role not in screening.REGION_ROLES:
            raise ValueError('标注用途须为 known_defect 或 exclude')
        if item['asset_id'] not in all_sheets:
            raise ValueError('标注所挂印张不在当前分组中')
        if role == 'known_defect' and item['asset_id'] not in known_defect_allowed:
            raise ValueError('已知缺陷标注只能挂在缺陷印张或候选印张上')
        geometry = item.get('geometry')
        if not isinstance(geometry, dict):
            raise ValueError('标注缺少几何区域')
        record = {
            'id': str(item.get('id') or identifier('ann')),
            'asset_id': str(item['asset_id']),
            'role': role,
            'label': safe_label(item.get('label', '标注'), 120),
            'geometry': geometry,
        }
        if role == 'known_defect':
            kind = item.get('kind', 'other')
            if kind not in screening.DEFECT_KINDS:
                raise ValueError('缺陷类型须为 missing_print、spot、broken_stroke 或 other')
            record['kind'] = kind
        normalized.append(record)
    return normalized


def _annotations_for_sheet(annotations, asset_id):
    return [item for item in annotations if item['asset_id'] == asset_id]


def create_plan(store, job_id, payload):
    groups = _normalize_groups(payload)
    annotations = _normalize_annotations(payload, groups)
    # Validate asset existence and identical dimensions.
    shapes = set()
    for asset_id in sum(groups.values(), []):
        asset = _require_job_asset(store, job_id, asset_id)
        shapes.add((asset['width'], asset['height']))
    if len(shapes) > 1:
        raise ValueError('同版同尺寸筛查要求所有印张像素尺寸一致；请先完成几何校正')
    plan = {
        'id': identifier('plan'),
        'name': safe_label(payload.get('name', '可复用筛查方案')),
        'status': 'draft',
        'groups': groups,
        'annotations': annotations,
        'envelope_settings': {
            'noise_floor': float(payload.get('noise_floor', 0.02)),
            'min_support': int(payload.get('min_support', 3)),
        },
        'policies': list(screening.POLICIES.keys()),
        'validation': None,
        'confirmed_policy': None,
        'baseline': None,
        'frozen_digest': None,
        'created_at': utc_now(),
        'updated_at': utc_now(),
        'history': [{'at': utc_now(), 'event': 'created', 'detail': '建立草稿分组与标注'}],
    }
    plan['frozen_digest'] = _plan_digest(plan)
    return store.save_screening_plan(job_id, plan)


def _plan_digest(plan):
    material = {
        'groups': plan['groups'],
        'annotations': plan['annotations'],
        'envelope_settings': plan['envelope_settings'],
        'policies': plan['policies'],
    }
    return _digest(material)


def get_plan(store, plan_id):
    return store.get_screening_plan(plan_id)


def update_plan(store, plan_id, payload):
    plan = store.get_screening_plan(plan_id)['payload']
    if plan['status'] == 'confirmed':
        raise ValueError('已确认方案已冻结；补充样本请先复制为新版本，旧报告不会被改动')
    groups = _normalize_groups(payload)
    annotations = _normalize_annotations(payload, groups)
    for asset_id in sum(groups.values(), []):
        _require_job_asset(store, plan_job_id(store, plan_id), asset_id)
    plan['groups'] = groups
    plan['annotations'] = annotations
    if 'noise_floor' in payload:
        plan['envelope_settings']['noise_floor'] = float(payload['noise_floor'])
    if 'min_support' in payload:
        plan['envelope_settings']['min_support'] = int(payload['min_support'])
    plan['status'] = 'draft'
    plan['validation'] = None
    plan['baseline'] = None
    plan['frozen_digest'] = _plan_digest(plan)
    plan['updated_at'] = utc_now()
    plan['history'].append({'at': utc_now(), 'event': 'updated', 'detail': '修改分组或标注，需重新验证'})
    return store.save_screening_plan(plan_job_id(store, plan_id), plan)


def plan_job_id(store, plan_id):
    return store.get_screening_plan(plan_id)['job_id']


def clone_plan(store, plan_id, payload=None):
    payload = payload or {}
    source_row = store.get_screening_plan(plan_id)
    source = source_row['payload']
    job_id = source_row['job_id']
    groups = {role: list(source['groups'][role]) for role in screening.SHEET_ROLES}
    for role, additions in (payload.get('add_groups') or {}).items():
        if role not in groups:
            continue
        for asset_id in additions:
            if asset_id not in groups[role]:
                groups[role].append(str(asset_id))
    annotations = [dict(item) for item in source['annotations']]
    for item in payload.get('add_annotations', []):
        item = dict(item)
        item['id'] = identifier('ann')
        annotations.append(item)
    draft = {
        'id': identifier('plan'),
        'name': safe_label(payload.get('name', source['name'] + ' 新版本')),
        'status': 'draft',
        'groups': groups,
        'annotations': _normalize_annotations({'annotations': annotations}, groups),
        'envelope_settings': dict(source['envelope_settings']),
        'policies': list(source['policies']),
        'validation': None,
        'confirmed_policy': None,
        'baseline': None,
        'frozen_digest': None,
        'created_at': utc_now(),
        'updated_at': utc_now(),
        'history': [{'at': utc_now(), 'event': 'cloned', 'detail': f'从已冻结方案 {plan_id} 复制，补充样本不影响旧报告'}],
    }
    # Mark source superseded once a successor exists.
    if source['status'] == 'confirmed':
        source['status'] = 'superseded'
        source['history'].append({'at': utc_now(), 'event': 'superseded', 'detail': f'由新方案 {draft["id"]} 接续'})
        store.save_screening_plan(job_id, source)
    draft['frozen_digest'] = _plan_digest(draft)
    return store.save_screening_plan(job_id, draft)


def _build_full_envelope(store, plan):
    normal_arrays = [_load_rgb(store, asset_id) for asset_id in plan['groups']['normal']]
    return screening.build_envelope(normal_arrays, **plan['envelope_settings'])


def validate(store, plan_id):
    row = store.get_screening_plan(plan_id)
    plan = row['payload']
    job_id = row['job_id']
    normal_arrays = [_load_rgb(store, asset_id) for asset_id in plan['groups']['normal']]
    defect_arrays = [_load_rgb(store, asset_id) for asset_id in plan['groups']['defect']]
    annotations_by_sheet = {}
    for index, asset_id in enumerate(plan['groups']['defect']):
        annotations_by_sheet[index] = _annotations_for_sheet(plan['annotations'], asset_id)
    normal_annotations = {}
    for index, asset_id in enumerate(plan['groups']['normal']):
        normal_annotations[index] = _annotations_for_sheet(plan['annotations'], asset_id)
    envelope_kwargs = dict(plan['envelope_settings'])
    result = screening.validate_plan(normal_arrays, defect_arrays, envelope_kwargs,
                                     annotations_by_sheet, policies=plan['policies'],
                                     min_support=plan['envelope_settings']['min_support'],
                                     normal_annotations=normal_annotations)
    envelope = _build_full_envelope(store, plan)
    plan['validation'] = json_safe(result)
    plan['baseline'] = result['baseline']
    plan['status'] = 'validated'
    plan['frozen_digest'] = _plan_digest(plan)
    plan['updated_at'] = utc_now()
    plan['history'].append({'at': utc_now(), 'event': 'validated',
                            'detail': f"整印张留一验证：正常{result['normal_count']}张、缺陷{result['defect_count']}张"})
    saved = store.save_screening_plan(job_id, plan, envelope=envelope)
    return {'plan': saved, 'validation': result}


def confirm_plan(store, plan_id, policy_name):
    if policy_name not in screening.POLICIES:
        raise ValueError('确认取向须为 low_false_alarm 或 high_sensitivity')
    row = store.get_screening_plan(plan_id)
    plan = row['payload']
    if plan['status'] not in ('validated', 'confirmed'):
        raise ValueError('方案必须先通过整印张验证才能确认')
    record = plan['validation']['policies'][policy_name]
    if not record['sample_sufficiency']['sufficient']:
        raise ValueError('正常样本数量不足以支持所选误报要求：' + record['sample_sufficiency']['note'])
    plan['confirmed_policy'] = policy_name
    plan['status'] = 'confirmed'
    plan['frozen_digest'] = _plan_digest(plan)
    plan['updated_at'] = utc_now()
    plan['history'].append({'at': utc_now(), 'event': 'confirmed',
                            'detail': f"确认取向 {policy_name}；误报95%上限 {record['false_alarm_rate_upper95']:.3f}"})
    return store.save_screening_plan(row['job_id'], plan)


def _load_or_build_envelope(store, row):
    plan = row['payload']
    try:
        envelope = store.load_screening_envelope(row['id'])
    except ValueError:
        # Draft plans have no persisted envelope yet: build on demand for a
        # trial run, but never persist it so validation remains the only path
        # that freezes an envelope into the confirmed plan.
        if not plan['groups']['normal']:
            raise ValueError('方案还没有正常印张，无法建立波动包络')
        envelope = _build_full_envelope(store, plan)
    envelope['summary'] = plan['validation']['envelope_summary'] if plan.get('validation') else {
        'min_support': plan['envelope_settings']['min_support'],
        'noise_floor': plan['envelope_settings']['noise_floor']}
    return envelope


def inspect_asset(store, plan_id, asset_id, policy_name=None):
    row = store.get_screening_plan(plan_id)
    plan = row['payload']
    job_id = row['job_id']
    asset = _require_job_asset(store, job_id, asset_id)
    chosen_policy = policy_name or plan.get('confirmed_policy')
    if not chosen_policy:
        raise ValueError('请选择判定取向；或先确认一套用于后续印张的方案')
    envelope = _load_or_build_envelope(store, row)
    candidate = _load_rgb(store, asset_id)
    notes = _annotations_for_sheet(plan['annotations'], asset_id)
    result = screening.inspect_sheet(candidate, envelope, policy=chosen_policy,
                                     annotations=notes, baseline=plan.get('baseline'))
    # Recompute digest to prove the confirmed plan material did not change.
    current_digest = _plan_digest(plan)
    integrity_ok = current_digest == plan.get('frozen_digest')
    run = {
        'id': identifier('run'),
        'plan_id': plan_id,
        'plan_digest_at_run': plan.get('frozen_digest'),
        'plan_digest_current': current_digest,
        'plan_material_unchanged': integrity_ok,
        'asset_id': asset_id,
        'asset_name': asset['name'],
        'policy_name': chosen_policy,
        'policy_settings': result['summary']['settings'],
        'verdict': result['summary']['verdict'],
        'summary': result['summary'],
        'regions': result['tables']['缺陷组件'],
        'created_at': utc_now(),
        'frozen': plan['status'] == 'confirmed',
    }
    if not integrity_ok:
        run['summary']['drift_reasons'] = run['summary'].get('drift_reasons', []) + ['方案分组/标注在记录后发生变化，本结果标记为待复检']
    saved_run = store.save_screening_run(plan_id, job_id, asset_id, chosen_policy, 'inspection', run)
    # Persist the evidence images as assets so exports share one confirmed scheme.
    image_assets = []
    for label, array in result['images'].items():
        image_assets.append(store.add_asset(job_id, f"{asset['name']} · {label}", 'rgb', array,
                                            {'screening_run_id': run['id'], 'plan_id': plan_id,
                                             'evidence_key': label, 'synthetic': False}))
    return {'run': saved_run, 'images': [{'id': item['id'], 'name': item['name'], 'preview_url': item['preview_url']} for item in image_assets]}


def _evidence_bundle(store, plan_id, asset_id, region_rank, padding=12, policy_name=None):
    row = store.get_screening_plan(plan_id)
    plan = row['payload']
    envelope = _load_or_build_envelope(store, row)
    candidate = _load_rgb(store, asset_id)
    notes = _annotations_for_sheet(plan['annotations'], asset_id)
    chosen_policy = policy_name or plan.get('confirmed_policy') or 'low_false_alarm'
    result = screening.inspect_sheet(candidate, envelope, policy=chosen_policy,
                                     annotations=notes, baseline=plan.get('baseline'))
    rows = result['tables']['缺陷组件']
    if region_rank < 1 or region_rank > len(rows):
        raise KeyError('组件编号不存在')
    component = rows[region_rank - 1]
    local = screening.local_evidence(candidate, envelope, component, padding=padding)
    return plan, component, local


def evidence(store, plan_id, asset_id, region_rank, padding=12, policy_name=None):
    plan, component, local = _evidence_bundle(store, plan_id, asset_id, region_rank, padding, policy_name)
    return {'component': component,
            'policy_name': policy_name or plan.get('confirmed_policy') or 'low_false_alarm',
            'evidence': {'box': local['box'], 'stats': json_safe(local['stats']),
                         'image_keys': list(local['images'].keys())},
            'normal_envelope_summary': plan.get('validation', {}).get('envelope_summary')}


def evidence_image(store, plan_id, asset_id, region_rank, key, padding=12, policy_name=None):
    _, _, local = _evidence_bundle(store, plan_id, asset_id, region_rank, padding, policy_name)
    if key not in local['images']:
        raise KeyError('证据图不存在；可选：' + '、'.join(local['images'].keys()))
    return local['images'][key]


# ---------------------------------------------------------------------------
# 合成演示印张组：确定性生成，用于完整体验分组/标注/验证流程
# ---------------------------------------------------------------------------

def _base_sheet(width, height, seed):
    """Deterministic sheet with solid blocks, fine type and halftone texture."""
    rng = np.random.default_rng(seed)
    image = np.ones((height, width, 3), dtype=np.float32) * np.float32([0.97, 0.965, 0.94])
    # Solid color blocks.
    cv2.rectangle(image, (30, 30), (220, 170), (0.12, 0.28, 0.55), -1)
    cv2.rectangle(image, (250, 40), (420, 150), (0.62, 0.16, 0.2), -1)
    cv2.rectangle(image, (40, 200), (260, 300), (0.16, 0.45, 0.28), -1)
    # Fine dark type and thin rules (the area where broken strokes matter).
    font = cv2.FONT_HERSHEY_SIMPLEX
    for line, text_value in enumerate(('PRINTLAB FINE TYPE 0123456789', 'ABCDEFGHIJKLMN small glyphs', 'thin stroke registration line')):
        _put_text(image, text_value, (290, 210 + line * 26), font, 0.42, (0.06, 0.06, 0.08), 1, cv2.LINE_AA)
    for index in range(10):
        y = 300 + index * 6
        cv2.line(image, (290, y), (width - 40, y), (0.08, 0.08, 0.1), 1)
    # Halftone-style dot field with a stable lattice and slight phase jitter.
    yy, xx = np.mgrid[0:height, 0:width]
    pitch = 9
    jitter_x = rng.uniform(-1.2, 1.2, (height // pitch + 2, width // pitch + 2)).astype(np.float32)
    jitter_y = rng.uniform(-1.2, 1.2, (height // pitch + 2, width // pitch + 2)).astype(np.float32)
    gx = (xx / pitch).astype(np.int32)
    gy = (yy / pitch).astype(np.int32)
    jx = jitter_x[np.clip(gy, 0, jitter_x.shape[0] - 1), np.clip(gx, 0, jitter_x.shape[1] - 1)]
    jy = jitter_y[np.clip(gy, 0, jitter_y.shape[0] - 1), np.clip(gx, 0, jitter_y.shape[1] - 1)]
    phase_x = xx - gx * pitch + jx - pitch / 2
    phase_y = yy - gy * pitch + jy - pitch / 2
    dot_field = (np.hypot(phase_x, phase_y) < 2.4).astype(np.float32)
    region = np.zeros((height, width), dtype=bool)
    region[320:height - 30, 40:width - 40] = True
    image[region] = image[region] * (1 - 0.55 * dot_field[region, None])
    # Variable content strip: deliberately different per sheet (barcode-like).
    variable = np.zeros((height, width), dtype=bool)
    variable[height - 46:height - 24, 30:220] = True
    return image, variable, rng


def _print_variation(image, rng):
    """Normal press variation: mild tone drift, registration jitter, scan noise."""
    dx = rng.uniform(-0.6, 0.6)
    dy = rng.uniform(-0.6, 0.6)
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderValue=(0.97, 0.965, 0.94))
    gain = np.float32([rng.uniform(0.97, 1.02), rng.uniform(0.97, 1.02), rng.uniform(0.95, 1.01)])
    shifted = np.clip(shifted * gain, 0, 1)
    shifted += rng.normal(0, 0.004, shifted.shape).astype(np.float32)
    return np.clip(shifted, 0, 1)


def _variable_content(image, variable, rng):
    """Fill the variable strip with random bars so it must be excluded."""
    output = image.copy()
    x0, x1 = 30, 220
    y0, y1 = image.shape[0] - 46, image.shape[0] - 24
    cursor = x0
    while cursor < x1:
        bar_width = int(rng.integers(1, 5))
        if rng.random() > 0.45:
            output[y0:y1, cursor:min(cursor + bar_width, x1)] = (0.05, 0.05, 0.08)
        cursor += bar_width + int(rng.integers(1, 4))
    return output, variable


def create_demo_set(store, job_id, payload=None):
    payload = payload or {}
    width = int(payload.get('width', 480))
    height = int(payload.get('height', 360))
    normal_count = int(payload.get('normal_count', 7))
    defect_count = int(payload.get('defect_count', 3))
    if normal_count < 4 or normal_count > 40 or defect_count < 1 or defect_count > 8:
        raise ValueError('演示组建议正常印张4-40张、缺陷印张1-8张')
    base, variable, base_rng = _base_sheet(width, height, seed=1234)
    groups = {'normal': [], 'defect': [], 'candidate': []}
    annotations = []
    asset_meta = {'synthetic': True, 'screening_demo': True}
    # Normal sheets: independent variation realizations.
    for index in range(normal_count):
        rng = np.random.default_rng(2000 + index)
        sheet, _ = _variable_content(_print_variation(base, rng), variable, rng)
        asset = store.add_asset(job_id, f'正常印张 N{index + 1:02d}', 'rgb', sheet, dict(asset_meta, sheet_role='normal'))
        groups['normal'].append(asset['id'])
    # Defect sheets with documented ground truth.
    defect_specs = [
        ('missing_print', (120, 95), 9, '漏印：蓝色块内浅色圆斑'),
        ('spot', (330, 95), 7, '污点：红色块内深色点'),
        ('broken_stroke', (360, 232), None, '断笔：小字笔画缺口'),
    ]
    for index in range(defect_count):
        rng = np.random.default_rng(4000 + index)
        sheet, _ = _variable_content(_print_variation(base, rng), variable, rng)
        kind, (cx, cy), radius, label_text = defect_specs[index % len(defect_specs)]
        if kind == 'missing_print':
            cv2.circle(sheet, (cx, cy), radius, (0.92, 0.92, 0.88), -1)
            geometry = {'type': 'rectangle', 'x': cx - radius, 'y': cy - radius, 'width': 2 * radius, 'height': 2 * radius}
        elif kind == 'spot':
            cv2.circle(sheet, (cx, cy), radius, (0.03, 0.03, 0.04), -1)
            geometry = {'type': 'rectangle', 'x': cx - radius, 'y': cy - radius, 'width': 2 * radius, 'height': 2 * radius}
        else:
            # Erase a short run of glyph strokes on the second fine-type line.
            cv2.rectangle(sheet, (cx - 14, cy - 7), (cx + 4, cy + 5), (0.97, 0.965, 0.94), -1)
            geometry = {'type': 'rectangle', 'x': cx - 14, 'y': cy - 7, 'width': 18, 'height': 12}
        asset = store.add_asset(job_id, f'缺陷印张 D{index + 1:02d}', 'rgb', sheet, dict(asset_meta, sheet_role='defect', defect_kind=kind))
        groups['defect'].append(asset['id'])
        annotations.append({'id': identifier('ann'), 'asset_id': asset['id'], 'role': 'known_defect',
                            'label': label_text, 'kind': kind, 'geometry': geometry})
    # A candidate sheet (normal realization, no ground truth) for later runs.
    rng = np.random.default_rng(9000)
    candidate, _ = _variable_content(_print_variation(base, rng), variable, rng)
    candidate_asset = store.add_asset(job_id, '待检印张 C01', 'rgb', candidate, dict(asset_meta, sheet_role='candidate'))
    groups['candidate'].append(candidate_asset['id'])
    # Variable-content exclusion applies to every sheet; attach to first normal as plan template.
    exclude_geometry = {'type': 'rectangle', 'x': 30, 'y': height - 46, 'width': 190, 'height': 22}
    annotations.append({'id': identifier('ann'), 'asset_id': groups['normal'][0],
                        'role': 'exclude', 'label': '可变条码区（不检查）', 'geometry': exclude_geometry})
    plan_payload = {
        'name': '演示产品筛查方案',
        'groups': groups,
        'annotations': annotations,
        'noise_floor': 0.02,
        'min_support': 3,
    }
    plan = create_plan(store, job_id, plan_payload)
    # Exclude regions are plan-level geometry; replicate onto every sheet id for evaluation.
    stored = plan['payload']
    exclude_note = next(item for item in stored['annotations'] if item['role'] == 'exclude')
    for asset_id in groups['normal'][1:] + groups['defect'] + groups['candidate']:
        replica = dict(exclude_note)
        replica['id'] = identifier('ann')
        replica['asset_id'] = asset_id
        stored['annotations'].append(replica)
    stored['frozen_digest'] = _plan_digest(stored)
    plan = store.save_screening_plan(job_id, stored)
    return {'plan': plan, 'groups': groups, 'annotation_count': len(stored['annotations']),
            'note': '合成演示印张组：正常波动含亚像素相位漂移与扫描噪声，缺陷含漏印/污点/断笔，可变条码区需排除'}
