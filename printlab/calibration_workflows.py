"""实测校准的会话工作流与可执行操作。

会话（session）保存原始参考、色块几何、纳入/排除判断与每次读数；证据（evidence）
保存两套候选模型的整组留出比较。确认时把完整模型与决定快照写进不可变记录，
之后修改参考只会产生修订版，旧结果不变。
"""
import copy

import numpy as np

from . import calibration as calib
from .pixels import heatmap, luminance, number, rgb_image
from .storage import safe_label, utc_now


SESSION_VERSION = 'printlab-calibration-session-v1'


def _asset_scan_ids(session):
    return [item['scan_id'] for item in session['scans']]


def create_session(store, job_id, payload):
    name = safe_label(payload.get('name', '实测扫描校准'))
    scan_asset_ids = payload.get('scan_asset_ids') or []
    if not isinstance(scan_asset_ids, list) or not scan_asset_ids:
        raise ValueError('请至少选择一张同设置扫描色卡图像')
    if len(scan_asset_ids) > 24:
        raise ValueError('一次校准最多选择24张扫描')
    scans = []
    seen = set()
    for index, asset_id in enumerate(scan_asset_ids):
        asset = store.get_asset(asset_id)
        if asset['job_id'] != job_id:
            raise ValueError('扫描图像必须来自当前作业')
        if asset['channels'] not in (1, 3):
            raise ValueError(f'{asset["name"]} 不是RGB/灰度图像，不能取色块读数')
        if asset_id in seen:
            raise ValueError('扫描图像不能重复选择')
        seen.add(asset_id)
        scans.append({'scan_id': asset_id, 'name': asset['name'], 'width': asset['width'],
                      'height': asset['height'], 'cards': []})
    setup = payload.get('setup') or {}
    if not isinstance(setup, dict):
        raise ValueError('拍摄设置必须为对象')
    session = {
        'version': SESSION_VERSION,
        'workspace': calib.WORKSPACE,
        'setup': {
            'exposure': str(setup.get('exposure', ''))[:120],
            'white_balance': str(setup.get('white_balance', ''))[:120],
            'lighting': str(setup.get('lighting', ''))[:120],
            'note': str(setup.get('note', ''))[:1000],
        },
        'inset': float(number(setup.get('inset', 0.15), '色块内缩比例', 0, 0.45)),
        'scans': scans,
        'coverage_gaps': [],
        'created_at': utc_now(),
        'updated_at': utc_now(),
    }
    record = store.save_calibration(job_id, name, session, {'status': 'no_readings'})
    return store.get_calibration(record['id'])


def _find_scan(session, scan_id):
    for scan in session['scans']:
        if scan['scan_id'] == scan_id:
            return scan
    raise KeyError('会话中没有该扫描')


def _find_card(scan, card_id):
    for card in scan['cards']:
        if card['card_id'] == card_id:
            return card
    raise KeyError('该扫描下没有这块色卡')


def add_card(store, calibration_id, payload):
    record = store.get_calibration(calibration_id)
    if record['status'] != 'draft':
        raise ValueError('已确认校准不能编辑；请创建修订版')
    session = copy.deepcopy(record['session'])
    scan = _find_scan(session, payload.get('scan_id'))
    card_id = safe_label(payload.get('card_id') or f'card_{len(scan["cards"]) + 1}', 60)
    if any(card['card_id'] == card_id for card in scan['cards']):
        raise ValueError('该扫描下色卡编号已存在：' + card_id)
    scan['cards'].append({'card_id': card_id, 'name': str(payload.get('name') or card_id)[:120], 'patches': []})
    session['updated_at'] = utc_now()
    return store.update_calibration_session(calibration_id, session, record['evidence'])


def _resample_patch(store, scan, patch, inset):
    image = rgb_image(store.load_array(scan['scan_id']))
    sampled = calib.sample_patch(image, patch['geometry'], patch['reference_rgb'],
                                 inset=inset, manual_quality=patch.get('quality'))
    merged = {k: v for k, v in sampled.items()}
    merged['name'] = patch['name']
    merged['card_id'] = patch.get('card_id')
    return merged


def add_patch(store, calibration_id, payload):
    record = store.get_calibration(calibration_id)
    if record['status'] != 'draft':
        raise ValueError('已确认校准不能编辑；请创建修订版')
    session = copy.deepcopy(record['session'])
    scan = _find_scan(session, payload.get('scan_id'))
    card_id = payload.get('card_id')
    card = _find_card(scan, card_id) if card_id else (scan['cards'][-1] if scan['cards'] else None)
    if card is None:
        raise ValueError('请先在该扫描下添加色卡')
    name = safe_label(payload.get('name', f'P{len(card["patches"]) + 1}'), 80)
    if any(p['name'] == name for p in card['patches']):
        raise ValueError('同一色卡上色块名称不能重复：' + name)
    geometry = payload.get('geometry') or {}
    if geometry.get('type') != 'rectangle':
        raise ValueError('目前支持矩形色块，请提供 x,y,width,height')
    reference = calib._validate_rgb(payload.get('reference_rgb'), '参考颜色')
    image = rgb_image(store.load_array(scan['scan_id']))
    sampled = calib.sample_patch(image, geometry, reference, inset=session['inset'],
                                 manual_quality=payload.get('quality'))
    sampled['name'] = name
    sampled['card_id'] = card['card_id']
    card['patches'].append(sampled)
    session['updated_at'] = utc_now()
    return store.update_calibration_session(calibration_id, session, record['evidence'])


def import_patches_csv(store, calibration_id, payload):
    record = store.get_calibration(calibration_id)
    if record['status'] != 'draft':
        raise ValueError('已确认校准不能编辑；请创建修订版')
    session = copy.deepcopy(record['session'])
    scan = _find_scan(session, payload.get('scan_id'))
    rows = calib.parse_patch_csv(payload.get('csv_text', ''))
    image = rgb_image(store.load_array(scan['scan_id']))
    added = 0
    for row in rows:
        card = next((c for c in scan['cards'] if c['card_id'] == row['card_id']), None)
        if card is None:
            card = {'card_id': row['card_id'], 'name': row['card_id'], 'patches': []}
            scan['cards'].append(card)
        if any(p['name'] == row['name'] for p in card['patches']):
            raise ValueError('同一色卡上色块名称重复：' + row['name'])
        sampled = calib.sample_patch(image, row['geometry'], row['reference_rgb'],
                                     inset=session['inset'], manual_quality=row['quality'])
        sampled['name'] = row['name']
        sampled['card_id'] = card['card_id']
        card['patches'].append(sampled)
        added += 1
    session['updated_at'] = utc_now()
    updated = store.update_calibration_session(calibration_id, session, record['evidence'])
    return {'calibration': updated, 'added': added}


def update_patch(store, calibration_id, payload):
    """修改参考颜色、质量/纳入标记或几何；重采样读数，保留判断历史。"""
    record = store.get_calibration(calibration_id)
    if record['status'] != 'draft':
        raise ValueError('已确认校准不能编辑；请创建修订版')
    session = copy.deepcopy(record['session'])
    scan = _find_scan(session, payload.get('scan_id'))
    card = _find_card(scan, payload.get('card_id'))
    patch = next((p for p in card['patches'] if p['name'] == payload.get('name')), None)
    if patch is None:
        raise KeyError('色块不存在')
    history = patch.get('history', [])
    history.append({'at': utc_now(), 'reference_rgb': patch['reference_rgb'],
                    'quality': patch['quality'], 'geometry': patch['geometry']})
    if payload.get('reference_rgb') is not None:
        patch['reference_rgb'] = calib._validate_rgb(payload['reference_rgb'], '参考颜色')
    if payload.get('quality') is not None:
        if payload['quality'] not in ('ok', 'saturated', 'dirty', 'excluded'):
            raise ValueError('质量标记只支持 ok、saturated、dirty、excluded')
        patch['quality'] = payload['quality']
        patch['included'] = payload['quality'] != 'excluded'
    if payload.get('geometry') is not None:
        if payload['geometry'].get('type') != 'rectangle':
            raise ValueError('目前支持矩形色块')
        patch['geometry'] = payload['geometry']
    image = rgb_image(store.load_array(scan['scan_id']))
    resampled = calib.sample_patch(image, patch['geometry'], patch['reference_rgb'],
                                   inset=session['inset'], manual_quality=patch['quality'])
    preserved_name, preserved_card = patch['name'], patch['card_id']
    patch.clear()
    patch.update(resampled)
    patch['name'] = preserved_name
    patch['card_id'] = preserved_card
    patch['history'] = history
    session['updated_at'] = utc_now()
    return store.update_calibration_session(calibration_id, session, record['evidence'])


def remove_patch(store, calibration_id, payload):
    record = store.get_calibration(calibration_id)
    if record['status'] != 'draft':
        raise ValueError('已确认校准不能编辑；请创建修订版')
    session = copy.deepcopy(record['session'])
    scan = _find_scan(session, payload.get('scan_id'))
    card = _find_card(scan, payload.get('card_id'))
    before = len(card['patches'])
    card['patches'] = [p for p in card['patches'] if p['name'] != payload.get('name')]
    if len(card['patches']) == before:
        raise KeyError('色块不存在')
    session['updated_at'] = utc_now()
    return store.update_calibration_session(calibration_id, session, record['evidence'])


def build_evidence(store, calibration_id, payload=None):
    """对当前会话采样结果构建两套候选模型与整组留出证据。

    数据不足以支撑灵活（含照明场）模型时，仍返回简单模型证据，并明确标注
    灵活方案因数据不足未参与比较，而不是让整个流程失败。
    """
    payload = payload or {}
    record = store.get_calibration(calibration_id)
    if record['status'] != 'draft':
        raise ValueError('已确认校准不能重新拟合；请创建修订版')
    session = copy.deepcopy(record['session'])
    readings = calib.collect_readings(session['scans'])
    groups = {r['group_id'] for r in readings}
    if len(groups) < calib.MIN_GROUPS_FOR_EVIDENCE:
        raise ValueError(
            f'独立验证至少需要{calib.MIN_GROUPS_FOR_EVIDENCE}个不同的（扫描,色卡）组，当前只有{len(groups)}组；'
            '请增加另一张扫描或另一块色卡，否则无法给出未参与拟合的独立误差')
    degree = int(payload.get('degree', 2))
    field_degree = int(payload.get('field_degree', 1))
    spatial_warning = None
    simple_model = calib.build_simple(readings, degree=degree)
    try:
        spatial_model = calib.build_spatial(readings, degree=degree, field_degree=field_degree)
    except ValueError as exc:
        spatial_model = None
        spatial_warning = str(exc)
    simple_builder = lambda train: calib.build_simple(train, degree=degree)

    def spatial_builder(train):
        # 留出后训练子集可能不再满足空间模型的最小样本要求；明确报错给交叉验证层，
        # 由上层决定该候选是否可比较，而不是产生静默的退化结果。
        return calib.build_spatial(train, degree=degree, field_degree=field_degree)

    evidence = {'simple': {'model': simple_model, 'training_fit': simple_model.pop('training_fit')}}
    if spatial_model is not None:
        evidence['spatial'] = {'model': spatial_model, 'training_fit': spatial_model.pop('training_fit')}
    else:
        evidence['spatial'] = None
    scan_groups = {r['scan_id'] for r in readings}
    card_groups = {r['group_id'] for r in readings}
    for label, builder in (('simple', simple_builder), ('spatial', spatial_builder)):
        if evidence[label] is None:
            continue
        entry = evidence[label]
        try:
            entry['scan_holdout'] = calib.cross_validation(readings, builder, 'scan') if len(scan_groups) >= calib.MIN_GROUPS_FOR_EVIDENCE else None
            entry['card_holdout'] = calib.cross_validation(readings, builder, 'card') if len(card_groups) >= calib.MIN_GROUPS_FOR_EVIDENCE else None
        except ValueError as exc:
            # 灵活模型在留出子训练集上欠定：标记不可比较，回退到只比较简单方案。
            if label == 'spatial':
                evidence['spatial'] = None
                spatial_warning = (spatial_warning + '；' if spatial_warning else '') + '整组留出子训练集不足：' + str(exc)
                continue
            raise
    if evidence['spatial'] is None:
        evidence['recommendation'] = {
            'recommended': 'simple',
            'reasons': ['数据不足以拟合含照明场的灵活方案：' + spatial_warning, '只比较较简单方案，避免在欠定数据上外推空间场'],
            'rule': '训练误差低不构成选择灵活方案的理由；灵活方案需要足够的多位置读数与独立留出证据',
        }
    else:
        evidence['recommendation'] = calib.choose_recommendation(evidence, len(scan_groups), len(card_groups))
    evidence['spatial_unavailable'] = spatial_warning
    evidence['coverage_gaps'] = calib.coverage_gaps(readings)
    evidence['readings_table'] = [
        {'scan_id': r['scan_id'], 'card_id': r['card_id'], 'patch_name': r['patch_name'],
         'group_id': r['group_id'], 'quality': r['quality'],
         'measured_linear': r['measured'].tolist(), 'reference_linear': r['reference'].tolist(),
         'nx': r['nx'], 'ny': r['ny']}
        for r in readings
    ]
    evidence['built_at'] = utc_now()
    session['coverage_gaps'] = evidence['coverage_gaps']
    session['updated_at'] = utc_now()
    return store.update_calibration_session(calibration_id, session, evidence)


def confirm(store, calibration_id, payload):
    """用户查看证据后确认使用哪套模型；快照写入不可变记录。"""
    record = store.get_calibration(calibration_id)
    if record['status'] != 'draft':
        raise ValueError('该校准已经确认')
    evidence = record['evidence']
    if evidence.get('status') == 'no_readings' or 'simple' not in evidence:
        raise ValueError('请先构建模型比较证据，再确认校准')
    chosen = payload.get('model_kind')
    if chosen not in ('simple', 'spatial'):
        raise ValueError('请选择 simple（较简单）或 spatial（含照明场）模型')
    if chosen == 'spatial' and evidence.get('spatial') is None:
        raise ValueError('当前数据不足以支撑灵活（含照明场）模型，请选择较简单方案或补充多位置色块')
    model = copy.deepcopy(evidence[chosen]['model'])
    # 把训练范围等应用所需信息固化进模型快照。
    model['calibration_id'] = calibration_id
    model['confirmed_at'] = utc_now()
    model['session_version'] = record['session']['version']
    decision = {
        'chosen': chosen,
        'recommendation': evidence.get('recommendation'),
        'user_note': str(payload.get('note', ''))[:1000],
        'evidence_built_at': evidence.get('built_at'),
        'declaration': '选择基于未参与拟合的整次扫描/整块色卡留出证据；训练误差不作为推荐依据',
    }
    return store.confirm_calibration(calibration_id, model, decision)


def revise(store, calibration_id, payload=None):
    """基于已确认档案创建可编辑修订版；原档案和旧结果保持不变。"""
    payload = payload or {}
    previous = store.get_calibration(calibration_id)
    if previous['status'] not in ('confirmed', 'superseded'):
        raise ValueError('只有已确认的校准才能创建修订版')
    session = copy.deepcopy(previous['session'])
    session['updated_at'] = utc_now()
    name = safe_label(payload.get('name') or previous['name'] + '（修订）')
    return store.save_calibration(previous['job_id'], name, session, {'status': 'revised_pending_refit'},
                                  revision_of=calibration_id, revision=previous['revision'] + 1)


# ---------------------------------------------------------------------------
# 可执行操作：应用校准 / 校准后质量测量
# ---------------------------------------------------------------------------

def apply_scan_calibration(store, source_asset, calibration_id, strength=1.0):
    calibration = store.get_calibration(calibration_id)
    if calibration['job_id'] != source_asset['job_id']:
        raise ValueError('校准档案不属于当前作业')
    if calibration['status'] != 'confirmed' or not calibration.get('confirmed_model'):
        raise ValueError('请选择已确认的校准档案')
    image = rgb_image(store.load_array(source_asset['id']))
    model = calibration['confirmed_model']
    result_full = calib.apply_calibration(model, image, include_reliability=True)
    alpha = float(number(strength, '校准强度', 0, 1))
    if alpha < 1:
        calibrated = image * (1 - alpha) + result_full['calibrated'] * alpha
    else:
        calibrated = result_full['calibrated']
    domain = result_full['reliability']
    overlay = calibrated.copy()
    overlay[~domain['supported']] = overlay[~domain['supported']] * 0.55 + np.asarray([0.75, 0.25, 0.15]) * 0.45
    summary = {
        'calibration_id': calibration_id,
        'calibration_name': calibration['name'],
        'model_kind': model['kind'],
        'workspace': model['workspace'],
        'strength': alpha,
        'confirmed_at': model.get('confirmed_at'),
        'revision': calibration['revision'],
        'supported_fraction': domain['fractions']['supported'],
        'position_extrapolation_fraction': domain['fractions']['position_extrapolation'],
        'tone_extrapolation_fraction': domain['fractions']['tone_extrapolation'],
        'color_extrapolation_fraction': domain['fractions']['color_extrapolation'],
        'clipped_fraction': result_full['clipped_fraction'],
        'reliability_rule': '默认质量测量只把 supported=True 的像素视为可靠读数；外推像素保留在图中但单独标记',
        'scope': '仅适用于同曝光、白平衡与灯光设置的后续校样；不同设置需重新校准',
    }
    images = {
        '实测校准结果': np.clip(calibrated, 0, 1).astype(np.float32),
        '可靠范围叠图': overlay.astype(np.float32),
        '可靠像素掩码': domain['supported'].astype(np.float32),
        '位置外推掩码': domain['reasons']['position'].astype(np.float32),
        '亮度外推掩码': domain['reasons']['tone'].astype(np.float32),
        '颜色外推掩码': domain['reasons']['color'].astype(np.float32),
    }
    tables = {'可靠范围统计': [
        {'range': 'supported（有训练数据支持）', 'fraction': domain['fractions']['supported']},
        {'range': 'position_extrapolation（位置外推）', 'fraction': domain['fractions']['position_extrapolation']},
        {'range': 'tone_extrapolation（暗部/亮部外推）', 'fraction': domain['fractions']['tone_extrapolation']},
        {'range': 'color_extrapolation（颜色外推）', 'fraction': domain['fractions']['color_extrapolation']},
    ]}
    metadata = {
        'calibration_id': calibration_id,
        'model_kind': model['kind'],
        'confirmed_at': model.get('confirmed_at'),
        'revision': calibration['revision'],
        'reliability_supported_fraction': domain['fractions']['supported'],
    }
    return {'summary': summary, 'images': images, 'tables': tables,
            'metadata': {'实测校准结果': metadata, '可靠像素掩码': dict(metadata, mask_role='supported'),
                         '位置外推掩码': dict(metadata, mask_role='position'),
                         '亮度外推掩码': dict(metadata, mask_role='tone'),
                         '颜色外推掩码': dict(metadata, mask_role='color')},
            '_reliability_supported': domain['supported']}


def calibrated_quality(store, source_asset, calibration_id, white_rgb=(1, 1, 1), channel='K'):
    """校准后质量测量：可靠像素与外推像素分开报告，不把外推像素当可靠读数。"""
    calibration = store.get_calibration(calibration_id)
    if calibration['status'] != 'confirmed' or not calibration.get('confirmed_model'):
        raise ValueError('请选择已确认的校准档案')
    image = rgb_image(store.load_array(source_asset['id']))
    model = calibration['confirmed_model']
    applied = calib.apply_calibration(model, image, include_reliability=True)
    calibrated = applied['calibrated']
    supported = applied['reliability']['supported']
    density = calib.calibrated_density(calibrated, white_rgb, channel=channel)
    light = luminance(calibrated)

    def stats(mask):
        if not np.any(mask):
            return {'pixels': 0, 'mean_density': None, 'p95_density': None,
                    'mean_luminance': None, 'std_luminance': None}
        return {
            'pixels': int(mask.sum()),
            'mean_density': float(density[mask].mean()),
            'p95_density': float(np.percentile(density[mask], 95)),
            'mean_luminance': float(light[mask].mean()),
            'std_luminance': float(light[mask].std()),
        }

    rows = [
        dict({'range': 'supported_reliable', 'status': 'reliable'}, **stats(supported)),
        dict({'range': 'unsupported_extrapolation', 'status': 'not_reliable'}, **stats(~supported)),
        dict({'range': 'all_pixels', 'status': 'mixed_do_not_use_as_reliable'}, **stats(np.ones_like(supported))),
    ]
    summary = {
        'calibration_id': calibration_id,
        'model_kind': model['kind'],
        'channel': channel,
        'white_rgb': list(white_rgb),
        'supported_fraction': float(supported.mean()),
        'default_quality_basis': 'supported_reliable',
        'rule': '不受校准数据支持的像素保留但不计入可靠质量读数，整幅图不会被统一盖可靠标记',
    }
    images = {
        '校准后密度图': heatmap(density),
        '可靠测量范围': applied['reliability_overlay'].astype(np.float32),
    }
    return {'summary': summary, 'images': images, 'tables': {'校准质量读数': rows}}
