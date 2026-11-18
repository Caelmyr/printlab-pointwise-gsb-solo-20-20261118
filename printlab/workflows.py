import time
from pathlib import Path

import numpy as np

from . import press, profiles, reports, targets
from .operations import as_rgb, execute
from .pixels import CHANNELS, image_facts, json_safe, number, read_image
from .storage import safe_label


def create_job(store, payload):
    job = store.create_job(payload.get('name', '新印刷作业'), payload.get('dpi', 300), payload.get('description', ''))
    if payload.get('default_profiles', True):
        for profile in profiles.default_profiles():
            store.save_profile(job['id'], profile['kind'], profile['name'], profile['payload'])
    return store.get_job(job['id'])


def import_upload(store, job_id, filename, payload, matte=(1, 1, 1)):
    job = store.get_job(job_id)
    name = Path(str(filename)).name or '上传图像'
    name = safe_label(name)
    rgb, alpha, metadata = read_image(payload, matte=matte)
    metadata['original_filename'] = name
    metadata['import_kind'] = 'user_upload'
    metadata['synthetic'] = False
    metadata['facts_at_job_dpi'] = image_facts(rgb, job['dpi'])
    asset = store.add_asset(job_id, name, 'rgb', rgb, metadata)
    outputs = [asset]
    if np.any(alpha < 1):
        alpha_metadata = {'import_kind': 'upload_alpha', 'source_id': asset['id'], 'original_filename': name}
        outputs.append(store.add_asset(job_id, name + ' / 原始透明度', 'gray', alpha, alpha_metadata, asset['id']))
    return {'assets': outputs, 'metadata': metadata}


def target_result(kind, parameters):
    if kind == 'step_wedge':
        return press.build_step_wedge(**parameters)
    if kind == 'control_strip':
        return press.build_color_control_strip(**parameters)
    return targets.generate_target(kind, parameters)


def add_target(store, job_id, kind, parameters=None):
    job = store.get_job(job_id)
    settings = dict(parameters or {})
    started = time.perf_counter()
    result = target_result(kind, settings)
    outputs = []
    for label, array in result['images'].items():
        channels = array.shape[2] if array.ndim == 3 else 1
        image_kind = 'cmyk' if channels == 4 else 'rgb' if channels == 3 else 'gray'
        metadata = {
            'synthetic': True,
            'target_kind': kind,
            'target_parameters': settings,
            'target_summary': result.get('summary', {}),
            'target_tables': result.get('tables', {}),
            'facts_at_job_dpi': image_facts(as_rgb(array), job['dpi']),
        }
        outputs.append(store.add_asset(job_id, label, image_kind, array, metadata))
    return {'assets': outputs, 'summary': result.get('summary', {}), 'tables': result.get('tables', {}), 'elapsed_ms': (time.perf_counter() - started) * 1000}


def create_demo(store, name='PrintLab 合成演示作业'):
    job = create_job(store, {'name': name, 'dpi': 300, 'description': '合成印刷原稿、扫描校样、灰阶控制条和循环纹样。所有素材均由程序生成，用于体验，不代表真实印机观测。'})
    job_id = job['id']
    artwork = add_target(store, job_id, 'artwork', {'width': 720, 'height': 540})
    source_id = artwork['assets'][0]['id']
    add_target(store, job_id, 'proof', {'width': 720, 'height': 540})
    add_target(store, job_id, 'step_wedge', {'width': 880, 'height': 480, 'steps': 11})
    add_target(store, job_id, 'control_strip', {'width': 960, 'height': 160, 'repeats': 2})
    add_target(store, job_id, 'repeat', {'width': 320, 'height': 240})
    store.add_region(source_id, '中间测量区', {'type': 'rectangle', 'x': 160, 'y': 140, 'width': 180, 'height': 160}, 'measure')
    separated = execute(store, job_id, {'operation': 'separate', 'source_id': source_id, 'parameters': {'black_generation': 0.8, 'total_ink_limit': 3.0}})
    cmyk = next(output for output in separated['outputs'] if output['channels'] == 4)
    execute(store, job_id, {'operation': 'overprint', 'source_id': cmyk['id'], 'parameters': {}})
    store.add_note(job_id, '演示路径：选择原稿做分色 → 选择CMYK结果制作AM网版 → 叠印与频域检查；扫描校样可与原稿做平移套准及差异检查。')
    return store.get_job(job_id)


def duplicate_asset(store, asset_id, name=None):
    asset = store.get_asset(asset_id)
    array = store.load_array(asset_id)
    metadata = dict(asset['metadata'])
    metadata['copy_of'] = asset_id
    metadata['copy_reason'] = '用户独立图像副本'
    copied = store.add_asset(asset['job_id'], name or (asset['name'] + ' 副本'), asset['kind'], array, metadata, asset_id)
    for region in asset.get('regions', []):
        store.add_region(copied['id'], region['name'], region['geometry'], region['role'])
    return store.get_asset(copied['id'])


def asset_facts(store, asset_id):
    asset = store.get_asset(asset_id)
    job = store.get_job(asset['job_id'])
    array = store.load_array(asset_id)
    facts = image_facts(as_rgb(array), job['dpi'])
    facts['stored_kind'] = asset['kind']
    facts['stored_channels'] = asset['channels']
    facts['source_lineage_depth'] = len(store.lineage(asset_id)) - 1
    facts['region_count'] = len(asset['regions'])
    if asset['channels'] == 4:
        total = array.sum(axis=2)
        facts['maximum_total_ink_percent'] = float(total.max() * 100)
        facts['mean_total_ink_percent'] = float(total.mean() * 100)
        facts['mean_channel_percent'] = {channel: float(array[:, :, index].mean() * 100) for index, channel in enumerate(CHANNELS)}
    return facts


def preflight_asset(store, asset_id, parameters=None):
    asset = store.get_asset(asset_id)
    array = store.load_array(asset_id)
    job = store.get_job(asset['job_id'])
    settings = dict(parameters or {})
    settings.setdefault('dpi', job['dpi'])
    started = time.perf_counter()
    result = reports.image_preflight(as_rgb(array), array if asset['channels'] == 4 else None, **settings)
    return store.save_operation(asset['job_id'], 'preflight', asset_id, None, settings, result, (time.perf_counter() - started) * 1000)


def source_groups(job):
    assets = {asset['id']: asset for asset in job['assets']}
    groups = {}
    for asset in assets.values():
        current = asset
        visited = set()
        while current.get('parent_id') and current['parent_id'] in assets:
            if current['id'] in visited:
                raise ValueError('图像来源链循环')
            visited.add(current['id'])
            current = assets[current['parent_id']]
        root = current['id']
        if root not in groups:
            groups[root] = {'source': current, 'derived': []}
        if asset['id'] != root:
            groups[root]['derived'].append(asset)
    return list(groups.values())


def job_dashboard(store, job_id):
    job = store.get_job(job_id)
    metrics = reports.summary_metrics(job)
    groups = source_groups(job)
    recent = job['operations'][:8]
    operation_ids = {operation['id'] for operation in recent}
    latest_outputs = [asset for asset in job['assets'] if asset.get('operation_id') in operation_ids]
    sources = [group['source'] for group in groups]
    return {'job': {key: job[key] for key in ('id', 'name', 'description', 'dpi', 'revision', 'updated_at')}, 'metrics': metrics, 'sources': sources, 'recent_operations': recent, 'latest_outputs': latest_outputs, 'source_groups': [{'source_id': group['source']['id'], 'derived_count': len(group['derived'])} for group in groups]}


def validate_job_links(store, job_id):
    job = store.get_job(job_id)
    ids = {asset['id'] for asset in job['assets']}
    operation_ids = {operation['id'] for operation in job['operations']}
    problems = []
    checked_arrays = 0
    for asset in job['assets']:
        parent = asset.get('parent_id')
        if parent and parent not in ids:
            problems.append({'asset_id': asset['id'], 'problem': 'missing_parent'})
        operation_id = asset.get('operation_id')
        if operation_id and operation_id not in operation_ids:
            problems.append({'asset_id': asset['id'], 'problem': 'missing_operation'})
        try:
            array = store.load_array(asset['id'])
            checked_arrays += 1
            channels = array.shape[2] if array.ndim == 3 else 1
            if array.shape[:2] != (asset['height'], asset['width']) or channels != asset['channels']:
                problems.append({'asset_id': asset['id'], 'problem': 'array_shape_mismatch'})
            if not np.all(np.isfinite(array)):
                problems.append({'asset_id': asset['id'], 'problem': 'nonfinite_pixels'})
        except Exception as exc:
            problems.append({'asset_id': asset['id'], 'problem': 'unreadable_pixels', 'message': str(exc)})
    for operation in job['operations']:
        if operation['source_id'] not in ids:
            problems.append({'operation_id': operation['id'], 'problem': 'missing_source'})
        reference = operation.get('reference_id')
        if reference and reference not in ids:
            problems.append({'operation_id': operation['id'], 'problem': 'missing_reference'})
    calibration_ids = {calibration['id'] for calibration in job.get('calibrations', [])}
    for calibration in job.get('calibrations', []):
        revision_of = calibration.get('revision_of')
        if revision_of and revision_of not in calibration_ids:
            problems.append({'calibration_id': calibration['id'], 'problem': 'missing_revision_parent'})
        scan_ids = {scan['scan_id'] for scan in calibration['session'].get('scans', [])}
        if not scan_ids or not scan_ids.issubset(ids):
            problems.append({'calibration_id': calibration['id'], 'problem': 'missing_calibration_scan'})
        if calibration['status'] == 'confirmed' and not calibration.get('confirmed_model'):
            problems.append({'calibration_id': calibration['id'], 'problem': 'confirmed_without_model'})
    return {'job_id': job_id, 'ok': not problems, 'checked_arrays': checked_arrays, 'checked_operations': len(operation_ids),
            'checked_calibrations': len(calibration_ids), 'problems': problems}
