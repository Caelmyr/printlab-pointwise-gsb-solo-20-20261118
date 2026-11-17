import csv
import html
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

import cv2
import numpy as np

from .pixels import CHANNELS, cmyk_image, heatmap, image_facts, integer, json_safe, luminance, number, overlay_mask, percentile_statistics, rgb_image
from .separation import composite_simple, plate_statistics


def image_preflight(rgb, cmyk=None, dpi=300, minimum_dpi=150, ink_limit=3, border_pixels=8, black_threshold=0.15):
    image = rgb_image(rgb)
    resolution = number(dpi, '当前DPI', 36, 2400)
    required = number(minimum_dpi, '最低DPI', 36, 1200)
    limit = number(ink_limit, '总墨量警戒', 1, 4)
    border = integer(border_pixels, '边界检查宽度', 1, 128)
    darkness = number(black_threshold, '深色阈值', 0.01, 0.5)
    height, width = image.shape[:2]
    edge_width = min(border, height // 2, width // 2)
    perimeter = np.zeros((height, width), dtype=bool)
    perimeter[:edge_width] = True
    perimeter[-edge_width:] = True
    perimeter[:, :edge_width] = True
    perimeter[:, -edge_width:] = True
    light = luminance(image)
    dark = light < darkness
    colorful = image.max(axis=2) - image.min(axis=2) > 0.2
    clipped = np.any((image <= 1 / 255) | (image >= 254 / 255), axis=2)
    checks = []
    checks.append({'check': '印刷分辨率', 'status': 'pass' if resolution >= required else 'warning', 'value': resolution, 'threshold': required, 'unit': 'DPI', 'detail': '按当前作业DPI解释像素尺寸，不依据文件名推断'})
    edge_fraction = float(np.mean(dark[perimeter]))
    checks.append({'check': '深色触边', 'status': 'warning' if edge_fraction > 0.05 else 'pass', 'value': edge_fraction, 'threshold': 0.05, 'unit': 'fraction', 'detail': '触边内容可能是设计的一部分；请核对出血与裁切线'})
    clipping_fraction = float(clipped.mean())
    checks.append({'check': '端点像素', 'status': 'info', 'value': clipping_fraction, 'threshold': None, 'unit': 'fraction', 'detail': '仅统计通道端点；不自动认定丢失细节'})
    maps = {'深色触边': overlay_mask(image, dark & perimeter)}
    if cmyk is not None:
        plates = cmyk_image(cmyk)
        if plates.shape[:2] != image.shape[:2]:
            raise ValueError('分色和预览大小不同')
        total = plates.sum(axis=2)
        over = total > limit
        rich_black = dark & (plates[:, :, :3].sum(axis=2) > 0.3)
        ink_fraction = float(over.mean())
        checks.append({'check': '总墨量', 'status': 'warning' if np.any(over) else 'pass', 'value': float(total.max() * 100), 'threshold': limit * 100, 'unit': 'percent', 'detail': f'{int(over.sum())}个像素超过警戒量'})
        checks.append({'check': '深色区域含彩墨', 'status': 'info', 'value': int(rich_black.sum()), 'threshold': None, 'unit': 'pixels', 'detail': '仅作为分色审阅线索；不自动判断细字或照片阴影'})
        maps['总墨量警戒'] = overlay_mask(image, over)
        maps['深色彩墨区域'] = overlay_mask(image, rich_black, color=(0.1, 0.4, 1))
    facts = image_facts(image, resolution)
    warnings = [check for check in checks if check['status'] == 'warning']
    return {'summary': {'checks': len(checks), 'warnings': len(warnings), 'dpi': resolution, 'physical_size_mm': [facts['width_mm'], facts['height_mm']], 'automatic_release': False, 'scope': '像素与阈值预检，不替代操作员对印张用途的确认'}, 'images': maps, 'tables': {'预检条目': checks, '图像统计': [facts]}}


def numeric_rows(table):
    if not isinstance(table, list):
        return []
    result = []
    for row in table:
        if isinstance(row, dict):
            result.append(row)
    return result


def table_csv(rows):
    records = numeric_rows(rows)
    if not records:
        return '\ufeff'.encode('utf-8')
    columns = []
    seen = set()
    for row in records:
        for key in row:
            if key not in seen:
                columns.append(key)
                seen.add(key)
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    for row in records:
        encoded = {}
        for key in columns:
            value = json_safe(row.get(key))
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)
            if isinstance(value, str) and value.startswith(('=', '+', '-', '@')):
                value = "'" + value
            encoded[key] = value
        writer.writerow(encoded)
    return ('\ufeff' + output.getvalue()).encode('utf-8')


def summary_metrics(job):
    assets = job.get('assets', [])
    operations = job.get('operations', [])
    kinds = Counter(asset['kind'] for asset in assets)
    operation_counts = Counter(operation['operation'] for operation in operations)
    sizes = [asset['width'] * asset['height'] for asset in assets]
    durations = [operation['elapsed_ms'] for operation in operations]
    sources = [asset for asset in assets if not asset.get('parent_id')]
    derived = [asset for asset in assets if asset.get('parent_id')]
    return {'asset_count': len(assets), 'source_count': len(sources), 'derived_count': len(derived), 'operation_count': len(operations), 'profile_count': len(job.get('profiles', [])), 'note_count': len(job.get('notes', [])), 'kind_counts': dict(kinds), 'operation_counts': dict(operation_counts), 'total_stored_pixels': int(sum(sizes)), 'maximum_image_pixels': max(sizes, default=0), 'total_processing_ms': float(sum(durations)), 'median_processing_ms': float(np.median(durations)) if durations else 0.0}


def lineage_graph(job):
    asset_map = {asset['id']: asset for asset in job.get('assets', [])}
    operation_map = {operation['id']: operation for operation in job.get('operations', [])}
    nodes = []
    edges = []
    for asset in asset_map.values():
        nodes.append({'id': asset['id'], 'name': asset['name'], 'kind': asset['kind'], 'size': [asset['width'], asset['height']], 'created_at': asset['created_at']})
        parent = asset.get('parent_id')
        operation_id = asset.get('operation_id')
        if parent:
            operation = operation_map.get(operation_id, {})
            edges.append({'source': parent, 'target': asset['id'], 'operation_id': operation_id, 'operation': operation.get('operation'), 'reference_id': operation.get('reference_id')})
    return {'nodes': nodes, 'edges': edges, 'missing_parents': [edge['source'] for edge in edges if edge['source'] not in asset_map]}


def output_inventory(job):
    records = []
    dpi = float(job.get('dpi', 300))
    for asset in job.get('assets', []):
        records.append({'asset_id': asset['id'], 'name': asset['name'], 'kind': asset['kind'], 'width_pixels': asset['width'], 'height_pixels': asset['height'], 'channels': asset['channels'], 'width_mm_at_job_dpi': asset['width'] / dpi * 25.4, 'height_mm_at_job_dpi': asset['height'] / dpi * 25.4, 'parent_id': asset.get('parent_id'), 'operation_id': asset.get('operation_id'), 'created_at': asset['created_at']})
    return records


def quality_observations(job):
    observations = []
    numeric_keys = {'rms_rgb_error', 'before_rmse', 'after_rmse', 'phase_response', 'rms_landmark_error', 'max_landmark_error', 'clipped_fraction', 'mean_coverage_percent', 'total_mean_percent', 'mean_density', 'max_density', 'mean_delta_e', 'rms_error', 'defect_fraction', 'narrow_skeleton_pixels'}
    for operation in job.get('operations', []):
        summary = operation.get('summary', {})
        for key, value in summary.items():
            if key in numeric_keys and isinstance(value, (int, float)):
                observations.append({'operation_id': operation['id'], 'operation': operation['operation'], 'metric': key, 'value': value, 'source_id': operation['source_id'], 'created_at': operation['created_at']})
        limitation = summary.get('limitation') or summary.get('limitations')
        if limitation:
            observations.append({'operation_id': operation['id'], 'operation': operation['operation'], 'metric': 'limitation', 'value': limitation, 'source_id': operation['source_id'], 'created_at': operation['created_at']})
    return observations


def job_report(store, job_id):
    job = store.get_job(job_id)
    report = {
        'format': 'printlab-report-v1',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'job': {key: job[key] for key in ('id', 'name', 'description', 'dpi', 'created_at', 'updated_at', 'revision')},
        'metrics': summary_metrics(job),
        'inventory': output_inventory(job),
        'lineage': lineage_graph(job),
        'observations': quality_observations(job),
        'profiles': job.get('profiles', []),
        'notes': job.get('notes', []),
        'operations': job.get('operations', []),
        'limitations': ['合成演示素材均显式标注。', 'RGB近似油墨模型不代替ICC色彩管理或印刷认证。', '默认所有图像操作生成新结果，保留原图和参数。', '几何套准目前仅提供全局平移、人工仿射和四角平面校正。'],
    }
    return json_safe(report)


def format_value(value):
    if value is None:
        return '—'
    if isinstance(value, float):
        return f'{value:.6g}'
    if isinstance(value, (dict, list)):
        return json.dumps(json_safe(value), ensure_ascii=False)
    return str(value)


def html_table(rows, maximum=100):
    records = numeric_rows(rows)[:maximum]
    if not records:
        return '<p>无记录</p>'
    keys = []
    for row in records:
        for key in row:
            if key not in keys:
                keys.append(key)
    header = ''.join('<th>' + html.escape(str(key)) + '</th>' for key in keys)
    body = []
    for row in records:
        cells = ''.join('<td>' + html.escape(format_value(row.get(key))) + '</td>' for key in keys)
        body.append('<tr>' + cells + '</tr>')
    return '<table><thead><tr>' + header + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table>'


def html_report(report):
    job = report['job']
    title = html.escape(job['name'])
    pieces = []
    pieces.append('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PrintLab 作业报告</title>')
    pieces.append('<style>body{font-family:system-ui,sans-serif;margin:32px;color:#172830}table{border-collapse:collapse;font-size:12px;max-width:100%;display:block;overflow:auto}td,th{border:1px solid #ccd5d9;padding:7px;vertical-align:top;text-align:left}h2{margin-top:32px}pre{white-space:pre-wrap}small{color:#566}section{break-inside:avoid}</style>')
    pieces.append('<h1>PrintLab · ' + title + '</h1>')
    pieces.append('<p>' + html.escape(job['description']) + '</p>')
    pieces.append('<p>报告生成时间：' + html.escape(report['generated_at']) + '</p>')
    pieces.append('<h2>作业概况</h2>')
    pieces.append(html_table([report['metrics']]))
    pieces.append('<h2>图像清单</h2>')
    pieces.append(html_table(report['inventory']))
    pieces.append('<h2>质量观测</h2>')
    pieces.append(html_table(report['observations']))
    for operation in reversed(report['operations']):
        pieces.append('<section><h2>' + html.escape(operation['operation']) + '</h2>')
        pieces.append('<small>' + html.escape(operation['created_at']) + ' · ' + html.escape(operation['id']) + '</small>')
        pieces.append('<h3>输入与参数</h3><pre>' + html.escape(json.dumps(operation['parameters'], ensure_ascii=False, indent=2)) + '</pre>')
        pieces.append('<h3>处理摘要</h3>' + html_table([operation['summary']]))
        for label, rows in operation['tables'].items():
            pieces.append('<h3>' + html.escape(label) + '</h3>')
            pieces.append(html_table(rows))
        pieces.append('</section>')
    pieces.append('<h2>备注</h2>')
    pieces.append(html_table(report['notes']))
    pieces.append('<h2>适用边界</h2><ul>')
    for limitation in report['limitations']:
        pieces.append('<li>' + html.escape(limitation) + '</li>')
    pieces.append('</ul></html>')
    return ''.join(pieces).encode('utf-8')


def compare_operation_metrics(first, second):
    if first['operation'] != second['operation']:
        raise ValueError('请比较相同处理方式的两次结果')
    keys = set(first['summary']) | set(second['summary'])
    rows = []
    for key in sorted(keys):
        a = first['summary'].get(key)
        b = second['summary'].get(key)
        if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
            delta = b - a
            relative = delta / abs(a) if abs(a) > 1e-12 else None
            rows.append({'metric': key, 'first': a, 'second': b, 'change': delta, 'relative_change': relative})
    changed_parameters = []
    parameter_keys = set(first['parameters']) | set(second['parameters'])
    for key in sorted(parameter_keys):
        a = first['parameters'].get(key)
        b = second['parameters'].get(key)
        if a != b:
            changed_parameters.append({'parameter': key, 'first': a, 'second': b})
    return {'operation': first['operation'], 'first_id': first['id'], 'second_id': second['id'], 'same_source': first['source_id'] == second['source_id'], 'same_reference': first.get('reference_id') == second.get('reference_id'), 'metrics': rows, 'parameter_changes': changed_parameters, 'interpretation': '不同输入的指标差值仅为记录比较，不视为算法改善证据'}
