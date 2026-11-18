import io
import json
import logging
import os
import threading
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

from . import measurements, operations, plates, profiles, reports, spotcolor, targets, workflows
from .pixels import MAX_FILE_BYTES, gray_image, integer, json_safe, png_bytes, read_image, tiff_bytes
from .storage import Store


def create_app(data_root=None):
    project_root = Path(__file__).resolve().parent.parent
    app = Flask(__name__, template_folder=str(project_root / 'templates'), static_folder=str(project_root / 'static'))
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_BYTES + 1024 * 1024
    app.json.ensure_ascii = False
    root = data_root or os.environ.get('PRINTLAB_DATA_DIR', str(project_root / 'data'))
    store = Store(root)
    app.config['PRINTLAB_STORE'] = store
    processing_lock = threading.Lock()

    def payload():
        value = request.get_json(silent=True)
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError('请求内容必须为JSON对象')
        return value

    def response(value, status=200):
        return jsonify(json_safe(value)), status

    def output_file(data, mimetype, filename):
        return send_file(io.BytesIO(data), mimetype=mimetype, as_attachment=True, download_name=filename, max_age=0)

    @app.errorhandler(ValueError)
    def invalid(error):
        return response({'error': str(error), 'type': 'validation'}, 400)

    @app.errorhandler(KeyError)
    def missing(error):
        return response({'error': str(error).strip("'"), 'type': 'not_found'}, 404)

    @app.errorhandler(TypeError)
    def invalid_type(error):
        app.logger.warning('Parameter error: %s', error)
        return response({'error': '输入参数格式不符合所选操作：' + str(error), 'type': 'parameter'}, 400)

    @app.errorhandler(HTTPException)
    def http_error(error):
        message = '上传文件超过24 MiB限制' if error.code == 413 else str(error.description)
        return response({'error': message, 'type': 'http'}, error.code)

    @app.errorhandler(Exception)
    def unexpected(error):
        app.logger.exception('PrintLab request failed')
        return response({'error': '处理失败，输入和既有结果仍保留。请核对参数后重试。', 'type': 'internal'}, 500)

    @app.get('/')
    def home():
        return render_template('index.html')

    @app.get('/api/health')
    def health():
        return response(store.health())

    @app.get('/api/catalog')
    def catalog():
        target_list = targets.target_catalog()
        target_list += [{'id': 'step_wedge', 'name': '四版灰阶靶标', 'description': '带可测量色块位置的阶梯样张'}, {'id': 'control_strip', 'name': '实色控制条', 'description': 'CMYK实色、半调及叠印色块'}]
        return response({'operations': operations.operation_catalog(), 'targets': target_list, 'profile_defaults': profiles.default_profiles(), 'limits': {'maximum_pixels': 4_194_304, 'maximum_dimension': 4096, 'upload_bytes': MAX_FILE_BYTES, 'processing_concurrency': 1}, 'limitations': ['图像数值工作流不执行ICC转换', '油墨与纸张模拟为近似演示', '专色批分色使用与叠印预览一致的RGB透射率相乘模型，非光谱打样', '合成素材不是实际印机测量', '目前套准仅支持平移、人工仿射与四角平面校正']})

    @app.get('/api/jobs')
    def jobs_list():
        return response({'jobs': store.list_jobs()})

    @app.post('/api/jobs')
    def job_create():
        return response(workflows.create_job(store, payload()), 201)

    @app.post('/api/demo')
    def demo_create():
        body = payload()
        with processing_lock:
            result = workflows.create_demo(store, body.get('name', 'PrintLab 合成演示作业'))
        return response(result, 201)

    @app.get('/api/jobs/<job_id>')
    def job_get(job_id):
        return response(store.get_job(job_id))

    @app.patch('/api/jobs/<job_id>')
    def job_patch(job_id):
        return response(store.update_job(job_id, payload()))

    @app.get('/api/jobs/<job_id>/dashboard')
    def dashboard(job_id):
        return response(workflows.job_dashboard(store, job_id))

    @app.post('/api/jobs/<job_id>/upload')
    def upload(job_id):
        if 'file' not in request.files:
            raise ValueError('请选择上传图片')
        file = request.files['file']
        content = file.read(MAX_FILE_BYTES + 1)
        return response(workflows.import_upload(store, job_id, file.filename, content), 201)

    @app.post('/api/jobs/<job_id>/targets')
    def target_create(job_id):
        body = payload()
        with processing_lock:
            result = workflows.add_target(store, job_id, body.get('kind'), body.get('parameters', {}))
        return response(result, 201)

    @app.post('/api/jobs/<job_id>/process')
    def process(job_id):
        body = payload()
        if not processing_lock.acquire(blocking=False):
            return response({'error': '当前有图像正在处理，请等待结果后重试', 'type': 'busy'}, 409)
        try:
            result = operations.execute(store, job_id, body)
        finally:
            processing_lock.release()
        return response(result, 201)

    @app.get('/api/assets/<asset_id>')
    def asset_get(asset_id):
        return response(store.get_asset(asset_id))

    @app.get('/api/assets/<asset_id>/preview')
    def asset_preview(asset_id):
        return Response(store.preview(asset_id), mimetype='image/png', headers={'Cache-Control': 'private, max-age=3600'})

    @app.get('/api/assets/<asset_id>/download')
    def asset_download(asset_id):
        asset = store.get_asset(asset_id)
        array = store.load_array(asset_id)
        file_format = request.args.get('format', 'tiff' if asset['channels'] == 4 else 'png')
        if file_format == 'npy':
            stream = io.BytesIO()
            np.save(stream, array, allow_pickle=False)
            return output_file(stream.getvalue(), 'application/octet-stream', asset_id + '.npy')
        if file_format == 'tiff':
            dpi = store.get_job(asset['job_id'])['dpi']
            return output_file(tiff_bytes(array, dpi), 'image/tiff', asset_id + '.tiff')
        if file_format != 'png':
            raise ValueError('导出格式支持png、tiff或npy')
        if asset['channels'] == 4:
            array = operations.as_rgb(array)
        return output_file(png_bytes(array), 'image/png', asset_id + '.png')

    @app.get('/api/assets/<asset_id>/facts')
    def asset_facts(asset_id):
        return response(workflows.asset_facts(store, asset_id))

    @app.get('/api/assets/<asset_id>/lineage')
    def asset_lineage(asset_id):
        return response({'lineage': store.lineage(asset_id)})

    @app.post('/api/assets/<asset_id>/copy')
    def asset_copy(asset_id):
        return response(workflows.duplicate_asset(store, asset_id, payload().get('name')), 201)

    @app.post('/api/assets/<asset_id>/preflight')
    def preflight(asset_id):
        with processing_lock:
            result = workflows.preflight_asset(store, asset_id, payload())
        return response(result, 201)

    @app.post('/api/assets/<asset_id>/regions')
    def region_add(asset_id):
        body = payload()
        return response(store.add_region(asset_id, body.get('name', '区域'), body.get('geometry'), body.get('role', 'measure')), 201)

    @app.delete('/api/regions/<region_id>')
    def region_delete(region_id):
        return response(store.delete_region(region_id))

    @app.post('/api/jobs/<job_id>/profiles')
    def profile_save(job_id):
        body = payload()
        return response(store.save_profile(job_id, body.get('kind'), body.get('name'), body.get('payload'), body.get('id')), 201)

    @app.get('/api/profiles/<profile_id>')
    def profile_get(profile_id):
        return response(store.get_profile(profile_id))

    @app.get('/api/profiles/<profile_id>/curve-preview')
    def curve_preview(profile_id):
        profile = store.get_profile(profile_id)
        if profile['kind'] != 'curves':
            raise ValueError('只有曲线配置可生成曲线图')
        return Response(png_bytes(profiles.curve_preview(profile['payload'])), mimetype='image/png')

    @app.post('/api/profiles/compare')
    def profile_compare():
        body = payload()
        first = store.get_profile(body.get('first_id'))
        second = store.get_profile(body.get('second_id'))
        return response(profiles.profile_differences(first, second))

    @app.post('/api/jobs/<job_id>/notes')
    def note_add(job_id):
        body = payload()
        return response(store.add_note(job_id, body.get('text', ''), body.get('asset_id')), 201)

    @app.get('/api/jobs/<job_id>/events')
    def events(job_id):
        return response({'events': store.events(job_id, request.args.get('limit', 100))})

    @app.get('/api/operations/<operation_id>')
    def operation_get(operation_id):
        return response(store.get_operation(operation_id))

    @app.get('/api/operations/<operation_id>/table')
    def operation_table(operation_id):
        operation = store.get_operation(operation_id)
        key = request.args.get('name')
        if key not in operation['tables']:
            raise KeyError('结果表不存在')
        return output_file(reports.table_csv(operation['tables'][key]), 'text/csv; charset=utf-8', operation_id + '.csv')

    @app.post('/api/operations/compare')
    def operation_compare():
        body = payload()
        first = store.get_operation(body.get('first_id'))
        second = store.get_operation(body.get('second_id'))
        return response(reports.compare_operation_metrics(first, second))

    @app.get('/api/jobs/<job_id>/report')
    def report_get(job_id):
        report = reports.job_report(store, job_id)
        format_name = request.args.get('format', 'json')
        if format_name == 'html':
            return Response(reports.html_report(report), mimetype='text/html; charset=utf-8')
        if format_name == 'csv':
            return output_file(reports.table_csv(report['inventory']), 'text/csv; charset=utf-8', job_id + '-inventory.csv')
        if format_name != 'json':
            raise ValueError('报告格式支持json、html或csv')
        return response(report)

    @app.get('/api/jobs/<job_id>/archive')
    def archive_get(job_id):
        with processing_lock:
            data = store.export_archive(job_id, include_arrays=request.args.get('arrays') == '1')
        return output_file(data, 'application/zip', job_id + '-printlab.zip')

    @app.post('/api/jobs/<job_id>/spot-batches')
    def spot_batch_create(job_id):
        if not processing_lock.acquire(blocking=False):
            return response({'error': '当前有图像正在处理，请等待结果后重试', 'type': 'busy'}, 409)
        try:
            result = spotcolor.create_batch(store, job_id, payload())
        finally:
            processing_lock.release()
        return response(result, 201)

    @app.get('/api/jobs/<job_id>/spot-batches')
    def spot_batch_list(job_id):
        return response({'batches': store.list_spot_batches(job_id)})

    @app.get('/api/spot-batches/<batch_id>')
    def spot_batch_get(batch_id):
        return response(store.get_spot_batch(batch_id))

    @app.post('/api/spot-batches/<batch_id>/confirm')
    def spot_batch_confirm(batch_id):
        if not processing_lock.acquire(blocking=False):
            return response({'error': '当前有图像正在处理，请等待结果后重试', 'type': 'busy'}, 409)
        try:
            result = spotcolor.confirm_batch(store, batch_id, payload().get('plan_id'))
        finally:
            processing_lock.release()
        return response(result, 201)

    @app.get('/api/spot-batches/<batch_id>/plans/<plan_id>/preview/<asset_id>')
    def spot_plan_preview(batch_id, plan_id, asset_id):
        data = spotcolor.plan_preview_png(store, batch_id, plan_id, asset_id, request.args.get('kind', 'preview'))
        return Response(data, mimetype='image/png', headers={'Cache-Control': 'private, max-age=3600'})

    @app.get('/api/spot-batches/<batch_id>/usage')
    def spot_batch_usage(batch_id):
        rows = spotcolor.usage_rows(store.get_spot_batch(batch_id))
        if request.args.get('format') == 'csv':
            return output_file(reports.table_csv(rows), 'text/csv; charset=utf-8', batch_id + '-usage.csv')
        return response({'rows': rows})

    @app.get('/api/jobs/<job_id>/integrity')
    def integrity(job_id):
        with processing_lock:
            result = workflows.validate_job_links(store, job_id)
        return response(result)


    @app.post('/api/jobs/<job_id>/assemble-plates')
    def assemble_plates(job_id):
        store.get_job(job_id)
        channels = {}
        sources = {}
        for channel in ('C', 'M', 'Y', 'K'):
            if channel not in request.files:
                continue
            file = request.files[channel]
            array, alpha, metadata = read_image(file.read(MAX_FILE_BYTES + 1))
            channels[channel] = gray_image(array)
            sources[channel] = {'filename': file.filename, 'metadata': metadata}
        with processing_lock:
            result = plates.assemble_channels(channels, request.form.get('polarity', 'printed'))
            outputs = []
            for label, array in result['images'].items():
                kind = 'cmyk' if array.ndim == 3 and array.shape[2] == 4 else 'rgb'
                metadata = {'import_kind': 'separate_plates', 'plate_sources': sources, 'polarity': request.form.get('polarity', 'printed')}
                outputs.append(store.add_asset(job_id, label, kind, array, metadata))
        return response({'assets': outputs, 'summary': result['summary'], 'tables': result['tables']}, 201)

    @app.post('/api/assets/<asset_id>/replace-plate')
    def replace_plate(asset_id):
        body = payload()
        asset = store.get_asset(asset_id)
        replacement = store.get_asset(body.get('replacement_id'))
        if replacement['job_id'] != asset['job_id']:
            raise ValueError('替换图像必须来自同一作业')
        source = store.load_array(asset_id)
        other = store.load_array(replacement['id'])
        settings = {'channel': body.get('channel', 'K'), 'polarity': body.get('polarity', 'coverage')}
        with processing_lock:
            result = plates.replace_channel(source, operations.as_rgb(other), **settings)
            saved = store.save_operation(asset['job_id'], 'replace_plate', asset_id, replacement['id'], settings, result, 0)
        return response(saved, 201)

    @app.post('/api/measurements/compare')
    def compare_measurements():
        body = payload()
        first = store.get_operation(body.get('first_id'))
        second = store.get_operation(body.get('second_id'))
        if first['job_id'] != second['job_id']:
            raise ValueError('只能比较同一作业内的测量')
        first_key = body.get('first_table', '色块读数')
        second_key = body.get('second_table', '色块读数')
        if first_key not in first['tables'] or second_key not in second['tables']:
            raise ValueError('请选择包含色块RGB和密度读数的结果表')
        result = measurements.compare_patch_tables(first['tables'][first_key], second['tables'][second_key])
        return response(result)

    @app.get('/api/assets/<asset_id>/pixel')
    def pixel_sample(asset_id):
        array = store.load_array(asset_id)
        height, width = array.shape[:2]
        x = integer(request.args.get('x'), '像素x', 0, width - 1)
        y = integer(request.args.get('y'), '像素y', 0, height - 1)
        radius = integer(request.args.get('radius', 0), '采样半径', 0, 20)
        x0 = max(0, x - radius)
        y0 = max(0, y - radius)
        x1 = min(width, x + radius + 1)
        y1 = min(height, y + radius + 1)
        patch = array[y0:y1, x0:x1]
        rgb = operations.as_rgb(array)[y0:y1, x0:x1]
        if patch.ndim == 2:
            values = patch.reshape(-1, 1)
        else:
            values = patch.reshape(-1, patch.shape[2])
        return response({'x': x, 'y': y, 'radius': radius, 'sample_box': [x0, y0, x1 - x0, y1 - y0], 'stored_pixel': array[y, x], 'stored_mean': values.mean(axis=0), 'stored_std': values.std(axis=0), 'display_rgb_mean': rgb.mean(axis=(0, 1)), 'pixels': len(values)})

    @app.get('/api/assets/<asset_id>/tile')
    def image_tile(asset_id):
        array = store.load_array(asset_id)
        height, width = array.shape[:2]
        x = integer(request.args.get('x', 0), '裁片x', 0, width - 1)
        y = integer(request.args.get('y', 0), '裁片y', 0, height - 1)
        tile_width = integer(request.args.get('width', min(width - x, 512)), '裁片宽度', 1, min(width - x, 1024))
        tile_height = integer(request.args.get('height', min(height - y, 512)), '裁片高度', 1, min(height - y, 1024))
        rgb = operations.as_rgb(array)[y:y + tile_height, x:x + tile_width]
        return Response(png_bytes(rgb), mimetype='image/png')

    @app.get('/api/profiles/<profile_id>/curve-table')
    def profile_curve_table(profile_id):
        profile = store.get_profile(profile_id)
        if profile['kind'] != 'curves':
            raise ValueError('所选配置不是墨量曲线')
        rows = profiles.curve_table(profile['payload'], request.args.get('samples', 21))
        if request.args.get('format') == 'csv':
            return output_file(reports.table_csv(rows), 'text/csv; charset=utf-8', profile_id + '.csv')
        return response({'rows': rows})

    return app
