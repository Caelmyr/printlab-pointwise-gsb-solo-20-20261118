import inspect
import time

import numpy as np

from . import geometry, inspection, measurements, plates, prepress, press, regions, screening, separation, sheets, targets
from .pixels import CHANNELS, cmyk_image, gray_image, image_facts, json_safe, mask_image, rgb_image


OPERATIONS = {
    'dot_shapes': ('质量检查', '网点形状测量', screening.dot_shape_analysis, 'cmyk'),
    'feature_width': ('质量检查', '细线最小宽度检查', inspection.stroke_integrity, 'rgb'),
    'measure_csv': ('质量检查', 'CSV色块位置测量', measurements.measure_csv, 'rgb'),
    'patch_grid': ('质量检查', '色块网格测量', measurements.patch_template, 'rgb'),
    'channel_union': ('色版整理', '合并色版覆盖率', plates.channel_union, 'cmyk'),
    'plate_threshold': ('色版整理', '二值色版清理', plates.plate_threshold, 'cmyk'),
    'plate_edges': ('色版整理', '单版边缘检查', plates.plate_edges, 'cmyk'),
    'plate_overlap': ('色版整理', '多版重叠检查', plates.plate_overlap, 'cmyk'),
    'separate': ('颜色分解', '基础CMYK分色', separation.basic_separation, 'rgb'),
    'extract_plate': ('颜色分解', '单版提取', separation.plate_extract, 'cmyk'),
    'limit_ink': ('颜色分解', '总墨量限制', separation.limit_ink, 'cmyk'),
    'channel_mix': ('颜色分解', '油墨线性调整', separation.channel_mix, 'cmyk'),
    'spot_mask': ('颜色分解', '参考色选取专版', separation.spot_mask, 'rgb'),
    'grayscale': ('颜色分解', '单黑灰度分色', separation.grayscale_conversion, 'rgb'),
    'duotone': ('颜色分解', '双色调预览', separation.duotone, 'rgb'),
    'neutral_balance': ('颜色分解', '阶调颜色平衡', separation.neutral_balance, 'rgb'),
    'posterize': ('颜色分解', '色阶压缩', separation.posterize, 'rgb'),
    'ink_overview': ('颜色分解', '区域总墨量检查', separation.ink_overview, 'cmyk'),
    'knockout': ('颜色分解', '区域色版挖空', separation.knockout, 'cmyk'),
    'am_screen': ('网屏制作', 'AM旋转网点', screening.render_am, 'cmyk'),
    'fm_screen': ('网屏制作', 'FM随机网点', screening.render_fm, 'cmyk'),
    'spectrum': ('网屏制作', '频域网纹检查', screening.spectrum_analysis, 'rgb'),
    'banding': ('网屏制作', '横纵条带检查', screening.banding_analysis, 'rgb'),
    'repeat_seam': ('循环图案', '接缝检查', screening.repeat_seam, 'rgb'),
    'repeat': ('循环图案', '循环排布预览', geometry.tile_repeat, 'rgb'),
    'offset': ('循环图案', '接缝循环偏移', geometry.offset_preview, 'rgb'),
    'curves': ('印机与油墨', '应用墨量曲线', press.apply_ink_curves, 'cmyk'),
    'dot_gain': ('印机与油墨', '网点扩大模拟/补偿', press.dot_gain, 'cmyk'),
    'overprint': ('印机与油墨', '油墨叠印预览', press.overprint, 'cmyk'),
    'ink_estimate': ('印机与油墨', '印量用墨预算', press.ink_estimate, 'cmyk'),
    'crop': ('画布与套准', '裁切图像', geometry.crop_image, 'rgb'),
    'bleed': ('画布与套准', '添加出血及裁切线', geometry.add_bleed, 'rgb'),
    'rotate': ('画布与套准', '旋转画布', geometry.rotate_image, 'rgb'),
    'perspective': ('画布与套准', '四角平面校正', geometry.perspective_rectify, 'rgb'),
    'plate_shift': ('画布与套准', '单版偏移模拟', geometry.translate_plates, 'cmyk'),
    'translation': ('画布与套准', '全图平移套准', geometry.register_translation, 'pair_rgb'),
    'landmarks': ('画布与套准', '人工标点仿射套准', geometry.register_landmarks, 'pair_rgb'),
    'manual_trap': ('画布与套准', '手工单版边缘补偿', geometry.manual_trap, 'cmyk'),
    'compare': ('质量检查', '对应像素差异检查', inspection.compare_images, 'pair_rgb'),
    'density': ('质量检查', '网格密度检查', inspection.density_grid, 'rgb'),
    'sharpness': ('质量检查', '边缘清晰度检查', inspection.sharpness_analysis, 'rgb'),
    'uniformity': ('质量检查', '均匀性检查', inspection.uniformity_inspection, 'rgb'),
    'histogram': ('质量检查', '通道直方图', inspection.histogram_analysis, 'rgb'),
    'line_profile': ('质量检查', '线段剖面测量', inspection.line_profile, 'rgb'),
    'threshold_region': ('区域工具', '亮度选区', regions.threshold_region, 'rgb'),
    'edge_region': ('区域工具', '边缘选区', regions.edge_region, 'rgb'),
    'morphology': ('区域工具', '选区形态处理', regions.morphology_mask, 'gray'),
    'masked_edit': ('区域工具', '区域图像处理', regions.masked_edit, 'rgb'),
    'sample_grid': ('区域工具', '网格颜色采样', regions.sample_grid, 'rgb'),
    'region_distance': ('区域工具', '选区边界距离', regions.region_distance, 'gray'),
    'levels': ('扫描整理', '手动色阶', prepress.levels, 'rgb'),
    'auto_levels': ('扫描整理', '参考区域自动色阶', prepress.auto_levels, 'rgb'),
    'paper_white': ('扫描整理', '纸白参考校正', prepress.paper_white, 'rgb'),
    'background': ('扫描整理', '背景亮度整理', prepress.flatten_background, 'rgb'),
    'denoise': ('扫描整理', '去噪', prepress.denoise, 'rgb'),
    'unsharp': ('扫描整理', '锐化', prepress.unsharp, 'rgb'),
    'resample': ('扫描整理', '印刷分辨率重采样', prepress.resample_print, 'rgb'),
    'paper_preview': ('扫描整理', '纸张观感模拟', prepress.poster_preview, 'rgb'),
    'local_contrast': ('扫描整理', '局部反差调整', prepress.local_contrast, 'rgb'),
    'trim_whitespace': ('扫描整理', '自动去纸白边', prepress.trim_whitespace, 'rgb'),
    'border': ('扫描整理', '图像边框', prepress.color_border, 'rgb'),
    'binary': ('扫描整理', '二值线稿', prepress.threshold_halftone_preview, 'rgb'),
    'grid_sheet': ('印张排布', '规则格拼版', sheets.grid_sheet, 'rgb'),
    'fixed_sheet': ('印张排布', '定尺寸拼版', sheets.fixed_size_sheet, 'rgb'),
    'sheet_compare': ('印张排布', '纸张方向比较', sheets.compare_sheet_layouts, 'rgb'),
    'ruler': ('印张排布', '尺寸标尺', sheets.ruler_overlay, 'rgb'),
}


SPECIAL = {
    'region_report': ('区域工具', '已存区域测量', 'rgb'),
    'measure_wedge': ('印机与油墨', '测量灰阶靶标', 'rgb'),
    'measure_bars': ('印机与油墨', '测量实色控制条', 'rgb'),
    'screen_coverage': ('网屏制作', '网版覆盖率对照', 'pair_cmyk'),
    'contact_sheet': ('印张排布', '作业图像联系表', 'rgb'),
}


def parameter_schema(function, kind):
    parameters = []
    signature = inspect.signature(function)
    hidden = {'rgb', 'cmyk', 'source', 'reference', 'candidate', 'mask', 'sample_mask', 'score', 'source_rgb', 'coverage'}
    for name, parameter in signature.parameters.items():
        if name in hidden:
            continue
        required = parameter.default is inspect.Parameter.empty
        default = None if required else parameter.default
        if isinstance(default, bool):
            value_type = 'boolean'
        elif isinstance(default, int):
            value_type = 'integer'
        elif isinstance(default, float):
            value_type = 'number'
        elif isinstance(default, (list, tuple, dict)) or default is None:
            value_type = 'json'
        else:
            value_type = 'string'
        parameters.append({'name': name, 'required': required, 'type': value_type, 'default': json_safe(default)})
    return parameters


def operation_catalog():
    records = []
    for name, (category, title, function, kind) in OPERATIONS.items():
        records.append({'id': name, 'category': category, 'name': title, 'input_kind': kind, 'parameters': parameter_schema(function, kind)})
    for name, (category, title, kind) in SPECIAL.items():
        records.append({'id': name, 'category': category, 'name': title, 'input_kind': kind, 'parameters': []})
    return records


def as_rgb(array):
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2)
    if array.shape[2] == 4:
        return separation.composite_simple(array)
    return rgb_image(array)


def as_cmyk(array):
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError('请先选择一张CMYK分色结果作为输入')
    return cmyk_image(array)


def operation_mask(asset, shape, region_ids):
    selected_regions = asset.get('regions', [])
    if region_ids:
        known = {row['id'] for row in selected_regions}
        if not set(region_ids).issubset(known):
            raise ValueError('选择的区域不属于当前图像')
        selected_regions = [row for row in selected_regions if row['id'] in region_ids]
    return regions.combine_regions(selected_regions, shape, default_full=True)


def target_patches(asset):
    metadata = asset.get('metadata', {})
    tables = metadata.get('target_tables', {})
    patches = tables.get('patches')
    if patches is None:
        raise ValueError('当前图像没有靶标色块位置；请使用生成的灰阶或控制条图像')
    return patches


def execute_special(store, name, asset, array, reference_asset, reference_array, parameters, mask):
    source = as_rgb(array)
    if name == 'region_report':
        return regions.region_summary(source, asset['regions'])
    if name == 'screen_coverage':
        if reference_array is None:
            raise ValueError('请选择原始连续调分色作为参考')
        return screening.screen_coverage(as_cmyk(reference_array), as_cmyk(array), **parameters)
    if name == 'contact_sheet':
        job = store.get_job(asset['job_id'])
        chosen = parameters.pop('asset_ids', None)
        asset_rows = [item for item in job['assets'] if chosen is None or item['id'] in chosen]
        if not asset_rows:
            raise ValueError('没有可用于联系表的图像')
        asset_rows = asset_rows[:36]
        images = [as_rgb(store.load_array(item['id'])) for item in asset_rows]
        return sheets.contact_sheet(images, names=[item['name'] for item in asset_rows], **parameters)
    patches = target_patches(asset)
    if name == 'measure_bars':
        channels = [patch.get('channel', 'K') for patch in patches]
        return press.measure_color_bars(source, patches, channels=channels, **parameters)
    if name == 'measure_wedge':
        channel = parameters.pop('channel', 'K')
        selected = [patch for patch in patches if patch.get('channel') == channel]
        if not selected:
            raise ValueError('靶标没有该色版灰阶')
        nominal = [patch.get('nominal', patch.get('coverage', 0)) for patch in selected]
        return press.analyse_step_wedge(source, selected, nominal, channel=channel, **parameters)
    raise ValueError('处理操作不存在')


def apply_profile(store, profile_id, job_id, name, parameters):
    if not profile_id:
        return parameters
    profile = store.get_profile(profile_id)
    if profile['job_id'] != job_id:
        raise ValueError('配置不属于当前作业')
    payload = profile['payload']
    if name == 'curves' and profile['kind'] == 'curves':
        parameters['curves'] = payload['curves']
    elif name == 'overprint' and profile['kind'] == 'inks':
        parameters['ink_rgb'] = [ink['rgb'] for ink in payload['inks']]
        parameters['paper_rgb'] = payload['paper_rgb']
    elif name in ('am_screen', 'fm_screen') and profile['kind'] == 'screen':
        if name == 'am_screen':
            parameters.update({'lpi': payload['lpi'], 'dpi': payload['dpi'], 'angles': payload['angles'], 'shape': payload['dot_shape']})
        else:
            parameters['seed'] = payload['seed']
    elif name == 'paper_preview' and profile['kind'] == 'paper':
        parameters['paper_rgb'] = payload['rgb']
    else:
        raise ValueError('该配置种类不适用于所选操作')
    return parameters


def execute(store, job_id, payload):
    name = payload.get('operation')
    if name not in OPERATIONS and name not in SPECIAL:
        raise ValueError('处理操作不存在')
    source_id = payload.get('source_id')
    if not source_id:
        raise ValueError('请先选择输入图像')
    asset = store.get_asset(source_id)
    if asset['job_id'] != job_id:
        raise ValueError('输入图像不属于当前作业')
    array = store.load_array(source_id)
    reference_id = payload.get('reference_id') or None
    reference_asset = None
    reference_array = None
    if reference_id:
        reference_asset = store.get_asset(reference_id)
        if reference_asset['job_id'] != job_id:
            raise ValueError('参考图像不属于当前作业')
        reference_array = store.load_array(reference_id)
    parameters = dict(payload.get('parameters') or {})
    parameters = apply_profile(store, payload.get('profile_id'), job_id, name, parameters)
    saved_parameters = dict(parameters)
    saved_parameters['region_ids'] = payload.get('region_ids') or []
    if payload.get('profile_id'):
        saved_parameters['profile_id'] = payload['profile_id']
    mask = operation_mask(asset, array.shape, payload.get('region_ids'))
    started = time.perf_counter()
    if name in SPECIAL:
        result = execute_special(store, name, asset, array, reference_asset, reference_array, parameters, mask)
    else:
        category, title, function, kind = OPERATIONS[name]
        signature = inspect.signature(function)
        allowed = set(signature.parameters)
        unknown = set(parameters) - allowed
        if unknown:
            raise ValueError('不支持的参数：' + '、'.join(sorted(unknown)))
        if 'mask' in allowed and next(iter(signature.parameters)) != 'mask':
            parameters['mask'] = mask
        if 'sample_mask' in allowed:
            if not asset['regions']:
                raise ValueError('请先在图像上添加纸白参考区域')
            parameters['sample_mask'] = mask
        if kind == 'cmyk':
            result = function(as_cmyk(array), **parameters)
        elif kind == 'gray':
            result = function(gray_image(array), **parameters)
        elif kind == 'pair_rgb':
            if reference_array is None:
                raise ValueError('该操作需要参考图像')
            if name == 'compare':
                result = function(as_rgb(reference_array), as_rgb(array), **parameters)
            else:
                result = function(as_rgb(array), as_rgb(reference_array), **parameters)
        else:
            result = function(as_rgb(array), **parameters)
    elapsed_ms = (time.perf_counter() - started) * 1000
    result.setdefault('summary', {})['elapsed_ms'] = elapsed_ms
    result['summary']['source_asset_id'] = source_id
    result['summary']['reference_asset_id'] = reference_id
    for label, output in result.get('images', {}).items():
        values = np.asarray(output)
        if values.shape[0] * values.shape[1] > 4_194_304:
            raise ValueError('处理结果超过419万像素预算')
        if not np.all(np.isfinite(values)):
            raise ValueError('处理结果出现非有限值')
    return store.save_operation(job_id, name, source_id, reference_id, saved_parameters, result, elapsed_ms)
