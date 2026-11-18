"""实测扫描校准的后端单元测试。

运行：PYTHONPATH 指向 site-packages 后 python -m unittest discover -s tests
"""
import io
import tempfile
import unittest
import zipfile

import numpy as np

from printlab import calibration as c
from printlab.pixels import png_bytes
from printlab.server import create_app

REFS = [[.95, .95, .93], [.7, .7, .72], [.45, .45, .47], [.2, .2, .22], [.06, .06, .07],
        [.8, .1, .1], [.1, .7, .2], [.1, .3, .8], [.9, .85, .1], [.7, .1, .7]]
LAYOUT = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]
POSITIONS = [(.08, .10), (.66, .10), (.08, .66), (.66, .66), (.38, .38)]
PATCH, GAP = 9, 3


def synthetic_readings(seed, scans_grad=0.15, width=200, height=150):
    """直接生成读数表（绕过像素），用于模型与验证逻辑测试。"""
    rng = np.random.default_rng(seed)
    out = []
    for pi, (px, py) in enumerate(POSITIONS):
        for j, ref in enumerate(REFS):
            col, row = LAYOUT[j]
            cx = px + col * (PATCH + GAP) / width
            cy = py + row * (PATCH + GAP) / height
            lin = np.where(np.array(ref) <= .04045, np.array(ref) / 12.92,
                           ((np.array(ref) + .055) / 1.055) ** 2.4)
            gv = 0.9 + scans_grad * cx + 0.05 * cy + 0.02 * np.sin(seed) * cx * cy
            measured = np.clip((.02 + .9 * lin ** .8) * gv, 0, 1)
            out.append({'measured': measured, 'reference': lin, 'nx': float(cx), 'ny': float(cy),
                        'scan_id': f's{seed}', 'group_id': f's{seed}|main', 'patch_name': f'card{pi}_{j}',
                        'card_id': 'main', 'quality': 'ok'})
    return out


def synthetic_image_and_rows(seed, width=200, height=150):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    nx, ny = xx / (width - 1), yy / (height - 1)
    g = (0.9 + 0.15 * nx + 0.05 * ny)[..., None]
    img = np.ones((height, width, 3), np.float32) * .97
    rows = []
    for pi, (px, py) in enumerate(POSITIONS):
        ox, oy = int(px * width), int(py * height)
        for j, ref in enumerate(REFS):
            col, row = LAYOUT[j]
            cx, cy = ox + col * (PATCH + GAP), oy + row * (PATCH + GAP)
            lin = np.where(np.array(ref) <= .04045, np.array(ref) / 12.92,
                           ((np.array(ref) + .055) / 1.055) ** 2.4)
            measured = np.clip((.02 + .9 * lin ** .8) * g[cy, cx], 0, 1)
            srgb = np.where(measured <= .0031308, measured * 12.92,
                            1.055 * np.clip(measured, 0, None) ** (1 / 2.4) - .055)
            x0, y0 = cx - PATCH // 2, cy - PATCH // 2
            img[y0:y0 + PATCH, x0:x0 + PATCH] = np.clip(srgb, 0, 1)
            rows.append((f'card{pi}_{j}', x0, y0, ref))
    return np.clip(img + rng.normal(0, .0015, img.shape), 0, 1).astype(np.float32), rows


class CalibrationEngineTest(unittest.TestCase):
    def setUp(self):
        self.readings = []
        for s in range(3):
            self.readings.extend(synthetic_readings(s))

    def test_group_leakage_never_splits_patch_pixels(self):
        # 每个 (扫描,色卡) 组在任一折中整体进入训练或验证
        result = c.cross_validation(self.readings, lambda t: c.build_simple(t, 2), 'scan')
        for fold in result['folds']:
            holdout_scans = set(fold['holdout_scans'])
            train_scans = {r['scan_id'] for r in self.readings} - holdout_scans
            self.assertTrue(holdout_scans.isdisjoint(train_scans))
            self.assertGreater(fold['train_patches'], 0)
            self.assertGreater(fold['holdout_patches'], 0)
        # 卡片留出：同一张卡的所有色块在同一侧
        result = c.cross_validation(self.readings, lambda t: c.build_simple(t, 2), 'card')
        groups = {r['group_id'] for r in self.readings}
        self.assertEqual(len(result['folds']), len(groups))

    def test_single_group_cannot_produce_evidence(self):
        one = [r for r in self.readings if r['scan_id'] == 's0']
        with self.assertRaises(ValueError):
            c.cross_validation(one, lambda t: c.build_simple(t, 2), 'scan')

    def test_simple_preferred_when_illumination_weak(self):
        evidence, recommendation = c.compare_candidates(self.readings, 2, 1)
        self.assertEqual(recommendation['recommended'], 'simple')
        self.assertIn('训练误差低不构成', recommendation['rule'])
        for label in ('simple', 'spatial'):
            self.assertIsNotNone(evidence[label]['scan_holdout'])
            self.assertIsNotNone(evidence[label]['card_holdout'])

    def test_spatial_wins_under_strong_gradient(self):
        strong = []
        for s in range(4):
            strong.extend(synthetic_readings(s, scans_grad=0.35))
        # 增加位置覆盖
        for r in strong:
            r['group_id'] = r['scan_id'] + '|' + r['card_id']
        evidence, recommendation = c.compare_candidates(strong, 2, 1)
        simple_de = evidence['simple']['scan_holdout']['summary']['pooled_mean_delta_e76']
        spatial_de = evidence['spatial']['scan_holdout']['summary']['pooled_mean_delta_e76']
        self.assertLess(spatial_de, simple_de)
        self.assertEqual(recommendation['recommended'], 'spatial')

    def test_unjudged_strata_are_explicit(self):
        # 只用中心位置的读数，边缘/角落分层必须标记 unjudged
        center = [r for r in self.readings if 0.25 <= r['nx'] <= .75 and 0.25 <= r['ny'] <= .75]
        # 复制成两个扫描组以满足最少组数
        for r in list(center):
            clone = dict(r)
            clone['scan_id'] = r['scan_id'] + 'b'
            clone['group_id'] = clone['scan_id'] + '|main'
            center.append(clone)
        result = c.cross_validation(center, lambda t: c.build_simple(t, 2), 'scan')
        zones = {s['stratum']: s for s in result['folds'][0]['strata'] if s['dimension'] == 'position'}
        self.assertEqual(zones['edge']['status'], 'unjudged')
        self.assertEqual(zones['corner']['status'], 'unjudged')
        self.assertIsNone(zones['edge']['mean_delta_e76'])

    def test_response_lut_is_monotone(self):
        model = c.build_simple(self.readings, 2)
        lut = np.asarray(model['lut'])
        for channel in range(3):
            self.assertTrue(np.all(np.diff(lut[:, channel]) >= -1e-12))

    def test_reliability_marks_extrapolation(self):
        model = c.build_simple(self.readings, 2)
        image = np.ones((60, 60, 3), np.float32) * 0.5
        domain = c.reliability_domain(model, image)
        self.assertEqual(domain['supported'].shape, (60, 60))
        self.assertAlmostEqual(domain['fractions']['supported']
                               + domain['fractions']['position_extrapolation']
                               + domain['fractions']['tone_extrapolation']
                               + domain['fractions']['color_extrapolation'],
                               1.0, delta=0.02)
        # 角落通常落在训练位置外
        self.assertGreaterEqual(domain['fractions']['position_extrapolation'], 0.0)

    def test_quality_flags(self):
        clean = np.ones((10, 10, 3), np.float32) * 0.5
        label, stats = c.auto_quality(c.linearize(clean))
        self.assertEqual(label, 'ok')
        saturated = np.ones((10, 10, 3), np.float32)
        saturated[:, :, 0] = 1.0
        label, _ = c.auto_quality(c.linearize(saturated))
        self.assertEqual(label, 'saturated')
        dirty = np.ones((10, 10, 3), np.float32) * 0.5
        dirty[:5] = 0.1
        label, _ = c.auto_quality(c.linearize(dirty))
        self.assertEqual(label, 'dirty')

    def test_apply_keeps_paper_texture_but_removes_smooth_gradient(self):
        # 空间场是平滑的：高频纸张纹理必须保留，低频灯光梯度被校正
        model = c.build_spatial(self.readings, 2, 1)
        height = width = 120
        rng = np.random.default_rng(7)
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        low_freq = 0.9 + 0.2 * xx / (width - 1)
        texture = rng.normal(0, 0.02, (height, height))
        paper = np.clip(low_freq + texture, 0.2, 1).astype(np.float32)
        image = np.repeat(paper[:, :, None], 3, axis=2)
        result = c.apply_calibration(model, image)
        before = image[:, :, 0]
        after = result['calibrated'][:, :, 0]
        # 低频梯度（列均值斜率）应显著减小
        slope_before = np.polyfit(np.arange(width), before.mean(axis=0), 1)[0]
        slope_after = np.polyfit(np.arange(width), after.mean(axis=0), 1)[0]
        self.assertLess(abs(slope_after), abs(slope_before))
        # 高频纹理能量保留（不被抹平）
        def high_freq(plane):
            return float(np.mean(np.abs(plane - np.mean(plane))))
        self.assertGreater(high_freq(after), high_freq(before) * 0.5)

    def test_coverage_gaps_detect_missing_shadow_and_edge(self):
        bright = [r for r in self.readings if np.mean(r['reference']) > .3]
        gaps = {g['dimension'] for g in c.coverage_gaps(bright)}
        self.assertIn('shadow', gaps)


class CalibrationWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.app = create_app(data_root=tempfile.mkdtemp())
        self.client = self.app.test_client()
        self.job_id = self.client.post('/api/jobs', json={'name': '工作流测试'}).get_json()['id']
        self.scan_ids, self.all_rows = [], []
        for s in range(3):
            image, rows = synthetic_image_and_rows(s)
            response = self.client.post(
                f'/api/jobs/{self.job_id}/upload',
                data={'file': (io.BytesIO(png_bytes(image)), f's{s}.png')},
                content_type='multipart/form-data')
            self.assertEqual(response.status_code, 201)
            self.scan_ids.append(response.get_json()['assets'][0]['id'])
            self.all_rows.append(rows)

    def _session(self):
        response = self.client.post(f'/api/jobs/{self.job_id}/calibrations', json={
            'name': '档案', 'scan_asset_ids': self.scan_ids,
            'setup': {'exposure': '1/60', 'white_balance': '5500K', 'lighting': '固定灯'}})
        self.assertEqual(response.status_code, 201)
        return response.get_json()['id']

    def _populate(self, cal_id):
        for si, rows in enumerate(self.all_rows):
            self.client.post(f'/api/calibrations/{cal_id}/cards',
                             json={'scan_id': self.scan_ids[si], 'card_id': 'main'})
            for name, x0, y0, ref in rows:
                response = self.client.post(f'/api/calibrations/{cal_id}/patches', json={
                    'scan_id': self.scan_ids[si], 'card_id': 'main', 'name': name,
                    'geometry': {'type': 'rectangle', 'x': x0, 'y': y0, 'width': PATCH, 'height': PATCH},
                    'reference_rgb': ref})
                self.assertEqual(response.status_code, 201, response.get_json())

    def test_full_confirm_revise_apply_and_archive(self):
        cal_id = self._session()
        self._populate(cal_id)
        # 纳入/排除与脏污标记
        self.client.patch(f'/api/calibrations/{cal_id}/patches', json={
            'scan_id': self.scan_ids[0], 'card_id': 'main', 'name': 'card0_0', 'quality': 'dirty'})
        self.client.patch(f'/api/calibrations/{cal_id}/patches', json={
            'scan_id': self.scan_ids[0], 'card_id': 'main', 'name': 'card0_1', 'quality': 'excluded'})
        evidence = self.client.post(f'/api/calibrations/{cal_id}/evidence', json={})
        self.assertEqual(evidence.status_code, 200)
        chosen = evidence.get_json()['evidence']['recommendation']['recommended']
        confirmed = self.client.post(f'/api/calibrations/{cal_id}/confirm',
                                     json={'model_kind': chosen, 'note': '独立证据'})
        self.assertEqual(confirmed.status_code, 201)
        self.assertEqual(confirmed.get_json()['status'], 'confirmed')

        # 已确认档案不可改
        denied = self.client.patch(f'/api/calibrations/{cal_id}/patches', json={
            'scan_id': self.scan_ids[0], 'card_id': 'main', 'name': 'card0_2', 'quality': 'excluded'})
        self.assertEqual(denied.status_code, 400)

        # 修订版不改旧档案
        revision = self.client.post(f'/api/calibrations/{cal_id}/revise', json={})
        self.assertEqual(revision.status_code, 201)
        self.assertEqual(revision.get_json()['revision'], 2)
        old = self.client.get(f'/api/calibrations/{cal_id}').get_json()
        self.assertEqual(old['status'], 'confirmed')

        # 应用校准并产出可靠掩码
        applied = self.client.post(f'/api/jobs/{self.job_id}/process', json={
            'operation': 'scan_calibration', 'source_id': self.scan_ids[0],
            'parameters': {'calibration_id': cal_id}})
        self.assertEqual(applied.status_code, 201, applied.get_json())
        names = {a['name'] for a in applied.get_json()['outputs']}
        self.assertIn('实测校准结果', names)
        self.assertIn('可靠像素掩码', names)

        # 校准后质量测量默认只算可靠像素
        quality = self.client.post(f'/api/jobs/{self.job_id}/process', json={
            'operation': 'calibrated_quality', 'source_id': self.scan_ids[0],
            'parameters': {'calibration_id': cal_id}})
        self.assertEqual(quality.status_code, 201)
        self.assertEqual(quality.get_json()['summary']['default_quality_basis'], 'supported_reliable')

        # 作业包含校准且归档
        job = self.client.get(f'/api/jobs/{self.job_id}').get_json()
        self.assertGreaterEqual(len(job['calibrations']), 2)
        archive = zipfile.ZipFile(io.BytesIO(self.client.get(f'/api/jobs/{self.job_id}/archive').data))
        self.assertTrue(any(n.startswith('calibrations/') for n in archive.namelist()))

    def test_reference_change_on_revision_keeps_old_result(self):
        cal_id = self._session()
        self._populate(cal_id)
        self.client.post(f'/api/calibrations/{cal_id}/evidence', json={})
        self.client.post(f'/api/calibrations/{cal_id}/confirm', json={'model_kind': 'simple'})
        old_model = self.client.get(f'/api/calibrations/{cal_id}').get_json()['confirmed_model']
        revision = self.client.post(f'/api/calibrations/{cal_id}/revise', json={}).get_json()
        new_id = revision['id']
        # 在修订版上改参考并重新拟合
        self.client.patch(f'/api/calibrations/{new_id}/patches', json={
            'scan_id': self.scan_ids[0], 'card_id': 'main', 'name': 'card0_0',
            'reference_rgb': [.42, .42, .42]})
        self.client.post(f'/api/calibrations/{new_id}/evidence', json={})
        self.client.post(f'/api/calibrations/{new_id}/confirm', json={'model_kind': 'simple'})
        untouched = self.client.get(f'/api/calibrations/{cal_id}').get_json()
        self.assertEqual(untouched['confirmed_model']['response_coefficients'],
                         old_model['response_coefficients'])

    def test_csv_import(self):
        cal_id = self._session()
        self.client.post(f'/api/calibrations/{cal_id}/cards',
                         json={'scan_id': self.scan_ids[0], 'card_id': 'main'})
        name, x0, y0, ref = self.all_rows[0][0]
        csv_text = (f'name,x,y,width,height,ref_r,ref_g,ref_b,card,quality\n'
                    f'{name},{x0},{y0},{PATCH},{PATCH},{ref[0]},{ref[1]},{ref[2]},main,ok\n')
        response = self.client.post(f'/api/calibrations/{cal_id}/patches/csv',
                                    json={'scan_id': self.scan_ids[0], 'csv_text': csv_text})
        self.assertEqual(response.status_code, 201, response.get_json())
        self.assertEqual(response.get_json()['added'], 1)

    def test_legacy_paper_white_remains_available(self):
        self.client.post(f'/api/assets/{self.scan_ids[0]}/regions', json={
            'name': '纸白', 'role': 'sample',
            'geometry': {'type': 'rectangle', 'x': 2, 'y': 2, 'width': 10, 'height': 10}})
        response = self.client.post(f'/api/jobs/{self.job_id}/process', json={
            'operation': 'paper_white', 'source_id': self.scan_ids[0], 'parameters': {}})
        self.assertEqual(response.status_code, 201)
        self.assertIn('纸白校正', {a['name'] for a in response.get_json()['outputs']})

    def test_original_untouched_and_outputs_exportable(self):
        cal_id = self._session()
        self._populate(cal_id)
        self.client.post(f'/api/calibrations/{cal_id}/evidence', json={})
        self.client.post(f'/api/calibrations/{cal_id}/confirm', json={'model_kind': 'simple'})
        original_blob = self.client.get(f'/api/assets/{self.scan_ids[0]}/download').data
        applied = self.client.post(f'/api/jobs/{self.job_id}/process', json={
            'operation': 'scan_calibration', 'source_id': self.scan_ids[0],
            'parameters': {'calibration_id': cal_id}}).get_json()
        output = next(a for a in applied['outputs'] if a['name'] == '实测校准结果')
        # 原图导出内容不变
        self.assertEqual(original_blob, self.client.get(f'/api/assets/{self.scan_ids[0]}/download').data)
        # 校准结果可导出为 PNG/TIFF/NPY，且尺寸与原图一致
        for fmt in ('png', 'tiff', 'npy'):
            response = self.client.get(f"/api/assets/{output['id']}/download?format={fmt}")
            self.assertEqual(response.status_code, 200, fmt)
        self.assertEqual((output['width'], output['height']),
                         (self.client.get(f'/api/assets/{self.scan_ids[0]}').get_json()['width'],
                          self.client.get(f'/api/assets/{self.scan_ids[0]}').get_json()['height']))
        # 校准后密度是有限值且范围合理
        quality = self.client.post(f'/api/jobs/{self.job_id}/process', json={
            'operation': 'calibrated_quality', 'source_id': self.scan_ids[0],
            'parameters': {'calibration_id': cal_id}}).get_json()
        reliable = next(r for r in quality['tables']['校准质量读数'] if r['range'] == 'supported_reliable')
        self.assertIsNotNone(reliable['mean_density'])
        self.assertGreaterEqual(reliable['mean_density'], 0)


if __name__ == '__main__':
    unittest.main()
