"""Standalone engine validation for the statistical screening module.

Runs entirely in-memory: synthetic normal sheets with halftone phase drift and
scan noise, plus defect sheets with missing print, a spot and a broken stroke.
Checks that whole-sheet hold-out validation keeps false alarms low while
known defects are detected, without ever blurring the candidate.
"""

import sys

import cv2
import numpy as np

sys.path.insert(0, '/workspace')
from printlab import sheet_screening as screening
from printlab.screening import _put_text


def make_base(width=320, height=240, seed=7):
    rng = np.random.default_rng(seed)
    image = np.ones((height, width, 3), dtype=np.float32) * 0.95
    cv2.rectangle(image, (24, 24), (150, 120), (0.15, 0.3, 0.55), -1)
    cv2.rectangle(image, (180, 30), (290, 110), (0.6, 0.18, 0.22), -1)
    for line, text in enumerate(('FINE TYPE 0123456789', 'ABCDEFG small glyphs', 'thin stroke line')):
        _put_text(image, text, (170, 140 + line * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0.05, 0.05, 0.07), 1, cv2.LINE_AA)
    yy, xx = np.mgrid[0:height, 0:width]
    pitch = 8
    jx = rng.uniform(-1, 1, (height // pitch + 2, width // pitch + 2)).astype(np.float32)
    jy = rng.uniform(-1, 1, (height // pitch + 2, width // pitch + 2)).astype(np.float32)
    gx = (xx / pitch).astype(np.int32)
    gy = (yy / pitch).astype(np.int32)
    ax = xx - gx * pitch + jx[np.clip(gy, 0, jy.shape[0] - 1), np.clip(gx, 0, jx.shape[1] - 1)] - pitch / 2
    ay = yy - gy * pitch + jy[np.clip(gy, 0, jy.shape[0] - 1), np.clip(gx, 0, jy.shape[1] - 1)] - pitch / 2
    dots = (np.hypot(ax, ay) < 2.2).astype(np.float32)
    field = np.zeros((height, width), bool)
    field[150:225, 25:295] = True
    image[field] = image[field] * (1 - 0.5 * dots[field, None])
    return image


def vary(base, seed):
    rng = np.random.default_rng(seed)
    matrix = np.float32([[1, 0, rng.uniform(-0.7, 0.7)], [0, 1, rng.uniform(-0.7, 0.7)]])
    out = cv2.warpAffine(base, matrix, (base.shape[1], base.shape[0]), borderValue=(0.95, 0.95, 0.95))
    out *= np.float32([rng.uniform(0.97, 1.02), rng.uniform(0.97, 1.02), rng.uniform(0.96, 1.01)])
    out += rng.normal(0, 0.004, out.shape).astype(np.float32)
    return np.clip(out, 0, 1)


def main():
    base = make_base()
    normals = [vary(base, 100 + i) for i in range(8)]
    defects = []
    annotations = []
    d1 = vary(base, 500)
    cv2.circle(d1, (80, 70), 8, (0.9, 0.9, 0.87), -1)
    defects.append(d1)
    annotations.append([{'id': 'a1', 'role': 'known_defect', 'label': '漏印', 'kind': 'missing_print',
                         'geometry': {'type': 'rectangle', 'x': 72, 'y': 62, 'width': 16, 'height': 16}}])
    d2 = vary(base, 501)
    cv2.circle(d2, (235, 70), 6, (0.03, 0.03, 0.04), -1)
    defects.append(d2)
    annotations.append([{'id': 'a2', 'role': 'known_defect', 'label': '污点', 'kind': 'spot',
                         'geometry': {'type': 'rectangle', 'x': 229, 'y': 64, 'width': 12, 'height': 12}}])
    d3 = vary(base, 502)
    cv2.rectangle(d3, (215, 158), (232, 170), (0.95, 0.95, 0.95), -1)
    defects.append(d3)
    annotations.append([{'id': 'a3', 'role': 'known_defect', 'label': '断笔', 'kind': 'broken_stroke',
                         'geometry': {'type': 'rectangle', 'x': 215, 'y': 158, 'width': 17, 'height': 12}}])

    report = screening.validate_plan(normals, defects, {'noise_floor': 0.02, 'min_support': 3},
                                     {i: notes for i, notes in enumerate(annotations)})
    print('baseline:', report['baseline'])
    for name, result in report['policies'].items():
        print('\n===', name, result['label'])
        print('false alarm sheets:', result['false_alarm_sheets'], '/', len(normals),
              'upper95:', round(result['false_alarm_rate_upper95'], 3),
              'meets target:', result['meets_target_false_alarm'])
        print('sample sufficiency:', result['sample_sufficiency']['sufficient'], result['sample_sufficiency']['note'])
        print('recall:', result['known_defect_detected'], '/', result['known_defect_total'],
              'lower95:', result['recall_lower95'])
        print('recall by kind:', result['recall_by_kind'])
        for run in result['defect_sheet_runs']:
            print('  defect sheet', run['sheet_index'], run['verdict'], [(m['label'], m['detected'], m.get('matched_class')) for m in run['matches']])

    # Inspect a fresh normal candidate with the confirmed-style envelope.
    envelope = screening.build_envelope(normals, noise_floor=0.02, min_support=3)
    fresh = vary(base, 777)
    run = screening.inspect_sheet(fresh, envelope, policy='low_false_alarm', annotations=[], baseline=report['baseline'])
    print('\nfresh normal verdict:', run['summary']['verdict'], 'components:', run['summary']['components'],
          'defect:', run['summary']['defect_review_components'])

    # Inspect a defect candidate and check evidence crop.
    run_d = screening.inspect_sheet(d3, envelope, policy='high_sensitivity', annotations=annotations[2], baseline=report['baseline'])
    print('broken-stroke verdict:', run_d['summary']['verdict'])
    matches = screening.match_known_defects(run_d['tables']['缺陷组件'], annotations[2], d3.shape[:2])
    print('broken-stroke match:', matches)
    evidence = screening.local_evidence(d3, envelope, run_d['tables']['缺陷组件'][0])
    print('evidence box:', evidence['box'], 'stats keys:', list(evidence['stats'].keys()))

    # Assertions for automated checking.
    low_fa = report['policies']['low_false_alarm']
    sensitive = report['policies']['high_sensitivity']
    assert sensitive['known_defect_detected'] == sensitive['known_defect_total'] == 3, 'sensitive policy must detect all three known defects'
    assert low_fa['known_defect_detected'] == low_fa['known_defect_total'] == 3, 'low-false-alarm policy must not trade away broken-stroke detection'
    assert low_fa['false_alarm_sheets'] <= 1, 'low-false-alarm policy should keep normal sheets clean'
    assert run['summary']['defect_review_components'] == 0, 'fresh normal sheet must not raise defect review'
    assert run_d['summary']['verdict'] == 'review_required'
    print('\nALL ENGINE CHECKS PASSED')


if __name__ == '__main__':
    main()
