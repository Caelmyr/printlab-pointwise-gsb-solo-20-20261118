"""End-to-end API workflow test using the Flask test client and a temp store."""

import os
import sys
import tempfile

sys.path.insert(0, '/workspace')

from printlab.server import create_app
from printlab.storage import Store
from printlab import screening_workflow


def main():
    tmp = tempfile.mkdtemp(prefix='printlab_screen_')
    app = create_app(data_root=tmp)
    client = app.test_client()

    # Health and catalog still work (legacy compare remains registered).
    catalog = client.get('/api/catalog').get_json()
    assert any(op['id'] == 'compare' for op in catalog['operations']), 'legacy compare must remain'
    print('legacy compare still present:', any(op['id'] == 'compare' for op in catalog['operations']))

    job = client.post('/api/jobs', json={'name': '筛查E2E', 'dpi': 300}).get_json()
    job_id = job['id']

    # 1. Generate the synthetic screening demo set (groups + annotations + draft plan).
    demo = client.post(f'/api/jobs/{job_id}/screening/demo-set',
                       json={'normal_count': 7, 'defect_count': 3}).get_json()
    assert 'error' not in demo, demo
    plan_id = demo['plan']['id']
    print('demo plan:', plan_id, 'annotations:', demo['annotation_count'], 'groups:',
          {k: len(v) for k, v in demo['groups'].items()})

    # 2. Validation: whole-sheet hold-out.
    validation = client.post(f'/api/screening/plans/{plan_id}/validate', json={}).get_json()
    assert 'error' not in validation, validation
    val = validation['validation']
    for policy, record in val['policies'].items():
        print(policy, 'FA sheets', record['false_alarm_sheets'], '/', val['normal_count'],
              'recall', record['known_defect_detected'], '/', record['known_defect_total'],
              'sufficient', record['sample_sufficiency']['sufficient'])
    assert val['normal_count'] == 7 and val['defect_count'] == 3

    # 3. Confirming with insufficient samples must be rejected for the 5% target.
    blocked = client.post(f'/api/screening/plans/{plan_id}/confirm',
                          json={'policy_name': 'low_false_alarm'}).get_json()
    print('confirm low_false_alarm blocked as expected:', 'error' in blocked, blocked.get('error', '')[:40])
    assert 'error' in blocked

    # high_sensitivity target 10% needs 30 sheets; also insufficient -> blocked as well.
    blocked2 = client.post(f'/api/screening/plans/{plan_id}/confirm',
                           json={'policy_name': 'high_sensitivity'}).get_json()
    assert 'error' in blocked2

    # 4. Build a larger normal group so the 10% target becomes confirmable.
    big = client.post(f'/api/jobs/{job_id}/screening/demo-set',
                      json={'name': '大样本方案', 'normal_count': 31, 'defect_count': 2}).get_json()
    big_id = big['plan']['id']
    big_val = client.post(f'/api/screening/plans/{big_id}/validate', json={}).get_json()['validation']
    hs = big_val['policies']['high_sensitivity']
    print('31-sheet high_sensitivity sufficient:', hs['sample_sufficiency']['sufficient'],
                  'upper95:', round(hs['false_alarm_rate_upper95'], 3))
    assert hs['sample_sufficiency']['sufficient']
    confirmed = client.post(f'/api/screening/plans/{big_id}/confirm',
                            json={'policy_name': 'high_sensitivity'}).get_json()
    assert confirmed['payload']['status'] == 'confirmed'
    print('confirmed policy:', confirmed['payload']['confirmed_policy'])

    # 5. Inspect the candidate sheet with the frozen confirmed plan.
    candidate_id = confirmed['payload']['groups']['candidate'][0]
    run_resp = client.post(f'/api/screening/plans/{big_id}/inspect/{candidate_id}', json={}).get_json()
    assert 'error' not in run_resp, run_resp
    run = run_resp['run']['payload']
    print('candidate verdict:', run['verdict'], 'integrity:', run['plan_material_unchanged'],
          'evidence images:', len(run_resp['images']))
    assert run['plan_material_unchanged'] is True
    assert len(run_resp['images']) >= 4

    # 6. Evidence endpoints: metadata JSON + separate PNG images.
    ev = client.get(f'/api/screening/plans/{big_id}/evidence/{candidate_id}?region=1&padding=10').get_json()
    if 'error' in ev:
        print('evidence (no component):', ev['error'][:50])
    else:
        print('evidence box:', ev['evidence']['box'], 'keys:', ev['evidence']['image_keys'])
        key = ev['evidence']['image_keys'][0]
        png = client.get(f'/api/screening/plans/{big_id}/evidence/{candidate_id}/image?region=1&padding=10&key={key}')
        assert png.status_code == 200 and png.data[:8] == b'\x89PNG\r\n\x1a\n'
        print('evidence PNG bytes:', len(png.data))

    # 7. Inspect a defect sheet and confirm known-defect detection.
    defect_id = confirmed['payload']['groups']['defect'][0]
    drun = client.post(f'/api/screening/plans/{big_id}/inspect/{defect_id}', json={}).get_json()
    print('defect sheet verdict:', drun['run']['payload']['verdict'])
    assert drun['run']['payload']['verdict'] == 'review_required'

    # 8. Frozen plan: clone with more samples; old plan becomes superseded and old runs untouched.
    clone = client.post(f'/api/screening/plans/{big_id}/clone', json={'name': '接续版本'}).get_json()
    assert clone['payload']['status'] == 'draft'
    old = client.get(f'/api/screening/plans/{big_id}').get_json()['payload']
    print('old plan status after clone:', old['status'])
    assert old['status'] == 'superseded'
    runs = client.get(f'/api/screening/plans/{big_id}/runs').get_json()['runs']
    print('historical runs preserved:', len(runs))
    assert len(runs) == 2

    # 9. CSV export of region table.
    csv_resp = client.get(f"/api/screening/runs/{run['id']}/export?format=csv")
    assert csv_resp.status_code == 200 and b'area_px' in csv_resp.data
    print('CSV export bytes:', len(csv_resp.data))

    # 10. Reopen persistence: fresh app on same data root keeps plans/runs.
    app2 = create_app(data_root=tmp)
    client2 = app2.test_client()
    reopened = client2.get(f'/api/screening/plans/{big_id}').get_json()['payload']
    assert reopened['frozen_digest'] == old['frozen_digest']
    envelope = Store(tmp).load_screening_envelope(big_id)
    print('reopened envelope shapes:', envelope['median'].shape, envelope['scale'].shape)
    print('\nALL E2E API CHECKS PASSED')


if __name__ == '__main__':
    main()
