"""Browser-level smoke test for the screening UI (Playwright + headless Chromium)."""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    data_dir = tempfile.mkdtemp(prefix='printlab_ui_')
    env = dict(__import__('os').environ)
    env.update({'PRINTLAB_DATA_DIR': data_dir, 'PORT': '18937',
                'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                'PYTHONPATH': '/opt/pointwise-env/site-packages'})
    server = subprocess.Popen([sys.executable, '/workspace/app.py'], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        time.sleep(4)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True,
                                         executable_path='/opt/pointwise-env/browsers/chromium_headless_shell-1255/chrome-headless-shell-linux64/chrome-headless-shell',
                                         args=['--no-sandbox'])
            page = browser.new_page(viewport={'width': 1500, 'height': 980})
            errors = []
            page.on('console', lambda msg: errors.append(msg.text) if msg.type == 'error' else None)
            page.on('pageerror', lambda exc: errors.append(str(exc)))
            page.goto('http://127.0.0.1:18937/', wait_until='networkidle')

            # Create a job, then open the screening tab and generate the demo set.
            page.click('#newjob')
            page.fill('#jobname', 'UI筛查作业')
            page.click('#jobform .primary')
            page.wait_for_function("document.querySelector('#jobs').value !== ''")
            page.click('[data-tab="screening"]')
            try:
                page.wait_for_selector('#sc-demo', timeout=10000)
            except Exception:
                page.screenshot(path='/tmp/screening_fail.png', full_page=True)
                print('CONSOLE ERRORS:', errors)
                print('PANEL HTML SNIPPET:', page.locator('main').inner_html()[:800])
                raise
            page.click('#sc-demo')
            # Demo generation opens the new plan directly in the editor.
            page.wait_for_selector('#sc-validate', timeout=60000)

            # Editor should show 12 grouped sheets (8 normal + 3 defect + 1 candidate) and
            # 15 annotations: 3 known defects plus the variable-content exclude on all 12 sheets.
            role_selects = page.locator('[data-role]')
            assert role_selects.count() == 12
            ann_rows = page.locator('#sc-ann-list tbody tr')
            print('annotation rows in editor:', ann_rows.count())
            assert ann_rows.count() == 15

            # Run whole-sheet validation from the UI.
            page.click('#sc-validate')
            page.wait_for_selector('.sc-policy', timeout=120000)
            policies = page.locator('.sc-policy')
            assert policies.count() == 2
            # Both confirm buttons disabled because 8 normal sheets cannot meet targets.
            confirm_buttons = page.locator('.sc-confirm')
            disabled = [confirm_buttons.nth(i).is_disabled() for i in range(2)]
            print('confirm buttons disabled (insufficient samples):', disabled)
            assert all(disabled)
            # Recall text present for both policies.
            body_text = page.locator('.screening-panel').inner_text()
            assert '按已知缺陷检出' in body_text and '无独立验证' in body_text
            assert '样本不足' in body_text

            # Go to inspection, run a defect sheet and open component evidence.
            page.click('#sc-inspect')
            page.wait_for_selector('#run-go')
            page.locator('#run-asset').select_option(label='缺陷印张 D01')
            # Unconfirmed plan: enable sensitive policy for the trial run.
            page.locator('#run-policy').select_option('high_sensitivity')
            page.click('#run-go')
            page.wait_for_selector('.sc-region-row', timeout=120000)
            verdict = page.locator('.sc-verdict-red, .sc-verdict-amber').first.inner_text()
            print('defect sheet verdict shown:', verdict[:40])
            region_count = page.locator('.sc-region-row').count()
            print('region rows:', region_count)
            assert region_count >= 1
            page.locator('.sc-region-row').first.click()
            page.wait_for_selector('#dialog .sc-evidence-grid img', timeout=60000)
            evidence_images = page.locator('#dialog .sc-evidence-grid img')
            assert evidence_images.count() == 4
            # Wait until every evidence image has actually decoded.
            page.wait_for_function(
                "Array.from(document.querySelectorAll('#dialog .sc-evidence-grid img')).every(i => i.naturalWidth > 0)",
                timeout=30000)
            loaded = page.eval_on_selector_all('#dialog .sc-evidence-grid img',
                                               'imgs => imgs.map(i => i.naturalWidth)')
            print('evidence image natural widths:', loaded)
            assert all(w > 0 for w in loaded)
            page.keyboard.press('Escape')

            # History view records the run; legacy compare operation still in catalog.
            page.click('#sc-runs')
            page.wait_for_selector('#sc-runs-table')
            history_rows = page.locator('#sc-runs-table tbody tr')
            print('history rows:', history_rows.count())
            assert history_rows.count() == 1

            # Switch back to assets tab: screening panel hidden, normal workspace restored.
            page.click('[data-tab="assets"]')
            assert not page.locator('#screening-panel').is_visible()
            assert page.locator('#viewer').is_visible()
            print('tab switch restores main workspace: OK')

            assert not errors, f'Browser console errors: {errors}'
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
    print('\nALL UI SMOKE CHECKS PASSED')


if __name__ == '__main__':
    main()
