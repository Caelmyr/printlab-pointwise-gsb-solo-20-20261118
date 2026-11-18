/* 实测扫描校准向导：选扫描 → 圈色块/填参考/标记质量 → 看整组留出证据 → 确认/应用。
   依赖 app.js 提供的全局 api/safe/toast/escapeHtml/dialog/state。 */
(function () {
  const cal = { id: null, record: null, step: 1, scanIndex: 0, cardId: 'main', draw: null, confirmedIdForApply: null };

  function $(id) { return document.getElementById(id); }
  function calApi(path, options = {}) { return api('/api' + path, options); }

  async function refreshRecord(id) {
    cal.record = await calApi('/calibrations/' + (id || cal.id));
    cal.id = cal.record.id;
    return cal.record;
  }

  function currentScan() { return cal.record.session.scans[cal.scanIndex]; }

  // ------------------------------------------------------------------
  // 步骤1：创建会话，选择同设置扫描
  // ------------------------------------------------------------------
  async function openWizard() {
    if (!state.job) return toast('请先新建作业');
    const assets = state.job.assets.filter(a => a.channels === 1 || a.channels === 3);
    if (!assets.length) return toast('作业内还没有可作为色卡扫描的 RGB 图像');
    $('calcontent').innerHTML = `
      <h2>实测扫描校准</h2>
      <p class="muted">选择同一曝光、白平衡与灯光设置下、多次扫描的色卡图像。色块参考颜色需在统一 RGB 数值工作空间（sRGB 线性化后拟合，不做 ICC 转换）。</p>
      <div class="cal-scroll" id="scanpick">
        ${assets.map(a => `<label class="cal-check"><input type="checkbox" value="${a.id}"> <span>${escapeHtml(a.name)}</span><small>${a.width}×${a.height}</small></label>`).join('')}
      </div>
      <div class="dialog-grid">
        <label>曝光设置<input id="setup-exposure" placeholder="如 1/60 f8"></label>
        <label>白平衡<input id="setup-wb" placeholder="如 5500K"></label>
        <label>灯光设置<input id="setup-light" placeholder="如 固定柔光箱45°"></label>
        <label>色块内缩比例<input id="setup-inset" type="number" step="any" value="0.15"></label>
      </div>
      <label>说明<textarea id="setup-note" rows="2" placeholder="同设置后续校样可复用本档案"></textarea></label>
      <div class="cal-actions"><button class="primary" id="cal-create">创建校准会话</button>
        <button id="cal-open-existing">打开已有校准档案</button></div>`;
    $('caldialog').showModal();
    $('cal-create').onclick = () => safe(createSession);
    $('cal-open-existing').onclick = () => safe(openExisting);
  }

  async function createSession() {
    const scan_asset_ids = [...$('scanpick').querySelectorAll('input:checked')].map(x => x.value);
    if (scan_asset_ids.length < 1) throw new Error('请至少选择一张扫描色卡');
    const record = await calApi(`/jobs/${state.job.id}/calibrations`, {
      method: 'POST',
      body: {
        name: '实测扫描校准 ' + new Date().toLocaleString(),
        scan_asset_ids,
        setup: {
          exposure: $('setup-exposure').value, white_balance: $('setup-wb').value,
          lighting: $('setup-light').value, note: $('setup-note').value,
          inset: Number($('setup-inset').value || 0.15),
        },
      },
    });
    cal.id = record.id; cal.record = record; cal.scanIndex = 0;
    await ensureCard();
    renderStep2();
  }

  async function openExisting() {
    const list = await calApi(`/jobs/${state.job.id}/calibrations`);
    if (!list.calibrations.length) return toast('当前作业还没有校准档案');
    $('calcontent').innerHTML = `<h2>已有校准档案</h2><div class="cal-scroll">${list.calibrations.map(r => `
      <div class="cal-record" data-id="${r.id}">
        <strong>${escapeHtml(r.name)}</strong>
        <small>${r.status === 'confirmed' ? '已确认 v' + r.revision : r.status === 'draft' ? '草稿 v' + r.revision : '已被取代 v' + r.revision}</small>
        <p class="muted">${r.session.scans.length} 张扫描 · ${r.session.scans.reduce((n, s) => n + s.cards.reduce((m, c) => m + c.patches.length, 0), 0)} 个色块</p>
      </div>`).join('')}</div>`;
    $('calcontent').querySelectorAll('.cal-record').forEach(el => el.onclick = () => safe(async () => {
      await refreshRecord(el.dataset.id);
      cal.scanIndex = 0;
      if (cal.record.status === 'draft') renderStep2(); else renderConfirmedView();
    }));
  }

  async function ensureCard() {
    const scan = currentScan();
    if (!scan.cards.length) {
      await calApi(`/calibrations/${cal.id}/cards`, { method: 'POST', body: { scan_id: scan.scan_id, card_id: cal.cardId } });
      await refreshRecord();
    }
  }

  // ------------------------------------------------------------------
  // 步骤2：圈定色块、填写参考、标记饱和/脏污/排除
  // ------------------------------------------------------------------
  async function renderStep2() {
    await refreshRecord();
    const scan = currentScan();
    const totalPatches = cal.record.session.scans.reduce((n, s) => n + s.cards.reduce((m, c) => m + c.patches.length, 0), 0);
    const included = cal.record.session.scans.reduce((n, s) => n + s.cards.reduce((m, c) => m + c.patches.filter(p => p.included).length, 0), 0);
    $('calcontent').innerHTML = `
      <h2>圈定色块与参考颜色 <span class="cal-step">第 2 步 / 共 4 步</span></h2>
      <div class="cal-toolbar">
        <select id="cal-scan">${cal.record.session.scans.map((s, i) => `<option value="${i}" ${i === cal.scanIndex ? 'selected' : ''}>${escapeHtml(s.name)}</option>`).join('')}</select>
        <button id="cal-csv">导入CSV</button>
        <span class="muted">色块 ${totalPatches} · 纳入 ${included}</span>
      </div>
      <div class="cal-work">
        <div class="cal-canvas-wrap"><img id="cal-img" src="/api/assets/${scan.scan_id}/preview" alt="扫描图"></div>
        <div class="cal-side">
          <div class="cal-form">
            <label>色块名称<input id="patch-name" placeholder="如 R05-C"></label>
            <div class="dialog-grid">
              <label>x<input id="patch-x" type="number"></label><label>y<input id="patch-y" type="number"></label>
              <label>宽<input id="patch-w" type="number" value="12"></label><label>高<input id="patch-h" type="number" value="12"></label>
            </div>
            <label>参考颜色 RGB（0–1，统一工作空间）
              <div class="dialog-grid">
                <input id="ref-r" type="number" step="any" min="0" max="1" placeholder="R">
                <input id="ref-g" type="number" step="any" min="0" max="1" placeholder="G">
                <input id="ref-b" type="number" step="any" min="0" max="1" placeholder="B">
              </div>
            </label>
            <label>读数质量
              <select id="patch-quality">
                <option value="ok">纳入（正常）</option>
                <option value="saturated">饱和/裁切（纳入但标记）</option>
                <option value="dirty">脏污/不均（纳入但标记）</option>
                <option value="excluded">排除，不参与拟合</option>
              </select>
            </label>
            <button class="primary wide" id="patch-add">保存色块（也可在图上拖框）</button>
          </div>
          <div class="cal-patches cal-scroll" id="patch-list"></div>
        </div>
      </div>
      <div class="cal-actions">
        <button id="cal-back1">返回</button>
        <button class="primary" id="cal-evidence">构建模型比较证据 →</button>
      </div>`;
    $('cal-scan').onchange = () => safe(async () => { cal.scanIndex = Number($('cal-scan').value); await ensureCard(); renderStep2(); });
    $('cal-csv').onclick = () => safe(csvImport);
    $('cal-back1').onclick = () => safe(openWizard);
    $('cal-evidence').onclick = () => safe(buildEvidence);
    $('patch-add').onclick = () => safe(addPatchFromForm);
    setupCanvasDraw();
    renderPatchList();
  }

  function renderPatchList() {
    const scan = currentScan();
    const box = $('patch-list');
    const patches = scan.cards.flatMap(c => c.patches);
    box.innerHTML = patches.length ? patches.map(p => `
      <div class="patch-row q-${p.quality}">
        <strong>${escapeHtml(p.name)}</strong>
        <small>${p.quality}${p.auto_quality && p.auto_quality !== p.quality ? '（自动判为 ' + p.auto_quality + '）' : ''}</small>
        <small>测 ${p.measured_srgb.map(v => v.toFixed(2)).join('/')} · 参 ${p.reference_rgb.map(v => v.toFixed(2)).join('/')}</small>
        <div class="patch-btns">
          <button data-edit="${escapeHtml(p.name)}">编辑</button>
          <button data-del="${escapeHtml(p.name)}">删除</button>
        </div>
      </div>`).join('') : '<p class="muted">在左侧图上拖出矩形，或用表单填写后保存。</p>';
    box.querySelectorAll('[data-edit]').forEach(b => b.onclick = () => safe(() => editPatch(b.dataset.edit)));
    box.querySelectorAll('[data-del]').forEach(b => b.onclick = () => safe(async () => {
      const card = currentScan().cards[0];
      await calApi(`/calibrations/${cal.id}/patches/remove`, {
        method: 'POST', body: { scan_id: currentScan().scan_id, card_id: card.card_id, name: b.dataset.del },
      });
      renderStep2();
    }));
  }

  function setupCanvasDraw() {
    const img = $('cal-img');
    img.onload = () => { cal.draw = null; };
    let start = null;
    const toPixel = e => {
      const rect = img.getBoundingClientRect();
      const scan = currentScan();
      return {
        x: Math.min(scan.width - 1, Math.max(0, Math.floor((e.clientX - rect.left) / rect.width * scan.width))),
        y: Math.min(scan.height - 1, Math.max(0, Math.floor((e.clientY - rect.top) / rect.height * scan.height))),
      };
    };
    img.onmousedown = e => { start = toPixel(e); };
    img.onmouseup = e => {
      if (!start) return;
      const end = toPixel(e);
      const x = Math.min(start.x, end.x), y = Math.min(start.y, end.y);
      const w = Math.abs(end.x - start.x) || 8, h = Math.abs(end.y - start.y) || 8;
      $('patch-x').value = x; $('patch-y').value = y; $('patch-w').value = w; $('patch-h').value = h;
      start = null;
    };
  }

  async function addPatchFromForm() {
    const scan = currentScan();
    const geometry = {
      type: 'rectangle', x: Number($('patch-x').value), y: Number($('patch-y').value),
      width: Number($('patch-w').value), height: Number($('patch-h').value),
    };
    const name = $('patch-name').value.trim() || `P${Date.now() % 100000}`;
    await calApi(`/calibrations/${cal.id}/patches`, {
      method: 'POST',
      body: {
        scan_id: scan.scan_id, card_id: scan.cards[0].card_id, name, geometry,
        reference_rgb: [Number($('ref-r').value), Number($('ref-g').value), Number($('ref-b').value)],
        quality: $('patch-quality').value,
      },
    });
    toast('色块已保存并采样');
    renderStep2();
  }

  async function editPatch(name) {
    const scan = currentScan();
    const patch = scan.cards.flatMap(c => c.patches).find(p => p.name === name);
    $('patch-name').value = patch.name;
    $('patch-x').value = patch.geometry.x; $('patch-y').value = patch.geometry.y;
    $('patch-w').value = patch.geometry.width; $('patch-h').value = patch.geometry.height;
    $('ref-r').value = patch.reference_rgb[0]; $('ref-g').value = patch.reference_rgb[1]; $('ref-b').value = patch.reference_rgb[2];
    $('patch-quality').value = patch.quality;
    // 复用表单：先删后增的语义由 update 接口承担
    $('patch-add').textContent = '保存修改';
    $('patch-add').onclick = () => safe(async () => {
      await calApi(`/calibrations/${cal.id}/patches`, {
        method: 'PATCH',
        body: {
          scan_id: scan.scan_id, card_id: patch.card_id, name,
          reference_rgb: [Number($('ref-r').value), Number($('ref-g').value), Number($('ref-b').value)],
          quality: $('patch-quality').value,
          geometry: { type: 'rectangle', x: Number($('patch-x').value), y: Number($('patch-y').value),
                      width: Number($('patch-w').value), height: Number($('patch-h').value) },
        },
      });
      $('patch-add').textContent = '保存色块（也可在图上拖框）';
      $('patch-add').onclick = () => safe(addPatchFromForm);
      toast('参考/质量/几何已更新，读数重新采样');
      renderStep2();
    });
  }

  async function csvImport() {
    const scan = currentScan();
    dialog(`<h2>导入色块 CSV</h2>
      <p class="muted">列：name,x,y,width,height,ref_r,ref_g,ref_b，可选 card、quality（ok/saturated/dirty/excluded）。坐标为该扫描的原始像素。</p>
      <textarea id="csvbox" rows="10" style="width:100%;font-family:monospace;font-size:12px"
        placeholder="name,x,y,width,height,ref_r,ref_g,ref_b,card,quality&#10;R01,10,10,12,12,0.95,0.95,0.93,main,ok"></textarea>
      <button class="primary" id="csv-go">导入</button>`);
    $('csv-go').onclick = () => safe(async () => {
      const result = await calApi(`/calibrations/${cal.id}/patches/csv`, {
        method: 'POST', body: { scan_id: scan.scan_id, csv_text: $('csvbox').value },
      });
      $('dialog').close();
      toast('已导入 ' + result.added + ' 个色块');
      renderStep2();
    });
  }

  // ------------------------------------------------------------------
  // 步骤3：两套方案的整组留出证据比较
  // ------------------------------------------------------------------
  async function buildEvidence() {
    const included = cal.record.session.scans.reduce((n, s) => n + s.cards.reduce((m, c) => m + c.patches.filter(p => p.included).length, 0), 0);
    if (included < 6) throw new Error('纳入色块过少，至少需要 6 个并覆盖两个以上（扫描,色卡）组');
    toast('正在拟合两套模型并做整组留出验证…');
    const updated = await calApi(`/calibrations/${cal.id}/evidence`, { method: 'POST', body: {} });
    cal.record = updated;
    renderStep3();
  }

  function metricBlock(entry) {
    const sh = entry.scan_holdout, ch = entry.card_holdout;
    const foldRows = (h) => h ? h.folds.map(f => `<tr><td>${escapeHtml(f.holdout)}</td><td>${f.holdout_patches}</td>
      <td>${f.mean_delta_e76.toFixed(2)}</td><td>${f.p95_delta_e76.toFixed(2)}</td><td>${f.rgb_rmse.toFixed(4)}</td></tr>`).join('') : '';
    return `
      <div class="metric-block">
        <h4>训练拟合（不作为推荐依据）</h4>
        <p class="muted">平均 ΔE76 ${entry.training_fit.mean_delta_e76.toFixed(2)} · RGB RMSE ${entry.training_fit.rgb_rmse.toFixed(4)}</p>
        <h4>整次扫描留出</h4>
        ${sh ? `<table><thead><tr><th>留出扫描</th><th>色块</th><th>平均ΔE</th><th>p95 ΔE</th><th>RGB RMSE</th></tr></thead><tbody>${foldRows(sh)}</tbody></table>
        <p class="muted">汇总：平均 ${sh.summary.pooled_mean_delta_e76.toFixed(2)} · p95 ${sh.summary.pooled_p95_delta_e76.toFixed(2)} · 最大 ${sh.summary.pooled_max_delta_e76.toFixed(2)}</p>` : '<p class="cal-warn">扫描组数不足，无法做整扫描留出</p>'}
        <h4>整块色卡留出</h4>
        ${ch ? `<table><thead><tr><th>留出色卡</th><th>色块</th><th>平均ΔE</th><th>p95 ΔE</th><th>RGB RMSE</th></tr></thead><tbody>${foldRows(ch)}</tbody></table>
        <p class="muted">汇总：平均 ${ch.summary.pooled_mean_delta_e76.toFixed(2)} · p95 ${ch.summary.pooled_p95_delta_e76.toFixed(2)}</p>` : '<p class="cal-warn">色卡组数不足，无法做整卡留出</p>'}
        ${strataBlock(sh || ch)}
      </div>`;
  }

  function strataBlock(holdout) {
    if (!holdout) return '';
    const rows = holdout.folds[0].strata.map(s => `<tr class="${s.status === 'unjudged' ? 'cal-unjudged' : ''}">
      <td>${s.dimension === 'luminance' ? '亮度' : '位置'} · ${escapeHtml(s.stratum)}</td>
      <td>${s.status === 'unjudged' ? '无独立参考' : s.holdout_patches}</td>
      <td>${s.mean_delta_e76 === null ? '不能判断' : s.mean_delta_e76.toFixed(2)}</td>
      <td>${s.p95_delta_e76 === null ? '外推' : s.p95_delta_e76.toFixed(2)}</td></tr>`).join('');
    return `<h4>分亮度 / 分位置可靠范围（仅独立留出）</h4>
      <table><thead><tr><th>区间</th><th>留出色块</th><th>平均ΔE</th><th>p95</th></tr></thead><tbody>${rows}</tbody></table>`;
  }

  function renderStep3() {
    const ev = cal.record.evidence;
    if (!ev || ev.status === 'no_readings' || !ev.simple) return renderStep2();
    const rec = ev.recommendation;
    $('calcontent').innerHTML = `
      <h2>模型比较与可靠范围 <span class="cal-step">第 3 步 / 共 4 步</span></h2>
      <div class="cal-reco ${rec.recommended}">
        <strong>系统保守推荐：${rec.recommended === 'simple' ? '较简单方案（逐通道多项式响应）' : '灵活方案（响应 + 平滑照明场）'}</strong>
        <ul>${rec.reasons.map(r => `<li>${escapeHtml(r)}</li>`).join('')}</ul>
        <p class="muted">${escapeHtml(rec.rule)}</p>
      </div>
      <div class="cal-gaps"><strong>数据覆盖缺口</strong><ul>${ev.coverage_gaps.map(g => `<li>${escapeHtml(g.dimension)}：${escapeHtml(g.detail)}</li>`).join('')}</ul></div>
      <div class="cal-compare">
        <div class="cal-candidate"><h3>① 较简单方案</h3>${metricBlock(ev.simple)}</div>
        <div class="cal-candidate"><h3>② 灵活方案（含照明场）</h3>${
          ev.spatial ? metricBlock(ev.spatial)
            : `<p class="cal-warn">数据不足以拟合含照明场的灵活方案：${escapeHtml(ev.spatial_unavailable || '')}。请补充覆盖多个画面位置的色块（至少8个）后再比较；当前只能选择较简单方案。</p>`
        }</div>
      </div>
      <p class="muted">所有误差来自未参与拟合的整次扫描/整块色卡；同一色块的全部像素随组整体进入一侧，不做像素级拆分。标记 unjudged 的区间没有独立参考，应用时只会给外推标记，不会盖整图可靠章。</p>
      <div class="cal-actions">
        <button id="cal-back2">返回改色块</button>
        <button class="${rec.recommended === 'simple' ? 'primary' : ''}" id="confirm-simple">确认使用较简单方案</button>
        <button class="${rec.recommended === 'spatial' ? 'primary' : ''}" id="confirm-spatial" ${ev.spatial ? '' : 'disabled'}>确认使用灵活方案</button>
      </div>`;
    $('cal-back2').onclick = () => safe(renderStep2);
    $('confirm-simple').onclick = () => safe(() => confirmModel('simple'));
    $('confirm-spatial').onclick = () => safe(() => confirmModel('spatial'));
  }

  async function confirmModel(kind) {
    const note = prompt('确认使用「' + (kind === 'simple' ? '较简单' : '灵活（含照明场）') + '」方案。可填写选择说明：', '按独立整组留出证据选择') || '';
    const confirmed = await calApi(`/calibrations/${cal.id}/confirm`, { method: 'POST', body: { model_kind: kind, note } });
    cal.record = confirmed;
    toast('校准档案已确认并冻结，可用于同设置后续校样');
    renderConfirmedView();
  }

  // ------------------------------------------------------------------
  // 步骤4：确认档案、修订与应用入口
  // ------------------------------------------------------------------
  function renderConfirmedView() {
    const r = cal.record;
    const m = r.confirmed_model;
    const d = r.decision || {};
    $('calcontent').innerHTML = `
      <h2>校准档案 · ${escapeHtml(r.name)} <span class="cal-step">第 4 步 / 共 4 步</span></h2>
      <div class="cal-reco ${m.kind}">
        <strong>状态：${r.status === 'confirmed' ? '已确认（当前有效）' : '已被更新档案取代'} · 版本 v${r.revision}</strong>
        <p>模型：${m.kind === 'simple' ? '逐通道多项式响应（无空间项）' : '响应多项式 + 平滑乘法照明场'} · 工作空间 ${escapeHtml(m.workspace)}</p>
        <p class="muted">确认时间 ${escapeHtml(m.confirmed_at || '')} · 选择：${escapeHtml(d.chosen || '')} · ${escapeHtml(d.user_note || '')}</p>
      </div>
      <h3>应用范围与可靠规则</h3>
      <ul class="cal-scope">
        <li>仅适用于同曝光、白平衡与灯光设置的后续校样；设置变化需重新校准。</li>
        <li>应用后输出「可靠像素掩码」及位置/亮度/颜色三类外推掩码；默认质量测量只把 supported 像素当可靠读数。</li>
        <li>缺少暗部、边缘或颜色独立参考的范围只做外推并显式标记，不会给整幅图统一可靠标记。</li>
        <li>照明场是低阶平滑乘法场，只吸收缓慢灯光变化；纸张真实深浅纹理予以保留。</li>
      </ul>
      <div class="cal-actions">
        <button id="cal-revise">基于此档案创建修订版</button>
        <button class="primary" id="cal-use">在「扫描整理」中应用该校准</button>
      </div>`;
    $('cal-revise').onclick = () => safe(async () => {
      const rev = await calApi(`/calibrations/${cal.id}/revise`, { method: 'POST', body: {} });
      cal.record = rev; cal.id = rev.id; cal.scanIndex = 0;
      toast('已创建草稿修订版；原确认档案与旧结果保持不变');
      renderStep2();
    });
    $('cal-use').onclick = () => {
      cal.confirmedIdForApply = cal.record.id;
      $('caldialog').close();
      toast('已记住校准档案；在右侧「扫描整理 → 应用实测扫描校准」运行，参数已填好');
      prefillApplyOperation();
    };
  }

  function prefillApplyOperation() {
    const category = $('category');
    category.value = '扫描整理';
    setCategory();
    const op = $('operation');
    op.value = 'scan_calibration';
    renderParameters();
    // SPECIAL 操作没有动态参数，用高级 JSON 注入 calibration_id
    $('extra').value = JSON.stringify({ calibration_id: cal.confirmedIdForApply, strength: 1.0 });
  }

  document.addEventListener('DOMContentLoaded', () => {
    const button = $('calibrate');
    if (button) button.onclick = () => safe(openWizard);
  });
})();
