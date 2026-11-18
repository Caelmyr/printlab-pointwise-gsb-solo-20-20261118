// 整印张统计筛查方案界面：分组、标注、留一验证、取向比较、确认与证据回看
const scr = {
  view: 'list', plans: [], plan: null, run: null, runImages: [],
  groups: { normal: [], defect: [], candidate: [] },
  annotations: [], noise_floor: 0.02, min_support: 3,
};

const ROLE_LABEL = { normal: '正常', defect: '缺陷', candidate: '待检' };
const CLASS_LABEL = {
  defect_review: ['需复核缺陷', 'sc-badge-red'],
  insufficient_evidence: ['证据不足', 'sc-badge-amber'],
  normal: ['正常', 'sc-badge-green'],
};
const VERDICT_LABEL = {
  review_required: '需复核（发现缺陷组件或整张超出验证范围）',
  evidence_gap: '证据不足（存在弱差异/未覆盖像素，需人工复检）',
  normal: '正常（未见超出已验证包络的差异）',
};
const KIND_LABEL = { missing_print: '漏印', spot: '污点', broken_stroke: '断笔', other: '其他' };

async function renderScreening() {
  if (!window.state.job) { toast('请先选择作业'); return; }
  const main = document.querySelector('main');
  let panel = $('screening-panel');
  if (!panel) {
    panel = document.createElement('section');
    panel.className = 'workspace screening-panel';
    panel.id = 'screening-panel';
    main.insertBefore(panel, document.querySelector('.controls'));
  }
  panel.style.display = '';
  panel.classList.add('active');
  if (scr.view === 'list') await renderPlanList(panel);
  else if (scr.view === 'editor') renderEditor(panel);
  else if (scr.view === 'validation') await renderValidation(panel);
  else if (scr.view === 'inspect') renderInspect(panel);
  else if (scr.view === 'runs') await renderRuns(panel);
}

async function loadPlans() {
  const result = await api(`/api/jobs/${state.job.id}/screening/plans`);
  scr.plans = result.plans.map(row => row.payload);
}

async function renderPlanList(panel) {
  await loadPlans();
  const statusLabel = { draft: '草稿', validated: '已验证', confirmed: '已确认', superseded: '已接续' };
  panel.innerHTML = `
    <div class="workspace-top"><div><span class="eyebrow">SHEET SCREENING</span>
    <h1>整印张统计筛查方案</h1>
    <p class="muted">用同版同尺寸、已校正的正常印张建立逐像素稳健波动包络；整印张分组验证，不裁块混用；确认后冻结，补样本不改动旧报告。</p></div>
    <div class="toolbar">
      <button id="sc-demo">生成演示印张组</button>
      <button id="sc-new" class="primary">新建筛查方案</button>
    </div></div>
    <div class="sc-cards">${scr.plans.length ? scr.plans.map(p => `
      <div class="sc-card" data-plan="${escapeHtml(p.id)}">
        <div class="sc-card-head"><strong>${escapeHtml(p.name)}</strong><span class="sc-status sc-status-${p.status}">${statusLabel[p.status] || p.status}</span></div>
        <small>正常 ${p.groups.normal.length} 张 · 缺陷 ${p.groups.defect.length} 张 · 待检 ${p.groups.candidate.length} 张 · 标注 ${p.annotations.length} 条</small>
        <small class="muted">取向：${p.confirmed_policy ? escapeHtml(p.confirmed_policy) : '未确认'} · 更新 ${new Date(p.updated_at).toLocaleString()}</small>
        <div class="toolbar sc-card-actions"><button data-open="${escapeHtml(p.id)}">打开</button>
          ${p.status === 'confirmed' ? `<button data-clone="${escapeHtml(p.id)}">补样本/新版本</button>` : ''}</div>
      </div>`).join('') : '<p class="muted">还没有筛查方案。可先“生成演示印张组”体验完整流程，或用真实印张“新建筛查方案”。</p>'}</div>
    <div class="sc-notes"><strong>方法边界</strong><ul>
      <li>逐像素中位数 + MAD 尺度（含绝对噪声下限），不做整体模糊、不放大全局容差；局部相位容差仅在邻域已存在同类偏差时生效。</li>
      <li>验证单位是整张印张：正常印张逐张留一，缺陷印张用完整正常包络，按印张统计误报、按已知缺陷统计检出。</li>
      <li>像素阈值不等于整张合格判定；样本不足、类别无独立验证或新印张超出验证范围时一律标记待复检并说明原因。</li>
      <li>原“对应像素差异检查”单对图像操作保持不变，本方案是其上层可复用筛查能力。</li></ul></div>`;
  $('sc-new').onclick = () => safe(() => startEditor(null));
  $('sc-demo').onclick = () => safe(async () => {
    const result = await api(`/api/jobs/${state.job.id}/screening/demo-set`, { method: 'POST', body: { normal_count: 8, defect_count: 3 } });
    toast('演示印张组与草稿方案已生成');
    await openPlan(result.plan.id);
  });
  panel.querySelectorAll('[data-open]').forEach(el => el.onclick = () => safe(() => openPlan(el.dataset.open)));
  panel.querySelectorAll('[data-clone]').forEach(el => el.onclick = () => safe(async () => {
    const result = await api(`/api/screening/plans/${el.dataset.clone}/clone`, { method: 'POST', body: {} });
    toast('已创建接续草稿，旧方案与报告保持不变');
    await openPlan(result.id);
  }));
}

async function refreshJob() {
  if (window.state && window.state.job) {
    window.state.job = await api(`/api/jobs/${window.state.job.id}`);
  }
}

async function openPlan(planId) {
  await refreshJob();
  const row = await api(`/api/screening/plans/${planId}`);
  scr.plan = row.payload;
  scr.groups = { normal: [...scr.plan.groups.normal], defect: [...scr.plan.groups.defect], candidate: [...scr.plan.groups.candidate] };
  scr.annotations = scr.plan.annotations.map(a => ({ ...a }));
  scr.noise_floor = scr.plan.envelope_settings.noise_floor;
  scr.min_support = scr.plan.envelope_settings.min_support;
  scr.view = scr.plan.validation ? 'validation' : 'editor';
  await renderScreening();
}

function startEditor() {
  scr.plan = null;
  scr.groups = { normal: [], defect: [], candidate: [] };
  scr.annotations = [];
  state.job.assets.forEach(a => { /* 默认无分组，由用户选择 */ });
  scr.view = 'editor';
  renderScreening();
}

function assetRole(assetId) {
  for (const role of ['normal', 'defect', 'candidate']) if (scr.groups[role].includes(assetId)) return role;
  return '';
}

function renderEditor(panel) {
  const assets = state.job.assets;
  const frozen = scr.plan && scr.plan.status === 'confirmed';
  panel.innerHTML = `
    <div class="workspace-top"><div><span class="eyebrow">PLAN EDITOR</span>
    <h1>${scr.plan ? escapeHtml(scr.plan.name) : '新建筛查方案'}</h1>
    <p class="muted">把同版同尺寸印张分到正常/缺陷/待检三组；在缺陷或待检印张上标注已知缺陷与不检查的可变内容区。</p></div>
    <div class="toolbar"><button id="sc-back">返回列表</button>
      <button id="sc-save" class="primary" ${frozen ? 'disabled' : ''}>保存分组与标注</button>
      <button id="sc-validate" ${frozen ? 'disabled' : ''}>整印张验证</button></div></div>
    ${frozen ? '<p class="sc-warn">该方案已确认冻结。补充样本请用“新版本”复制；当前分组与标注只读。</p>' : ''}
    <div class="sc-grid">
      <section class="sc-box"><h3>印张分组（整印张为单位）</h3>
        <div class="table-scroll"><table><thead><tr><th>印张</th><th>尺寸</th><th>分组</th></tr></thead><tbody>
        ${assets.map(a => `<tr><td>${escapeHtml(a.name)}${a.metadata.synthetic ? ' <small class="muted">合成</small>' : ''}</td>
        <td>${a.width}×${a.height}</td><td><select data-role="${a.id}" ${frozen ? 'disabled' : ''}>
          <option value="">不使用</option>
          ${['normal', 'defect', 'candidate'].map(r => `<option value="${r}" ${assetRole(a.id) === r ? 'selected' : ''}>${ROLE_LABEL[r]}</option>`).join('')}
        </select></td></tr>`).join('')}
        </tbody></table></div>
        <small class="muted">同一张印张不能同时属于正常组与缺陷组；所有印张必须像素尺寸一致。</small>
      </section>
      <section class="sc-box"><h3>已知缺陷与排除区（${scr.annotations.length}）</h3>
        <div id="sc-ann-list">${renderAnnotationList()}</div>
        <h4>添加标注</h4>
        <div class="sc-form">
          <label>印张<select id="ann-asset">${annotatableAssets().map(id => assetOption(id)).join('')}</select></label>
          <label>用途<select id="ann-role"><option value="known_defect">已知缺陷</option><option value="exclude">不检查可变区</option></select></label>
          <label id="ann-kind-wrap">缺陷类型<select id="ann-kind">${Object.entries(KIND_LABEL).map(([k, v]) => `<option value="${k}">${v}</option>`).join('')}</select></label>
          <label>说明<input id="ann-label" placeholder="如：小字断笔 / 可变条码区"></label>
          <label class="checkbox" id="ann-all-wrap" style="display:none"><input type="checkbox" id="ann-all"> 同位置应用到全部分组印张（用于一致的可变内容排除区）</label>
          <div class="dialog-grid">
            <label>x<input id="ann-x" type="number" min="0"></label><label>y<input id="ann-y" type="number" min="0"></label>
            <label>width<input id="ann-w" type="number" min="1"></label><label>height<input id="ann-h" type="number" min="1"></label>
          </div>
          <button id="ann-add" ${frozen ? 'disabled' : ''}>添加矩形标注</button>
        </div>
        <h4>包络参数</h4>
        <div class="dialog-grid">
          <label>绝对噪声下限（0–1）<input id="env-noise" type="number" step="0.001" min="0.001" max="0.25" value="${scr.noise_floor}"></label>
          <label>最少支撑样本数<input id="env-support" type="number" step="1" min="2" max="40" value="${scr.min_support}"></label>
        </div>
      </section>
    </div>`;
  $('sc-back').onclick = () => { scr.view = 'list'; renderScreening(); };
  panel.querySelectorAll('[data-role]').forEach(sel => sel.onchange = () => {
    for (const role of ['normal', 'defect', 'candidate']) scr.groups[role] = scr.groups[role].filter(id => id !== sel.dataset.role);
    if (sel.value) scr.groups[sel.value].push(sel.dataset.role);
  });
  $('ann-role').onchange = () => {
    $('ann-kind-wrap').style.display = $('ann-role').value === 'known_defect' ? '' : 'none';
    $('ann-all-wrap').style.display = $('ann-role').value === 'exclude' ? '' : 'none';
  };
  $('ann-add').onclick = () => safe(addAnnotation);
  bindAnnotationDeletes();
  $('sc-save').onclick = () => safe(savePlan);
  $('sc-validate').onclick = () => safe(saveAndValidate);
}

function annotatableAssets() {
  return [...scr.groups.defect, ...scr.groups.candidate, ...scr.groups.normal];
}
function assetOption(id) {
  const a = state.job.assets.find(x => x.id === id);
  return a ? `<option value="${id}">${escapeHtml(a.name)}</option>` : '';
}
function assetName(id) {
  const a = state.job.assets.find(x => x.id === id);
  return a ? a.name : id;
}
function renderAnnotationList() {
  if (!scr.annotations.length) return '<p class="muted">暂无标注。可变印字/条码区请标为“不检查”，已知漏印/污点/断笔请标为“已知缺陷”。</p>';
  return `<div class="table-scroll"><table><thead><tr><th>印张</th><th>用途</th><th>类型/说明</th><th>矩形</th><th></th></tr></thead><tbody>
    ${scr.annotations.map((a, i) => `<tr><td>${escapeHtml(assetName(a.asset_id))}</td>
      <td>${a.role === 'exclude' ? '不检查' : '已知缺陷'}</td>
      <td>${a.role === 'known_defect' ? KIND_LABEL[a.kind] + ' · ' : ''}${escapeHtml(a.label)}</td>
      <td>${a.geometry.x},${a.geometry.y} ${a.geometry.width}×${a.geometry.height}</td>
      <td><button data-del-ann="${i}">删除</button></td></tr>`).join('')}</tbody></table></div>`;
}

function addAnnotation() {
  const asset_id = $('ann-asset').value;
  const role = $('ann-role').value;
  if (!asset_id) throw new Error('请先把印张加入缺陷/待检/正常组');
  const geometry = {
    type: 'rectangle',
    x: Number($('ann-x').value), y: Number($('ann-y').value),
    width: Number($('ann-w').value), height: Number($('ann-h').value),
  };
  if (![geometry.x, geometry.y, geometry.width, geometry.height].every(Number.isFinite) || geometry.width < 1 || geometry.height < 1) {
    throw new Error('请填写完整的正整数矩形坐标与尺寸');
  }
  const item = { asset_id, role, label: $('ann-label').value || (role === 'exclude' ? '排除区' : '已知缺陷'), geometry };
  if (role === 'known_defect') {
    if (!scr.groups.defect.includes(asset_id) && !scr.groups.candidate.includes(asset_id)) {
      throw new Error('已知缺陷只能标在缺陷组或待检组印张上');
    }
    item.kind = $('ann-kind').value;
  }
  const targets = (role === 'exclude' && $('ann-all').checked) ? annotatableAssets() : [asset_id];
  for (const targetId of targets) {
    const replica = { ...item, asset_id: targetId };
    if (!scr.annotations.some(a => a.asset_id === targetId && a.role === role &&
        JSON.stringify(a.geometry) === JSON.stringify(geometry))) {
      scr.annotations.push(replica);
    }
  }
  $('sc-ann-list').innerHTML = renderAnnotationList();
  bindAnnotationDeletes();
  toast(targets.length > 1 ? `标注已应用到 ${targets.length} 张印张，保存方案后生效` : '标注已加入，保存方案后生效');
}

function bindAnnotationDeletes() {
  document.querySelectorAll('[data-del-ann]').forEach(btn => btn.onclick = () => {
    scr.annotations.splice(Number(btn.dataset.delAnn), 1);
    $('sc-ann-list').innerHTML = renderAnnotationList();
    bindAnnotationDeletes();
  });
}

function planPayload() {
  return {
    name: scr.plan ? scr.plan.name : '可复用筛查方案',
    groups: scr.groups,
    annotations: scr.annotations,
    noise_floor: Number($('env-noise').value),
    min_support: Number($('env-support').value),
  };
}

async function savePlan() {
  const payload = planPayload();
  const row = scr.plan
    ? await api(`/api/screening/plans/${scr.plan.id}`, { method: 'PATCH', body: payload })
    : await api(`/api/jobs/${state.job.id}/screening/plans`, { method: 'POST', body: payload });
  scr.plan = row.payload;
  toast('方案已保存（草稿），验证前可继续调整');
}

async function saveAndValidate() {
  await savePlan();
  const result = await api(`/api/screening/plans/${scr.plan.id}/validate`, { method: 'POST', body: {} });
  scr.plan = result.plan.payload;
  scr.view = 'validation';
  renderScreening();
}

async function renderValidation(panel) {
  if (!scr.plan) { scr.view = 'list'; return renderScreening(); }
  if (!scr.plan.validation) { scr.view = 'editor'; return renderScreening(); }
  const v = scr.plan.validation;
  panel.innerHTML = `
    <div class="workspace-top"><div><span class="eyebrow">WHOLE-SHEET VALIDATION</span>
    <h1>整印张验证 · ${escapeHtml(scr.plan.name)}</h1>
    <p class="muted">${escapeHtml(v.holdout_protocol)}</p></div>
    <div class="toolbar"><button id="sc-back">返回列表</button><button id="sc-edit">编辑分组</button>
      <button id="sc-inspect">去检查印张</button></div></div>
    <div class="sc-box"><h3>验证基线（整张统计，与取向无关）</h3>
      <pre>${escapeHtml(JSON.stringify(v.baseline, null, 2))}</pre></div>
    <div class="sc-grid">${Object.entries(v.policies).map(([name, p]) => `
      <section class="sc-box sc-policy">
        <h3>${escapeHtml(p.label)} <small>${name}</small></h3>
        <table class="sc-kv"><tbody>
          <tr><th>按印张误报</th><td>${p.false_alarm_sheets}/${v.normal_count} 张触发 · 观测 ${(p.false_alarm_rate_observed * 100).toFixed(1)}% · 95%上限 <strong>${(p.false_alarm_rate_upper95 * 100).toFixed(1)}%</strong></td></tr>
          <tr><th>误报目标</th><td>单张 ≤ ${(p.target_sheet_false_alarm_rate * 100).toFixed(0)}% · ${p.sample_sufficiency.sufficient ? '<span class="sc-badge sc-badge-green">样本足够</span>' : '<span class="sc-badge sc-badge-amber">样本不足</span>'} <small>${escapeHtml(p.sample_sufficiency.note)}</small></td></tr>
          <tr><th>按已知缺陷检出</th><td>${p.known_defect_detected}/${p.known_defect_total} · 检出率 ${p.recall_observed === null ? '无标注' : (p.recall_observed * 100).toFixed(0) + '%'} · 95%下限 ${p.recall_lower95 === null ? '—' : (p.recall_lower95 * 100).toFixed(0) + '%'}</td></tr>
          <tr><th>分类别证据</th><td>${Object.entries(p.recall_by_kind).map(([kind, r]) => `${KIND_LABEL[kind] || kind} ${r.detected}/${r.known}${r.independent_validation ? '' : ' <span class="sc-badge sc-badge-amber">无独立验证</span>'}`).join('<br>') || '—'}</td></tr>
          <tr><th>判据参数</th><td><small>强z ${p.settings.z_strong} · 弱z ${p.settings.z_weak} · 最小面积 ${p.settings.min_area_px}px · 相位容差 ${p.settings.phase_allowance_px}px · 形态门 ${p.settings.morphology_gate ? '开' : '关'} · 组件平均z ${p.settings.min_component_mean_z}</small></td></tr>
        </tbody></table>
        <details><summary>每张正常印张误报明细 / 缺陷印张检出明细</summary>
          <div class="table-scroll"><table><thead><tr><th>正常印张</th><th>误报组件</th><th>误报像素</th><th>判定</th></tr></thead><tbody>
          ${p.normal_sheet_false_alarms.map(r => `<tr><td>N${r.sheet_index + 1}</td><td>${r.false_alarm_regions}</td><td>${r.false_alarm_pixels}</td><td>${r.verdict}</td></tr>`).join('')}
          </tbody></table>
          <table><thead><tr><th>缺陷印张</th><th>判定</th><th>缺陷组件</th><th>已知缺陷匹配</th></tr></thead><tbody>
          ${p.defect_sheet_runs.map(r => `<tr><td>D${r.sheet_index + 1}</td><td>${r.verdict}</td><td>${r.defect_components}</td>
            <td>${r.matches.map(m => `${escapeHtml(m.label)}：${m.detected ? '检出' : '漏检（待复检）'}`).join('<br>')}</td></tr>`).join('')}
          </tbody></table></div></details>
        <button class="primary sc-confirm" data-policy="${name}" ${!p.sample_sufficiency.sufficient ? 'disabled title="正常样本数量不足以支持该误报要求"' : ''}>确认采用此取向</button>
      </section>`).join('')}</div>
    <div class="sc-notes"><strong>限制与待复检规则</strong><ul>${v.limitations.map(l => `<li>${escapeHtml(l)}</li>`).join('')}</ul></div>`;
  $('sc-back').onclick = () => { scr.view = 'list'; renderScreening(); };
  $('sc-edit').onclick = () => { scr.view = 'editor'; renderScreening(); };
  $('sc-inspect').onclick = () => { scr.view = 'inspect'; renderScreening(); };
  panel.querySelectorAll('.sc-confirm').forEach(btn => btn.onclick = () => safe(async () => {
    const result = await api(`/api/screening/plans/${scr.plan.id}/confirm`, { method: 'POST', body: { policy_name: btn.dataset.policy } });
    scr.plan = result.payload;
    toast('方案已确认并冻结；后续印张将按该取向检查');
    renderScreening();
  }));
}

function renderInspect(panel) {
  if (!scr.plan) { scr.view = 'list'; return renderScreening(); }
  const inspectable = state.job.assets;
  const confirmed = scr.plan.status === 'confirmed';
  panel.innerHTML = `
    <div class="workspace-top"><div><span class="eyebrow">SHEET INSPECTION</span>
    <h1>印张检查 · ${escapeHtml(scr.plan.name)}</h1>
    <p class="muted">状态：${confirmed ? '已确认冻结（' + scr.plan.confirmed_policy + '）' : '未确认，可临时选择取向试跑，结果不计为确认方案'}</p></div>
    <div class="toolbar"><button id="sc-back">返回列表</button><button id="sc-runs">历史记录</button>${!confirmed ? '<button id="sc-val">查看验证</button>' : ''}</div></div>
    <div class="sc-box"><div class="sc-run-form">
      <label>受检印张<select id="run-asset">${inspectable.map(a => `<option value="${a.id}" ${scr.groups.candidate.includes(a.id) ? 'selected' : ''}>${escapeHtml(a.name)}</option>`).join('')}</select></label>
      <label>判定取向<select id="run-policy" ${confirmed ? 'disabled' : ''}>
        ${['low_false_alarm', 'high_sensitivity'].map(p => `<option value="${p}" ${scr.plan.confirmed_policy === p ? 'selected' : ''}>${p === 'low_false_alarm' ? '更少误报' : '更少漏检'}</option>`).join('')}
      </select></label>
      <button id="run-go" class="primary">运行整印张检查</button>
    </div>
    <div class="sc-legend"><span class="sc-badge sc-badge-red">需复核缺陷</span><span class="sc-badge sc-badge-amber">证据不足/待复检</span><span class="sc-badge sc-badge-green">正常区域</span><small>灰色=排除区，蓝色=正常样本未覆盖像素</small></div></div>
    <div id="sc-run-result"></div>`;
  $('sc-back').onclick = () => { scr.view = 'list'; renderScreening(); };
  $('sc-runs').onclick = () => { scr.view = 'runs'; renderScreening(); };
  if ($('sc-val')) $('sc-val').onclick = () => { scr.view = 'validation'; renderScreening(); };
  $('run-go').onclick = () => safe(runInspection);
}

async function runInspection() {
  const assetId = $('run-asset').value;
  const policyName = $('run-policy').value;
  const result = await api(`/api/screening/plans/${scr.plan.id}/inspect/${assetId}`, { method: 'POST', body: { policy_name: policyName } });
  scr.run = result.run.payload;
  scr.runImages = result.images;
  renderRunResult(assetId);
}

function renderRunResult(assetId) {
  const r = scr.run;
  const box = $('sc-run-result');
  const verdict = VERDICT_LABEL[r.verdict] || r.verdict;
  const verdictClass = r.verdict === 'normal' ? 'sc-verdict-green' : r.verdict === 'review_required' ? 'sc-verdict-red' : 'sc-verdict-amber';
  box.innerHTML = `
    <div class="sc-box">
      <h3>判定：<span class="${verdictClass}">${escapeHtml(verdict)}</span></h3>
      <small>方案完整性：${r.plan_material_unchanged ? '分组/标注/包络与确认时一致' : '方案材料已变化，结果待复检'} · 冻结记录：${r.frozen ? '是' : '否'}</small>
      <div class="sc-summary-grid">
        <div>有效像素 ${r.summary.valid_pixels}</div><div>排除像素 ${r.summary.excluded_pixels}</div>
        <div>证据不足像素 ${r.summary.unsupported_pixels}</div>
        <div>缺陷组件 ${r.summary.defect_review_components}</div><div>待复检组件 ${r.summary.insufficient_evidence_components}</div>
        <div>正常组件 ${r.summary.normal_components}</div>
      </div>
      ${r.summary.drift_reasons && r.summary.drift_reasons.length ? `<p class="sc-warn">超出验证范围：${r.summary.drift_reasons.map(escapeHtml).join('；')}</p>` : ''}
      <div class="sc-evidence-thumbs">${scr.runImages.map(img => `<figure><img src="${img.preview_url}" alt="${escapeHtml(img.name)}"><figcaption>${escapeHtml(img.name)}</figcaption></figure>`).join('')}</div>
      <div class="toolbar"><a class="button" href="/api/screening/runs/${r.id}/export?format=csv">导出组件位置/面积 CSV</a>
        <a class="button" href="/api/screening/runs/${r.id}/export?format=json">导出完整判定 JSON</a></div>
    </div>
    <div class="sc-box"><h3>组件明细（点行回看局部证据）</h3>
      <div class="table-scroll"><table><thead><tr><th>#</th><th>分类</th><th>位置 x,y</th><th>面积 px</th><th>平均/max z</th><th>极性</th><th>已知缺陷</th><th>判定依据</th></tr></thead><tbody>
      ${r.regions.map(row => `<tr class="sc-region-row" data-region="${row.region}" data-asset="${assetId}" style="cursor:pointer">
        <td>${row.region}</td><td><span class="sc-badge ${CLASS_LABEL[row.class][1]}">${CLASS_LABEL[row.class][0]}</span></td>
        <td>${row.x},${row.y} (${row.width}×${row.height})</td><td>${row.area_px}</td>
        <td>${row.mean_z.toFixed(1)} / ${row.max_z.toFixed(1)}</td><td>${row.polarity === 'lighter_than_normal' ? '偏浅' : '偏深'}</td>
        <td>${(row.known_defect_matches || []).map(m => escapeHtml(m.label)).join('，') || '—'}</td>
        <td><small>${escapeHtml(row.reason)}</small></td></tr>`).join('')}
      </tbody></table></div></div>`;
  box.querySelectorAll('.sc-region-row').forEach(tr => tr.onclick = () => safe(() => openEvidence(tr.dataset.asset, Number(tr.dataset.region))));
}

async function openEvidence(assetId, region) {
  const policyName = scr.run ? scr.run.policy_name : (scr.plan.confirmed_policy || 'low_false_alarm');
  const meta = await api(`/api/screening/plans/${scr.plan.id}/evidence/${assetId}?region=${region}&padding=12&policy_name=${encodeURIComponent(policyName)}`);
  const base = `/api/screening/plans/${scr.plan.id}/evidence/${assetId}/image?region=${region}&padding=12&policy_name=${encodeURIComponent(policyName)}&key=`;
  const keys = meta.evidence.image_keys;
  dialog(`<h2>组件 #${region} 局部证据</h2>
    <p class="muted">${escapeHtml(meta.component.reason)}</p>
    <div class="sc-evidence-grid">${keys.map(k => `<figure><img src="${base}${encodeURIComponent(k)}" alt="${escapeHtml(k)}"><figcaption>${escapeHtml(k)}</figcaption></figure>`).join('')}</div>
    <h3>判定依据与正常波动</h3>
    <div class="table-scroll"><table><tbody>
      <tr><th>组件分类</th><td>${CLASS_LABEL[meta.component.class][0]}</td><th>面积</th><td>${meta.component.area_px} px</td></tr>
      <tr><th>平均 z / max z</th><td>${meta.component.mean_z.toFixed(2)} / ${meta.component.max_z.toFixed(2)}</td><th>强/弱像素</th><td>${meta.component.strong_pixels} / ${meta.component.weak_pixels}</td></tr>
      <tr><th>局部候选均值 RGB</th><td>${fmtRgb(meta.evidence.stats.candidate_mean_rgb)}</td><th>正常中位均值 RGB</th><td>${fmtRgb(meta.evidence.stats.normal_mean_rgb)}</td></tr>
      <tr><th>局部最大 z</th><td>${meta.evidence.stats.local_max_z.toFixed(2)}</td><th>局部正常尺度中位</th><td>${meta.evidence.stats.local_median_scale.toFixed(4)}</td></tr>
      <tr><th>正常包络摘要</th><td colspan="3"><small>${escapeHtml(JSON.stringify(meta.normal_envelope_summary))}</small></td></tr>
    </tbody></table></div>`);
}

function fmtRgb(values) {
  return Array.isArray(values) ? values.map(v => Number(v).toFixed(3)).join(' / ') : '—';
}

async function renderRuns(panel) {
  const result = await api(`/api/screening/plans/${scr.plan.id}/runs`);
  const runs = result.runs.map(row => row.payload);
  panel.innerHTML = `
    <div class="workspace-top"><div><span class="eyebrow">FROZEN RUNS</span><h1>检查记录 · ${escapeHtml(scr.plan.name)}</h1>
    <p class="muted">记录与确认方案绑定；后来补样本产生新版本不会改动这些报告。</p></div>
    <div class="toolbar"><button id="sc-back">返回列表</button><button id="sc-inspect">继续检查</button></div></div>
    <div class="table-scroll"><table id="sc-runs-table"><thead><tr><th>时间</th><th>印张</th><th>取向</th><th>判定</th><th>方案一致</th><th>导出</th></tr></thead><tbody>
    ${runs.map(r => `<tr><td>${new Date(r.created_at).toLocaleString()}</td><td>${escapeHtml(r.asset_name)}</td>
      <td>${r.policy_name === 'low_false_alarm' ? '更少误报' : '更少漏检'}</td>
      <td><span class="sc-badge ${r.verdict === 'normal' ? 'sc-badge-green' : r.verdict === 'review_required' ? 'sc-badge-red' : 'sc-badge-amber'}">${VERDICT_LABEL[r.verdict] ? VERDICT_LABEL[r.verdict].split('（')[0] : r.verdict}</span></td>
      <td>${r.plan_material_unchanged ? '是' : '否（待复检）'}</td>
      <td><a href="/api/screening/runs/${r.id}/export?format=csv">CSV</a> · <a href="/api/screening/runs/${r.id}/export?format=json">JSON</a></td></tr>`).join('') || '<tr><td colspan="6" class="muted">还没有检查记录</td></tr>'}
    </tbody></table></div>`;
  $('sc-back').onclick = () => { scr.view = 'list'; renderScreening(); };
  $('sc-inspect').onclick = () => { scr.view = 'inspect'; renderScreening(); };
}

window.scr = scr;
