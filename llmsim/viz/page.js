const D = __DATA__;
const EP = D.ep;
const STAGE_COLOR = { dispatch: "var(--series-1)", combine: "var(--series-2)" };
const DIR_LABEL = { sent: "发出", recv: "收到" };
const RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
const state = { phase: 0, stage: "dispatch", dir: "sent", layer: 0 };
const dark = matchMedia("(prefers-color-scheme: dark)");
const tip = document.getElementById("tip");

const $ = id => document.getElementById(id);
const sum = a => a.reduce((x, y) => x + y, 0);
const max = a => Math.max(...a);
const phase = () => D.phases[state.phase];
const layer = () => phase().layers[state.layer];
const peak = st => max(st.card_time_us);
const permuteUs = st => st.steps.find(s => s.op === "permute").us;
const peakPermute = st => max(permuteUs(st));
const stepLabel = op => op === "a2a" ? "卡间传输" : state.stage === "dispatch" ? "permute" : "unpermute";

function fmtBytes(b) {
  if (b >= 1024 ** 3) return (b / 1024 ** 3).toFixed(2) + " GiB";
  if (b >= 1024 ** 2) return (b / 1024 ** 2).toFixed(2) + " MiB";
  return (b / 1024).toFixed(1) + " KiB";
}

function fmtTime(us) {
  return us >= 1000 ? (us / 1000).toFixed(2) + " ms" : us.toFixed(1) + " µs";
}

function niceMax(v) {
  const p = 10 ** Math.floor(Math.log10(v || 1));
  return [1, 2, 2.5, 5, 10].map(m => m * p).find(x => x >= v);
}

function rampColor(t) {
  const r = dark.matches ? [...RAMP].reverse() : RAMP;
  const x = Math.min(1, Math.max(0, t)) * (r.length - 1);
  const i = Math.min(r.length - 2, Math.floor(x));
  const rgb = h => [1, 3, 5].map(k => parseInt(h.slice(k, k + 2), 16));
  const a = rgb(r[i]), b = rgb(r[i + 1]);
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * (x - i))).join(",")})`;
}

function svg(w, h, body) {
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width: ${w}px">${body}</svg>`;
}

function text(x, y, s, anchor = "start", cls = "") {
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" class="${cls}">${s}</text>`;
}

function hline(x1, x2, y, stroke = "var(--grid)") {
  return `<line x1="${x1}" x2="${x2}" y1="${y}" y2="${y}" stroke="${stroke}" stroke-width="1"/>`;
}

// 竖直柱子：顶端 4px 圆角，底端方角
function columnTop(x, y, w, h, fill, r = 4) {
  r = Math.min(r, h, w / 2);
  return `<path fill="${fill}" d="M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}` +
         `Q${x + w},${y} ${x + w},${y + r}V${y + h}Z"/>`;
}

// 水平条：右端 4px 圆角，左端方角
function barRight(x, y, w, h, fill, r = 4) {
  r = Math.min(r, w, h / 2);
  return `<path fill="${fill}" d="M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}` +
         `V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z"/>`;
}

function timeTicks(top) {
  const ms = top >= 1000;
  return { div: ms ? 1000 : 1, unit: ms ? "ms" : "µs" };
}

function showTip(e, html) {
  tip.innerHTML = html;
  tip.style.display = "block";
  tip.style.left = Math.min(e.clientX + 12, innerWidth - tip.offsetWidth - 8) + "px";
  tip.style.top = Math.min(e.clientY + 12, innerHeight - tip.offsetHeight - 8) + "px";
}

// 容器内带 data-k 的元素：悬停显示提示，点击回调
function interact(root, html, click) {
  root.onmousemove = e => {
    const el = e.target.closest("[data-k]");
    el ? showTip(e, html(el.dataset.k)) : (tip.style.display = "none");
  };
  root.onmouseleave = () => (tip.style.display = "none");
  root.onclick = click && (e => {
    const el = e.target.closest("[data-k]");
    if (el) click(el.dataset.k);
  });
}

function selectLayer(i) {
  state.layer = +i;
  render();
}

function segment(id, options, key) {
  $(id).innerHTML = options.map(([v, label]) =>
    `<button aria-pressed="${String(state[key]) === String(v)}" data-v="${v}">${label}</button>`).join("");
  $(id).onclick = e => {
    const b = e.target.closest("button");
    if (!b) return;
    state[key] = key === "phase" ? +b.dataset.v : b.dataset.v;
    render();
  };
}

function renderHeader() {
  const m = D.model, t = D.transfer, p = D.precision;
  $("title").textContent = `${m.name} · ${D.server} × EP${EP} · MoE 层传输`;
  $("sub").textContent = `${m.n_experts} 个路由专家（每卡 ${D.experts_per_card}，${D.placement}）· ` +
    `top${m.topk} · hidden ${m.hidden} · ${t.mode} · 并发 ${t.concurrency} · 顺序变换 ${t.permute} · ` +
    `dispatch ${p.dispatch} / combine ${p.combine} · 只算传输与顺序变换，不含专家计算`;
  segment("f-phase", D.phases.map((ph, i) => [i, `${ph.name}（${ph.tokens_per_rank} token/卡）`]), "phase");
  segment("f-stage", [["dispatch", "dispatch"], ["combine", "combine"]], "stage");
  segment("f-dir", Object.entries(DIR_LABEL), "dir");
}

function renderTiles() {
  const ph = phase();
  const perCard = sum(ph.layers.map(l => sum(l.dispatch.sent) + sum(l.combine.sent))) / EP;
  const tiles = [
    ["每张卡经过的 MoE 层", ph.layers.length, `模型共 ${D.model.n_layers} 层`],
    [`每步传输时间（${ph.name}）`, fmtTime(ph.step_ms * 1000), "各层最慢卡之和"],
    ["每层平均", fmtTime(ph.layer_us), "dispatch + combine"],
    ["每卡每步发往对端", fmtBytes(perCard), "所有层 dispatch + combine"],
  ];
  $("tiles").innerHTML = tiles.map(([label, value, hint]) =>
    `<div class="tile"><div class="label">${label}</div><div class="value">${value}</div>` +
    `<div class="hint">${hint}</div></div>`).join("");
}

function renderLayers() {
  const L = phase().layers, W = 1000, H = 240, m = { l: 48, r: 8, t: 16, b: 24 };
  const pw = W - m.l - m.r, plotH = H - m.t - m.b, band = pw / L.length, bw = Math.min(24, band * 0.6);
  const top = niceMax(max(L.map(l => l.time_us)));
  const { div, unit } = timeTicks(top);
  const y = v => m.t + plotH * (1 - v / top);
  let body = text(0, 10, unit);
  for (let k = 0; k <= 4; k++) {
    const v = top * k / 4;
    body += hline(m.l, W - m.r, y(v), k ? "var(--grid)" : "var(--axis)") + text(m.l - 6, y(v) + 4, +(v / div).toFixed(2), "end");
  }
  L.forEach((l, i) => {
    const x = m.l + i * band + (band - bw) / 2, d = peak(l.dispatch), c = peak(l.combine);
    const fade = i === state.layer ? 1 : 0.35;
    body += `<g opacity="${fade}">` +
      `<rect x="${x}" y="${y(d)}" width="${bw}" height="${y(0) - y(d)}" fill="var(--series-1)"/>` +
      columnTop(x, y(d + c), bw, Math.max(0, y(d) - y(d + c) - 2), "var(--series-2)") + `</g>`;
    body += `<rect data-k="${i}" x="${m.l + i * band}" y="${m.t}" width="${band}" height="${plotH}" fill="transparent"/>`;
    if (i % 5 === 0 || i === state.layer) body += text(m.l + (i + 0.5) * band, H - 6, `L${l.layer}`, "middle",
                                                       i === state.layer ? "strong" : "");
  });
  $("layers").innerHTML = svg(W, H, body);
  interact($("layers"), k => {
    const l = L[k];
    return `<b>L${l.layer}</b> · 合计 ${fmtTime(l.time_us)}<br><span class="muted">dispatch</span> ${fmtTime(peak(l.dispatch))}` +
      `<span class="muted">（permute ${fmtTime(peakPermute(l.dispatch))}）</span> · <span class="muted">combine</span> ` +
      `${fmtTime(peak(l.combine))}<span class="muted">（unpermute ${fmtTime(peakPermute(l.combine))}）</span><br>` +
      `<span class="muted">负载不均衡</span> ${l.imbalance.toFixed(2)}`;
  }, selectLayer);
}

// 颜色映射到 [lo, hi]：卡与卡之间差异通常只有几个百分点，从 0 开始会看不出区别
function scale(values) {
  const lo = Math.min(...values), hi = max(values);
  return { lo, hi, color: v => rampColor(hi > lo ? (v - lo) / (hi - lo) : 0.5) };
}

function legendRamp(x, y, w, s) {
  const stops = RAMP.map((_, k) => `<stop offset="${k / (RAMP.length - 1)}" stop-color="${rampColor(k / (RAMP.length - 1))}"/>`);
  return `<defs><linearGradient id="ramp">${stops.join("")}</linearGradient></defs>` +
    `<rect x="${x}" y="${y}" width="${w}" height="8" rx="2" fill="url(#ramp)"/>` +
    text(x, y + 22, `最小 ${fmtBytes(s.lo)}`) + text(x + w, y + 22, `最大 ${fmtBytes(s.hi)}`, "end");
}

function renderHeat() {
  const L = phase().layers, { stage, dir } = state;
  const W = 1000, rowH = 22, m = { l: 48, r: 8, t: 4 }, pw = W - m.l - m.r, band = pw / L.length;
  const H = m.t + EP * rowH + 64;
  const s = scale(L.flatMap(l => l[stage][dir]));
  $("heat-title").textContent = `每张卡每层${DIR_LABEL[dir]}的数据量 · ${stage}`;
  let body = "";
  for (let c = 0; c < EP; c++) {
    body += text(m.l - 6, m.t + c * rowH + 15, `卡 ${c}`, "end");
    L.forEach((l, i) => {
      body += `<rect data-k="${i},${c}" x="${m.l + i * band + 1}" y="${m.t + c * rowH + 1}" width="${band - 2}" ` +
              `height="${rowH - 2}" rx="2" fill="${s.color(l[stage][dir][c])}"/>`;
    });
  }
  const yAxis = m.t + EP * rowH;
  body += `<rect x="${m.l + state.layer * band + 1}" y="${yAxis + 2}" width="${band - 2}" height="2" fill="var(--text-primary)"/>`;
  L.forEach((l, i) => {
    if (i % 5 === 0 || i === state.layer) body += text(m.l + (i + 0.5) * band, yAxis + 16, `L${l.layer}`, "middle",
                                                       i === state.layer ? "strong" : "");
  });
  body += legendRamp(m.l, yAxis + 30, 280, s);
  $("heat").innerHTML = svg(W, H, body);
  interact($("heat"), k => {
    const [i, c] = k.split(",").map(Number), st = L[i][stage];
    return `<b>卡 ${c} · L${L[i].layer}</b> · ${stage}<br><span class="muted">发出</span> ${fmtBytes(st.sent[c])}` +
      ` · <span class="muted">收到</span> ${fmtBytes(st.recv[c])}<br><span class="muted">本卡</span> ${fmtBytes(st.local[c])}` +
      ` · <span class="muted">耗时</span> ${fmtTime(st.card_time_us[c])}`;
  }, k => selectLayer(k.split(",")[0]));
}

// 本层某阶段卡 s 发给卡 d 的 token 份数（combine 是 dispatch 的反向）
function copies(l, s, d) {
  return state.stage === "dispatch" ? l.pair[s][d] : l.pair[d][s];
}

function renderMatrix() {
  const l = layer(), bpt = D.token_bytes[state.stage], ids = [...Array(EP).keys()];
  const cell = Math.min(36, 320 / EP), m = { l: 44, t: 20 }, size = m.l + EP * cell;
  const s = scale(ids.flatMap(src => ids.map(dst => copies(l, src, dst) * bpt)));
  $("matrix-title").textContent = `L${l.layer} 卡到卡数据量 · ${state.stage}`;
  let body = "";
  for (const src of ids) {
    body += text(m.l - 6, m.t + src * cell + cell / 2 + 4, `卡 ${src}`, "end") +
      text(m.l + src * cell + cell / 2, 12, src, "middle");
    for (const dst of ids) {
      body += `<rect data-k="${src},${dst}" x="${m.l + dst * cell + 1}" y="${m.t + src * cell + 1}" width="${cell - 2}" ` +
              `height="${cell - 2}" rx="2" fill="${s.color(copies(l, src, dst) * bpt)}"/>`;
    }
  }
  body += legendRamp(m.l, m.t + EP * cell + 12, EP * cell, s);
  $("matrix").innerHTML = svg(size + 8, m.t + EP * cell + 48, body);
  interact($("matrix"), k => {
    const [s, d] = k.split(",").map(Number), n = copies(l, s, d);
    const head = s === d ? `卡 ${s} → 本卡专家` : `卡 ${s} → 卡 ${d}`;
    return `<b>${head}</b><br>${fmtBytes(n * bpt)} <span class="muted">（${n} 份 token）</span>`;
  });
}

function renderCards() {
  const l = layer(), st = l[state.stage], W = 480, rowH = 24, m = { l: 44, r: 72, t: 8 };
  const pw = W - m.l - m.r, top = peak(st), x = v => pw * v / top, H = m.t + EP * rowH + 20;
  const color = op => op === "a2a" ? STAGE_COLOR[state.stage] : "var(--series-3)";
  $("cards-title").textContent = `L${l.layer} 每张卡的耗时 · ${state.stage}`;
  $("cards-legend").innerHTML = st.steps.map(s =>
    `<span><i class="swatch" style="background: ${color(s.op)}"></i>${stepLabel(s.op)}</span>`).join("");
  let body = "";
  for (let c = 0; c < EP; c++) {
    const y = m.t + c * rowH, segs = st.steps.map(s => [s.op, x(s.us[c])]).filter(([, w]) => w > 2);
    let x0 = m.l;
    body += text(m.l - 6, y + 15, `卡 ${c}`, "end");
    segs.forEach(([op, w], i) => {
      const last = i === segs.length - 1;
      body += last ? barRight(x0, y + 4, w, rowH - 8, color(op))
                   : `<rect x="${x0}" y="${y + 4}" width="${w - 2}" height="${rowH - 8}" fill="${color(op)}"/>`;
      x0 += w;
    });
    body += `<rect data-k="${c}" x="0" y="${y}" width="${W}" height="${rowH}" fill="transparent"/>`;
  }
  body += `<line x1="${m.l + x(top)}" x2="${m.l + x(top)}" y1="${m.t}" y2="${m.t + EP * rowH}" stroke="var(--text-secondary)" stroke-width="1"/>` +
    text(m.l + x(top) + 6, m.t + 12, fmtTime(top), "start", "strong");
  $("cards").innerHTML = svg(W, H, body);
  interact($("cards"), k => {
    const c = +k, t = st.card_time_us[c];
    return `<b>卡 ${c}</b> · ${fmtTime(t)}<span class="muted">，等待 ${fmtTime(top - t)}</span><br>` +
      st.steps.map(s => `<span class="muted">${stepLabel(s.op)}</span> ${fmtTime(s.us[c])}`).join(" · ") +
      `<span class="muted">（其中算顺序 ${fmtTime(st.index_us[c])}）</span><br>` +
      `<span class="muted">发出</span> ${fmtBytes(st.sent[c])} · <span class="muted">收到</span> ${fmtBytes(st.recv[c])}` +
      ` · <span class="muted">本卡</span> ${fmtBytes(st.local[c])}`;
  });
}

function renderTable() {
  const rows = phase().layers.map((l, i) =>
    `<tr data-i="${i}" class="${i === state.layer ? "selected" : ""}"><td>L${l.layer}</td>` +
    `<td>${fmtTime(peak(l.dispatch))}</td><td>${fmtTime(peakPermute(l.dispatch))}</td>` +
    `<td>${fmtTime(peak(l.combine))}</td><td>${fmtTime(peakPermute(l.combine))}</td><td>${fmtTime(l.time_us)}</td>` +
    `<td>${l.imbalance.toFixed(2)}</td><td>${fmtBytes(sum(l.dispatch.sent) / EP)}</td>` +
    `<td>${fmtBytes(sum(l.combine.sent) / EP)}</td></tr>`).join("");
  $("table").innerHTML = `<table><thead><tr><th>层</th><th>dispatch</th><th>其中 permute</th><th>combine</th>` +
    `<th>其中 unpermute</th><th>合计</th><th>负载不均衡</th><th>每卡发出 dispatch</th><th>每卡发出 combine</th></tr>` +
    `</thead><tbody>${rows}</tbody></table>`;
  $("table").onclick = e => {
    const tr = e.target.closest("tr[data-i]");
    if (tr) selectLayer(tr.dataset.i);
  };
}

function render() {
  renderHeader();
  renderTiles();
  renderLayers();
  renderHeat();
  renderMatrix();
  renderCards();
  renderTable();
}

dark.addEventListener("change", render);
render();
