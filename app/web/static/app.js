/* CADE analyst console. Vanilla JS, no external assets.
 * Every value from the API is rendered as text (never as HTML): incident content is untrusted. */
"use strict";

const state = { key: null, health: null, incidentId: null, vocabulary: null, notice: "" };
const $ = (selector) => document.querySelector(selector);

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (name.startsWith("on")) el.addEventListener(name.slice(2), value);
    else if (name === "class") el.className = value;
    else el.setAttribute(name, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}
const svg = (tag, attrs = {}) => {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
};
const pct = (x) => `${Math.round((x || 0) * 100)}%`;
const fmt = (x, digits = 2) => (typeof x === "number" ? x.toFixed(digits) : "-");

// An empty message restores the standing notice (e.g. SIMULATION MODE) instead of hiding it.
function banner(message, kind = "info") {
  const el = $("#banner");
  const text = message || state.notice;
  el.textContent = text;
  el.className = `banner ${kind === "error" ? "error" : ""} ${text ? "" : "hidden"}`;
}

async function api(method, path, body) {
  const headers = { Authorization: `Bearer ${state.key}` };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  const data = response.status === 204 ? null : await response.json().catch(() => null);
  if (!response.ok) {
    const error = data && data.error ? `${data.error.category}: ${data.error.message}` : `HTTP ${response.status}`;
    throw new Error(error);
  }
  return { status: response.status, data };
}

async function guarded(fn) {
  try { banner(""); await fn(); } catch (err) { banner(err.message, "error"); }
}

// --- connection ------------------------------------------------------------------------------

async function connect(key) {
  state.key = key;
  const { data } = await api("GET", "/health");
  if (!data.dependencies) throw new Error("the API key was not accepted");
  state.health = data;
  try { sessionStorage.setItem("cade-key", key); } catch (e) { /* storage may be unavailable */ }
  const mode = data.playbooks ? data.playbooks.execution_mode : "unknown";
  const badge = $("#mode-badge");
  badge.textContent = `${mode.toUpperCase()} mode - LLM ${data.dependencies.llm.status} - library ${data.playbooks ? data.playbooks.library_entries : "?"}`;
  badge.className = `badge ${mode === "live" ? "bad" : "synthetic"}`;
  const warnings = (data.playbooks && data.playbooks.warnings) || [];
  if (mode === "simulate") state.notice = "SIMULATION MODE: tool results are SYNTHETIC; no real tool is contacted and nothing is changed." + (warnings.length ? " " + warnings.join(" ") : "");
  else state.notice = warnings.join(" ");
  banner("");
  try { state.vocabulary = (await api("GET", "/vocabulary")).data; } catch (e) { state.vocabulary = null; }
  await loadIncidents();
}

$("#login").addEventListener("submit", (e) => { e.preventDefault(); guarded(() => connect($("#api-key").value.trim())); });
$("#logout").addEventListener("click", () => {
  try { sessionStorage.removeItem("cade-key"); } catch (e) { /* ignore */ }
  location.reload();
});
document.querySelectorAll(".tabs button").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b === button));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("hidden", t.id !== `tab-${button.dataset.tab}`));
  if (button.dataset.tab === "library") guarded(loadLibrary);
  if (button.dataset.tab === "audit") guarded(loadAudit);
}));

// --- incidents --------------------------------------------------------------------------------

async function loadIncidents() {
  const { data } = await api("GET", "/incidents?limit=100");
  const list = $("#incident-list");
  list.replaceChildren(...data.map((i) => h("li", { class: i.incident_id === state.incidentId ? "active" : "", onclick: () => guarded(() => openIncident(i.incident_id)) },
    h("div", {}, i.title || i.incident_id),
    h("div", { class: "meta" }, `${i.incident_id} - ${i.status} - ${i.severity || "no severity"}`))));
}
$("#refresh").addEventListener("click", () => guarded(loadIncidents));

$("#new-incident").addEventListener("submit", (e) => {
  e.preventDefault();
  guarded(async () => {
    const form = new FormData(e.target);
    const body = { description: form.get("description"), source: form.get("source") };
    if (form.get("title")) body.title = form.get("title");
    if (form.get("severity")) body.severity = form.get("severity");
    const { data } = await api("POST", "/incidents", body);
    e.target.reset();
    await loadIncidents();
    await openIncident(data.incident_id);
  });
});

async function openIncident(incidentId) {
  state.incidentId = incidentId;
  const detail = $("#detail");
  detail.replaceChildren(h("p", { class: "muted" }, "Loading..."));
  const incident = (await api("GET", `/incidents/${incidentId}`)).data;
  const analysis = incident.latest_analysis;
  let evidence = null, response = null;
  if (analysis && analysis.result) {
    evidence = (await api("GET", `/incidents/${incidentId}/evidence`)).data;
    response = (await api("GET", `/incidents/${incidentId}/response`)).data;
  }
  detail.replaceChildren(
    incidentHeader(incident.incident, analysis),
    analysis ? analysisCard(analysis) : h("div", { class: "card muted" }, "Not analysed yet."),
    evidence ? evidenceCard(evidence) : "",
    response ? routingCard(response.decisions) : "",
    ...(response ? response.plans.slice().reverse().map((p) => planCard(p, response.runs.filter((r) => r.plan_id === p.plan_id))) : []),
  );
  await loadIncidents();
}

function incidentHeader(incident, analysis) {
  const ready = analysis && analysis.result;
  const act = (label, fn, opts = {}) => h("button", { class: opts.secondary ? "secondary" : "", disabled: opts.disabled, onclick: () => guarded(fn) }, label);
  return h("div", { class: "card" },
    h("div", { class: "row between" },
      h("div", {}, h("h2", {}, incident.title || incident.incident_id), h("div", { class: "muted" }, `${incident.incident_id} - ${incident.source} - status ${incident.status}`)),
      h("div", { class: "row" },
        act("Analyse", async () => { banner("Analysing (the LLM may take a while)..."); await api("POST", `/incidents/${incident.incident_id}/analyze`); banner(""); await openIncident(incident.incident_id); }),
        act("Enrich evidence", async () => { await api("POST", `/incidents/${incident.incident_id}/enrich`, {}); await openIncident(incident.incident_id); }, { disabled: !ready, secondary: true }),
        act("Respond", () => startJob(incident.incident_id, "respond"), { disabled: !ready }),
        act("Force synthesis", () => startJob(incident.incident_id, "synthesize"), { disabled: !ready, secondary: true }))),
    h("pre", { class: "report" }, incident.raw_description));
}

async function startJob(incidentId, kind) {
  const { status, data } = await api("POST", `/incidents/${incidentId}/${kind}`);
  if (status !== 202) { await openIncident(incidentId); return; }
  for (;;) {
    const job = (await api("GET", `/jobs/${data.job_id}`)).data;
    banner(`Job ${job.job_id}: ${job.status} - ${job.progress || ""}`);
    if (job.status === "succeeded") { banner(""); break; }
    if (["failed", "interrupted"].includes(job.status)) throw new Error(`job ${job.status}: ${job.error ? job.error.message : ""}`);
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  await openIncident(incidentId);
}

function analysisCard(analysis) {
  if (!analysis.result) return h("div", { class: "card" }, h("h3", {}, "Analysis failed"), h("p", {}, analysis.error ? analysis.error.message : ""));
  const r = analysis.result;
  return h("div", { class: "card" },
    h("h3", {}, "Analysis ", h("span", { class: `badge ${analysis.extraction_status === "success" ? "ok" : "warn"}` }, analysis.extraction_status),
      h("span", { class: "badge muted" }, `${analysis.provider}/${analysis.model_name}`)),
    h("dl", { class: "kv" },
      h("dt", {}, "Intent"), h("dd", {}, `${r.intent.label} (confidence ${fmt(r.intent.confidence)})`),
      h("dt", {}, "Summary"), h("dd", {}, r.summary),
      h("dt", {}, "ATT&CK"), h("dd", {}, r.intent.attack_techniques.map((t) => `${t.technique_id} ${t.name || ""} [${t.validation_status}]`).join("; ") || "none"),
      h("dt", {}, "Entities"), h("dd", {}, Object.entries(r.entities).filter(([, v]) => v.length).map(([k, v]) => `${k}: ${v.map((x) => x.value || x).join(", ")}`).join(" | ") || "none")),
    analysis.warnings.length ? h("ul", { class: "warn-list" }, analysis.warnings.map((w) => h("li", {}, w))) : "");
}

function evidenceCard(evidence) {
  const items = (list) => list.length ? h("ul", {}, list.map((i) => h("li", {}, `${i.category}: ${i.description}`, i.criticality ? h("span", { class: "pill" }, i.criticality) : "", i.provenance ? h("span", { class: "pill" }, i.provenance.origin) : ""))) : h("p", { class: "muted" }, "none");
  return h("div", { class: "card" }, h("h3", {}, "Evidence"),
    h("div", { class: "grid2" },
      h("div", {}, h("h4", {}, "Observed (input)"), items(evidence.base.observed)),
      h("div", {}, h("h4", {}, "Missing"), items(evidence.base.missing)),
      h("div", {}, h("h4", {}, "Observed by real tools / analysts"), items(evidence.tool_observed.concat(evidence.analyst_supplied))),
      h("div", {}, h("h4", {}, "Simulated ", h("span", { class: "pill synthetic" }, "SYNTHETIC")),
        evidence.simulated.length ? h("ul", {}, evidence.simulated.map((s) => h("li", {}, `${s.category}: ${s.description}`))) : h("p", { class: "muted" }, "none"))),
    h("p", { class: "muted" }, `Real observed categories: ${evidence.real_observed_categories.join(", ") || "none"}`));
}

function routingCard(decisions) {
  if (!decisions.length) return h("div", { class: "card muted" }, "No routing decision yet. Use Respond.");
  const bar = (value) => h("div", { class: "bar" }, h("div", { style: `width:${pct(value)}` }));
  return h("div", { class: "card" }, h("h3", {}, "Routing decisions (Eq. 1-5)"),
    h("table", {}, h("thead", {}, h("tr", {}, ["#", "Round", "Action", "U_reuse", "U_request", "U_gen", "Cov", "Cov+", "H", "rho", "M", "s-hat", "Top candidate"].map((t) => h("th", {}, t)))),
      h("tbody", {}, decisions.map((d) => h("tr", {},
        h("td", {}, d.decision_id), h("td", {}, d.round), h("td", {}, h("strong", {}, d.action), d.synthetic_basis ? h("span", { class: "pill synthetic" }, "synthetic basis") : ""),
        h("td", {}, fmt(d.scores.reuse), bar(d.scores.reuse)), h("td", {}, fmt(d.scores.request_evidence), bar(d.scores.request_evidence)),
        h("td", {}, fmt(d.scores.synthesize), bar(d.scores.synthesize)),
        h("td", {}, fmt(d.terms.coverage)), h("td", {}, fmt(d.terms.counterfactual_coverage)), h("td", {}, fmt(d.terms.entropy)),
        h("td", {}, fmt(d.terms.requestability)), h("td", {}, fmt(d.terms.unmet_fraction)), h("td", {}, fmt(d.terms.smoothed_success)),
        h("td", {}, d.top_candidate ? d.top_candidate.name : "-"))))),
    h("p", { class: "muted" }, decisions[decisions.length - 1].rationale));
}

// --- plans ------------------------------------------------------------------------------------

function actionDef(id) {
  return state.vocabulary ? state.vocabulary.actions.find((a) => a.id === id) : null;
}

function planCard(plan, runs) {
  const fitness = plan.fitness;
  const synthesis = plan.synthesis;
  const card = h("div", { class: "card" },
    h("div", { class: "row between" },
      h("h3", {}, `Plan ${plan.plan_id} `, h("span", { class: "badge" }, plan.source), h("span", { class: "badge" }, plan.status),
        plan.synthetic_basis ? h("span", { class: "badge synthetic" }, "synthetic basis") : ""),
      h("span", { class: "muted" }, plan.playbook.name)),
    fitness ? h("div", { class: "row" }, ["ic", "ea", "os", "fitness"].map((k) => h("div", { style: "min-width:120px" }, h("div", { class: "muted" }, `${k.toUpperCase()} ${fmt(fitness[k])}${k === "os" ? ` (${fitness.os_source})` : ""}`), h("div", { class: "bar" }, h("div", { style: `width:${pct(fitness[k])}` }))))) : "",
    synthesis ? synthesisSummary(synthesis) : "",
    dagView(plan.playbook),
    nodeTable(plan),
    plan.status === "executed" ? "" : planActions(plan),
    ...runs.map(runView));
  return card;
}

function synthesisSummary(s) {
  const evo = s.evolution, seed = s.seeding, gen = s.generation;
  const history = evo.history || [];
  const width = 260, height = 60;
  const chart = svg("svg", { width, height, viewBox: `0 0 ${width} ${height}` });
  if (history.length > 1) {
    const points = history.map((p, i) => `${(i / (history.length - 1)) * (width - 10) + 5},${height - 5 - p.best_fitness * (height - 10)}`).join(" ");
    chart.append(svg("polyline", { points, fill: "none", stroke: "currentColor", "stroke-width": 2 }));
  }
  return h("div", { class: "row" },
    h("dl", { class: "kv" },
      h("dt", {}, "Generator"), h("dd", {}, `${gen.provider || "-"}/${gen.model || "-"} - ${gen.llm_valid} valid of ${gen.llm_calls} LLM samples${gen.constructive_used ? `, ${gen.constructive_used} constructive (${gen.fallback_reason})` : ""}`),
      h("dt", {}, "Adaptive K"), h("dd", {}, `${seed.k} seeds from ${seed.admissible} admissible (${seed.clusters} clusters, ${seed.duplicates} duplicates)`),
      h("dt", {}, "Evolution"), h("dd", {}, `${evo.termination} after ${evo.generations} generation(s), ${evo.evaluations} evaluations, ${evo.judge_calls} judge calls`)),
    h("div", { class: "muted", title: "best fitness per generation" }, chart));
}

function dagView(playbook) {
  const ids = playbook.nodes.map((n) => n.node_id);
  const preds = Object.fromEntries(ids.map((id) => [id, playbook.edges.filter((e) => e.target === id).map((e) => e.source)]));
  const layer = {};
  const depth = (id, seen = new Set()) => {
    if (layer[id] !== undefined) return layer[id];
    if (seen.has(id)) return 0;
    seen.add(id);
    layer[id] = preds[id].length ? 1 + Math.max(...preds[id].map((p) => depth(p, seen))) : 0;
    return layer[id];
  };
  ids.forEach((id) => depth(id));
  const columns = {};
  ids.forEach((id) => { (columns[layer[id]] = columns[layer[id]] || []).push(id); });
  const W = 170, H = 46, GX = 50, GY = 14;
  const pos = {};
  Object.entries(columns).forEach(([col, members]) => members.forEach((id, row) => { pos[id] = { x: 10 + col * (W + GX), y: 10 + row * (H + GY) }; }));
  const width = 20 + (Math.max(...Object.keys(columns).map(Number)) + 1) * (W + GX);
  const height = 20 + Math.max(...Object.values(columns).map((m) => m.length)) * (H + GY);
  const root = svg("svg", { width, height, viewBox: `0 0 ${width} ${height}` });
  for (const e of playbook.edges) {
    const a = pos[e.source], b = pos[e.target];
    if (!a || !b) continue;
    const path = svg("path", { class: "edge", d: `M${a.x + W},${a.y + H / 2} C${a.x + W + GX / 2},${a.y + H / 2} ${b.x - GX / 2},${b.y + H / 2} ${b.x},${b.y + H / 2}` });
    const title = svg("title"); title.textContent = e.artifact; path.append(title);
    root.append(path);
  }
  for (const n of playbook.nodes) {
    const def = actionDef(n.action) || {};
    const p = pos[n.node_id];
    const stroke = def.destructive ? "var(--bad)" : def.verification ? "var(--warn)" : "var(--border)";
    // CSS variables only resolve inside style, not in SVG presentation attributes
    root.append(svg("rect", { x: p.x, y: p.y, width: W, height: H, rx: 6,
      style: `fill: var(--phase-${def.phase || "investigation"}); stroke: ${stroke}; stroke-width: ${def.destructive || def.verification ? 2 : 1}px` }));
    const label = svg("text", { x: p.x + 8, y: p.y + 18 }); label.textContent = (def.name || n.action).slice(0, 26);
    const sub = svg("text", { x: p.x + 8, y: p.y + 34 }); sub.textContent = `${n.node_id} - ${def.phase || ""}`;
    root.append(label, sub);
  }
  return h("div", { class: "dag" }, root);
}

function nodeTable(plan) {
  return h("details", {}, h("summary", {}, `Steps (${plan.playbook.nodes.length})`),
    h("table", {}, h("thead", {}, h("tr", {}, ["Node", "Action", "Binding", "Params"].map((t) => h("th", {}, t)))),
      h("tbody", {}, plan.playbook.nodes.map((n) => {
        const def = actionDef(n.action) || {};
        return h("tr", {}, h("td", {}, n.node_id),
          h("td", {}, def.name || n.action, def.destructive ? h("span", { class: "pill destructive" }, "destructive") : "", def.verification ? h("span", { class: "pill verification" }, "verification") : ""),
          h("td", {}, def.binding || ""), h("td", {}, Object.keys(n.params).length ? JSON.stringify(n.params) : ""));
      }))));
}

function planActions(plan) {
  const box = h("div", { class: "stack" });
  if (plan.destructive_nodes.length) {
    const checks = plan.destructive_nodes.map((id) => {
      const node = plan.playbook.nodes.find((n) => n.node_id === id);
      const approved = plan.approvals.some((a) => a.node_id === id && a.plan_hash === plan.structural_hash);
      return h("label", { class: "row" }, h("input", { type: "checkbox", value: id, style: "width:auto", checked: approved }), `${id} ${node.action}${approved ? " (approved)" : ""}`);
    });
    box.append(h("div", { class: "row" }, h("strong", {}, "Approve destructive steps:"), ...checks,
      h("button", { class: "danger small", onclick: () => guarded(async () => {
        const ids = checks.map((c) => c.querySelector("input")).filter((i) => i.checked).map((i) => i.value);
        if (!ids.length) throw new Error("select at least one step to approve");
        await api("POST", `/response-plans/${plan.plan_id}/approve`, { node_ids: ids });
        await openIncident(plan.incident_id);
      }) }, "Approve (approver role)")));
  }
  if (plan.verification_nodes.length) {
    const nodeSel = h("select", { style: "width:auto" }, plan.verification_nodes.map((id) => h("option", { value: id }, id)));
    const verdict = h("select", { style: "width:auto" }, h("option", { value: "confirmed" }, "confirmed"), h("option", { value: "not_confirmed" }, "not confirmed"));
    const targets = h("input", { placeholder: "confirmed target(s), comma-separated", style: "width:260px" });
    box.append(h("div", { class: "row" }, h("strong", {}, "Record verification:"), nodeSel, verdict, targets,
      h("button", { class: "small", onclick: () => guarded(async () => {
        await api("POST", `/response-plans/${plan.plan_id}/verifications`, { node_id: nodeSel.value, verdict: verdict.value, targets: targets.value.split(",").map((t) => t.trim()).filter(Boolean) });
        await openIncident(plan.incident_id);
      }) }, "Record (approver role)")),
      plan.verifications.length ? h("div", { class: "muted" }, plan.verifications.map((v) => `${v.node_id}: ${v.verdict} ${v.targets.join(", ")} by ${v.verified_by}`).join(" | ")) : "");
  }
  box.append(h("div", {}, h("button", { onclick: () => guarded(async () => {
    banner("Executing...");
    await api("POST", `/response-plans/${plan.plan_id}/execute`);
    banner("");
    await openIncident(plan.incident_id);
  }) }, "Execute plan (executor role)")));
  return box;
}

function runView(run) {
  const feedback = run.feedback;
  const resolved = h("input", { type: "checkbox", style: "width:auto" });
  const retained = h("input", { type: "checkbox", style: "width:auto" });
  return h("div", { class: "card" },
    h("h4", {}, `Run ${run.run_id} `, h("span", { class: "badge" }, run.status), h("span", { class: "badge" }, run.mode), run.synthetic ? h("span", { class: "badge synthetic" }, "SYNTHETIC") : ""),
    run.block_reason ? h("p", { class: "status-blocked" }, run.block_reason) : "",
    h("table", {}, h("thead", {}, h("tr", {}, ["Node", "Action", "Status", "Targets", "Reason / findings"].map((t) => h("th", {}, t)))),
      h("tbody", {}, run.node_results.map((r) => h("tr", {}, h("td", {}, r.node_id),
        h("td", {}, r.action, r.destructive ? h("span", { class: "pill destructive" }, "destructive") : "", r.synthetic ? h("span", { class: "pill synthetic" }, "synthetic") : ""),
        h("td", { class: `status-${r.status}` }, r.status), h("td", {}, r.targets.join(", ")),
        h("td", {}, [r.reason, ...r.findings].filter(Boolean).join(" | ")))))),
    run.report.length ? h("details", {}, h("summary", {}, "Incident report"), h("pre", { class: "report" }, run.report.join("\n"))) : "",
    feedback
      ? h("p", { class: "muted" }, `Feedback by ${feedback.submitted_by}: resolved=${feedback.resolved}, retained=${feedback.findings_retained}; library: ${feedback.library_outcome.action} (${feedback.library_outcome.reason})`)
      : h("div", { class: "row" }, h("label", { class: "row" }, resolved, "resolved the incident"), h("label", { class: "row" }, retained, "findings retained"),
        h("button", { class: "small", onclick: () => guarded(async () => {
          await api("POST", `/playbook-runs/${run.run_id}/feedback`, { resolved: resolved.checked, findings_retained: retained.checked });
          await openIncident(run.incident_id);
        }) }, "Submit feedback")));
}

// --- library & audit ---------------------------------------------------------------------------

async function loadLibrary() {
  const { data } = await api("GET", "/library/playbooks");
  $("#library").replaceChildren(h("div", { class: "card" }, h("h3", {}, `Playbook library (${data.length})`),
    h("table", {}, h("thead", {}, h("tr", {}, ["Name", "Intent", "Origin", "s / n", "s-hat", "State", "Critical evidence"].map((t) => h("th", {}, t)))),
      h("tbody", {}, data.map((e) => h("tr", {}, h("td", {}, e.name), h("td", {}, e.intent_label), h("td", {}, e.origin),
        h("td", {}, `${e.successes} / ${e.executions}`), h("td", {}, fmt(e.smoothed_success)), h("td", {}, e.deprioritized ? "de-prioritised" : "active"),
        h("td", {}, e.required.critical.join(", "))))))));
}

async function loadAudit() {
  try {
    const { data } = await api("GET", "/audit?limit=200");
    $("#audit").replaceChildren(h("div", { class: "card" }, h("h3", {}, "Audit log (most recent first)"),
      h("table", {}, h("thead", {}, h("tr", {}, ["Time", "Actor", "Action", "Outcome", "Target", "Details"].map((t) => h("th", {}, t)))),
        h("tbody", {}, data.map((e) => h("tr", {}, h("td", {}, new Date(e.created_at).toLocaleString()), h("td", {}, e.actor || "system"),
          h("td", {}, e.action), h("td", {}, e.outcome), h("td", {}, `${e.target_type || ""} ${e.target_id || ""}`), h("td", {}, JSON.stringify(e.details))))))));
  } catch (err) {
    $("#audit").replaceChildren(h("div", { class: "card muted" }, `Audit log unavailable: ${err.message}`));
  }
}

// --- start ---------------------------------------------------------------------------------------
(function init() {
  let saved = null;
  try { saved = sessionStorage.getItem("cade-key"); } catch (e) { /* ignore */ }
  if (saved) { $("#api-key").value = saved; guarded(() => connect(saved)); }
})();
