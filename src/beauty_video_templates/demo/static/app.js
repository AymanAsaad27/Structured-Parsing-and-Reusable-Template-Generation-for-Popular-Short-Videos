"use strict";
const $ = (id) => document.getElementById(id);
const ROLES = {
  hook: "Opening hook",
  product_introduction: "Product introduction",
  texture_demonstration: "Texture detail",
  application_demonstration: "Application",
  result_comparison: "Result / comparison",
  presenter_explanation: "Presenter explanation",
  call_to_action: "Closing / call to action",
  detail_cutaway: "Detail cutaway",
};
const state = { session: null, template: null, busy: false, health: null };
const escapeHTML = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const timeLabel = (seconds) =>
  `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;

async function api(path, options = {}) {
  const headers = { "X-Demo-Client": "1", ...(options.headers || {}) };
  if (options.body && !(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(options.body);
  }
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    let detail;
    try {
      detail = (await response.json()).detail;
    } catch {
      detail = response.statusText;
    }
    if (Array.isArray(detail)) detail = detail.map((x) => x.msg).join("; ");
    throw new Error(detail || "The request could not be completed.");
  }
  return response.json();
}
function notice(text, success = false) {
  $("notice").textContent = text;
  $("notice").classList.toggle("hidden", !text);
  $("notice").classList.toggle("success", success);
}
function fail(error) {
  console.error(error);
  notice(error.message || String(error));
}
function setBusy(value) {
  state.busy = value;
  document.body.classList.toggle("busy", value);
  updateButtons();
}
function updateButtons() {
  $("analyse-button").disabled = state.busy || !state.session?.reference;
  $("render-button").disabled =
    state.busy || !state.template || !state.session?.assets?.length;
  $("sample-button").disabled = state.busy || !state.health?.sample_available;
}
function roleOptions(selected, auto = false) {
  return (
    (auto ? '<option value="auto">Automatic visual tag</option>' : "") +
    Object.entries(ROLES)
      .map(
        ([key, label]) =>
          `<option value="${key}" ${selected === key ? "selected" : ""}>${label}</option>`,
      )
      .join("")
  );
}
async function ensureSession() {
  if (!state.session) {
    state.session = await api("/api/sessions", { method: "POST" });
    setSessionURL();
  }
  return state.session.id;
}
function setSessionURL() {
  const url = new URL(location.href);
  url.searchParams.set("session", state.session.id);
  history.replaceState(null, "", url);
  localStorage.setItem("beauty-studio-session", state.session.id);
}
function applySession(data, keepTemplate = false) {
  state.session = data;
  if (!keepTemplate)
    state.template = data.template ? structuredClone(data.template) : null;
  setSessionURL();
  $("product-name").value = data.product_name || "";
  $("selling-points").value = (data.selling_points || []).join("\n");
  const reference = data.reference;
  $("reference-empty").classList.toggle("hidden", !!reference);
  $("reference-video").classList.toggle("hidden", !reference);
  $("reference-name").classList.toggle("hidden", !reference);
  $("replace-reference").classList.toggle("hidden", !reference);
  if (reference) {
    if (!$("reference-video").src.endsWith(reference.url))
      $("reference-video").src = reference.url;
    $("reference-name").textContent = reference.name;
  }
  $("music-name").textContent = data.music?.name || "Choose replacement music";
  paintAssets();
  paintTemplate();
  paintResult();
  updateButtons();
}
function paintAssets() {
  $("asset-list").innerHTML = (state.session?.assets || [])
    .map(
      (asset) =>
        `<div class="asset-row" data-id="${asset.id}">${asset.thumbnail_url || asset.kind === "image" ? `<img src="${escapeHTML(asset.thumbnail_url || asset.url)}" alt="${escapeHTML(asset.name)}">` : `<video src="${escapeHTML(asset.url)}" muted preload="metadata" aria-label="${escapeHTML(asset.name)}"></video>`}<div><div class="asset-name" title="${escapeHTML(asset.name)}">${escapeHTML(asset.name)}</div><select aria-label="Function tag for ${escapeHTML(asset.name)}" data-asset-role="${asset.id}">${roleOptions(asset.role, true)}</select></div><button class="remove-asset" data-remove-asset="${asset.id}" aria-label="Remove ${escapeHTML(asset.name)}">×</button></div>`,
    )
    .join("");
}
function paintTemplate() {
  const template = state.template;
  $("template-empty").classList.toggle("hidden", !!template);
  $("template-editor").classList.toggle("hidden", !template);
  $("template-status").textContent = template
    ? "Ready to review"
    : "Waiting for a reference";
  $("template-status").classList.toggle("ready", !!template);
  $("reference-profile").classList.toggle("hidden", !template);
  $("step2").classList.toggle("active", !!template);
  if (!template) return;
  const style = template.style || {};
  $("template-summary").innerHTML =
    `<div class="summary-metric"><strong>${template.slots.length}</strong><span>Shot slots</span></div><div class="summary-metric"><strong>${template.duration_seconds.toFixed(1)}<small>s</small></strong><span>Total duration</span></div><div class="summary-metric"><strong>${(template.duration_seconds / template.slots.length).toFixed(1)}<small>s</small></strong><span>Average shot</span></div>`;
  $("reference-profile").innerHTML =
    `<div class="profile-row"><span>Editing pace</span><strong>${escapeHTML(style.pace || "Imported template")}</strong></div><div class="profile-row"><span>Beat candidates</span><strong>${style.estimated_beat_times?.length || 0} estimated</strong></div><div class="profile-row"><span>Caption position</span><strong>${escapeHTML(style.caption_placement || "Editable")}</strong></div><div class="profile-row"><span>Reference palette</span><div class="palette">${(
      style.palette || []
    )
      .filter((x) => /^#[a-f0-9]{6}$/i.test(x))
      .map((c) => `<i style="background:${c}" title="${c}"></i>`)
      .join("")}</div></div>`;
  $("timeline").innerHTML = template.slots
    .map(
      (s, i) =>
        `<button style="flex:${s.end - s.start}" title="Shot ${i + 1}: ${escapeHTML(ROLES[s.role])}, ${(s.end - s.start).toFixed(2)}s" data-jump="${i}">${i + 1}</button>`,
    )
    .join("");
  $("timeline-end").textContent = timeLabel(template.duration_seconds);
  $("shot-list").innerHTML = template.slots
    .map((slot, i) => {
      const source =
        state.session?.reference &&
        template.provenance?.reference_asset_id === state.session.reference.id;
      const thumbnail = source
        ? `/api/sessions/${state.session.id}/media/analysis/shot_${String(i + 1).padStart(3, "0")}.jpg`
        : null;
      const assets = (state.session?.assets || [])
        .map(
          (a) =>
            `<option value="${a.id}" ${slot.asset_id === a.id ? "selected" : ""}>${escapeHTML(a.name)}</option>`,
        )
        .join("");
      return `<article class="shot-card" id="shot-${i}">${thumbnail ? `<img class="shot-thumb" src="${thumbnail}" alt="Reference shot ${i + 1}" loading="lazy">` : '<div class="shot-thumb"></div>'}<div><div class="shot-card-top"><strong>SHOT ${String(i + 1).padStart(2, "0")}</strong><span>${slot.start.toFixed(2)}–${slot.end.toFixed(2)}s</span></div><select aria-label="Function of shot ${i + 1}" data-slot="${i}" data-field="role">${roleOptions(slot.role)}</select><small class="role-source">${escapeHTML((slot.role_source || "Imported").replace("CLIP zero-shot estimate", "Visual estimate"))} · ${escapeHTML((slot.shot_size || "").replaceAll("_", " "))}</small><textarea aria-label="Caption for shot ${i + 1}" placeholder="Leave blank for product-specific caption" maxlength="180" data-slot="${i}" data-field="caption">${escapeHTML(slot.caption || "")}</textarea><div class="slot-controls"><label>Duration (s)<input type="number" value="${(slot.end - slot.start).toFixed(3)}" min="0.042" max="90" step="0.042" aria-label="Duration of shot ${i + 1}" data-duration="${i}"></label><label>Product material<select aria-label="Material for shot ${i + 1}" data-slot="${i}" data-field="asset_id"><option value="">Automatic matching</option>${assets}</select></label></div></div></article>`;
    })
    .join("");
}
function paintResult() {
  const render = state.session?.last_render;
  $("result-section").classList.toggle("hidden", !render);
  $("step3").classList.toggle("active", !!render);
  if (!render) return;
  if (!$("result-video").src.endsWith(render.video_url))
    $("result-video").src = render.video_url;
  if (state.session.reference) {
    $("compare-reference").src = state.session.reference.url;
    $("compare-reference").parentElement.classList.remove("hidden");
  } else {
    $("compare-reference").parentElement.classList.add("hidden");
  }
  $("download-video").href = render.video_url;
  $("download-comparison").classList.toggle("hidden", !render.comparison_url);
  $("download-comparison").href = render.comparison_url || "#";
  $("download-evidence").href = render.evidence_url;
  const e = render.evidence;
  const agreement = e.boundary_agreement?.f1;
  $("evidence-metrics").innerHTML = [
    ["Video duration", `${e.duration_seconds.toFixed(2)}s`],
    [
      "Planned timing error",
      `${(e.timing_error_seconds * 1000).toFixed(1)} ms`,
    ],
    [
      "Detected cut agreement",
      agreement == null ? "Not measured" : `${(agreement * 100).toFixed(0)}%`,
    ],
    ["New materials used", e.unique_materials],
    ["Reference audio reused", e.reference_audio_reused ? "Yes" : "No"],
  ]
    .map(
      ([a, b]) =>
        `<div class="evidence-row"><span>${a}</span><b>${b}</b></div>`,
    )
    .join("");
  $("result-warnings").innerHTML = [...new Set(e.warnings || [])]
    .map((w) => `<p class="result-warning">${escapeHTML(w)}</p>`)
    .join("");
}
async function upload(kind, files) {
  if (!files.length || state.busy) return;
  notice("");
  setBusy(true);
  try {
    const id = await ensureSession();
    const form = new FormData();
    for (const file of files) form.append("files", file);
    const product = $("product-name").value,
      points = $("selling-points").value;
    const data = await api(`/api/sessions/${id}/upload/${kind}`, {
      method: "POST",
      body: form,
    });
    applySession(data, kind !== "reference");
    $("product-name").value = product;
    $("selling-points").value = points;
    if (kind === "reference")
      notice("Reference uploaded. Analyse it to build the template.", true);
  } catch (error) {
    fail(error);
  } finally {
    setBusy(false);
  }
}
async function followJob(jobID, kind) {
  $("job-panel").classList.remove("hidden");
  try {
    while (true) {
      const job = await api(`/api/jobs/${jobID}`);
      $("job-message").textContent = job.message;
      $("job-percent").textContent = `${job.progress}%`;
      $("job-progress").value = job.progress;
      if (job.status === "failed") throw new Error(job.message);
      if (job.status === "succeeded") break;
      await new Promise((resolve) => setTimeout(resolve, 900));
    }
    applySession(await api(`/api/sessions/${state.session.id}`));
    const warnings = state.session.warnings || [];
    notice(
      warnings.length
        ? warnings.join("\n")
        : kind === "analysis"
          ? "Template ready. Review the shot functions, then create your product video."
          : "Your new video is ready.",
      !warnings.length,
    );
    if (kind === "render")
      $("result-section").scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
  } finally {
    $("job-panel").classList.add("hidden");
    setBusy(false);
  }
}
async function analyse() {
  notice("");
  setBusy(true);
  try {
    const job = await api(`/api/sessions/${state.session.id}/analyse`, {
      method: "POST",
      body: {
        read_text: $("read-text").checked,
        detect_beats: $("detect-beats").checked,
      },
    });
    await followJob(job.job_id, "analysis");
  } catch (error) {
    fail(error);
    setBusy(false);
  }
}
async function render() {
  notice("");
  setBusy(true);
  try {
    const size = Number($("export-size").value);
    state.template.width = size;
    state.template.height = size === 720 ? 1280 : 640;
    const job = await api(`/api/sessions/${state.session.id}/render`, {
      method: "POST",
      body: {
        template: state.template,
        product_name: $("product-name").value.trim() || "Your product",
        selling_points: $("selling-points")
          .value.split("\n")
          .map((x) => x.trim())
          .filter(Boolean),
        captions: $("use-captions").checked,
      },
    });
    await followJob(job.job_id, "render");
  } catch (error) {
    fail(error);
    setBusy(false);
  }
}
function bindDrop(id, input, kind) {
  const zone = $(id);
  zone.addEventListener("click", () => !state.busy && $(input).click());
  zone.addEventListener("keydown", (event) => {
    if (["Enter", " "].includes(event.key)) {
      event.preventDefault();
      if (!state.busy) $(input).click();
    }
  });
  zone.addEventListener("dragover", (event) => {
    event.preventDefault();
    zone.classList.add("drag-over");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    zone.classList.remove("drag-over");
    upload(kind, [...event.dataTransfer.files]);
  });
  $(input).addEventListener("change", (event) =>
    upload(kind, [...event.target.files]),
  );
}
bindDrop("reference-empty", "reference-input", "reference");
bindDrop("material-drop", "asset-input", "assets");
$("replace-reference").onclick = () => $("reference-input").click();
$("music-input").onchange = (event) => upload("music", [...event.target.files]);
$("analyse-button").onclick = analyse;
$("render-button").onclick = render;
$("method-toggle").onclick = () => $("method").classList.toggle("hidden");
$("sample-button").onclick = async () => {
  notice("");
  setBusy(true);
  try {
    applySession(await api("/api/sample", { method: "POST" }));
    notice(
      "Sample loaded: a real reference and four previously generated Medicube clips. Click Analyse reference to extract a new template.",
      true,
    );
  } catch (error) {
    fail(error);
  } finally {
    setBusy(false);
  }
};
$("import-button").onclick = () => $("template-input").click();
$("template-input").onchange = async (event) => {
  try {
    const file = event.target.files[0];
    if (!file) return;
    if (file.size > 1024 * 1024)
      throw new Error("Choose a template JSON smaller than 1 MB.");
    const template = JSON.parse(await file.text());
    const id = await ensureSession();
    applySession(
      await api(`/api/sessions/${id}/template`, {
        method: "PUT",
        body: { template },
      }),
    );
    notice(
      "Reusable template imported. Add new product materials to create another video.",
      true,
    );
  } catch (error) {
    fail(error);
  }
};
$("download-template").onclick = () => {
  const blob = new Blob([JSON.stringify(state.template, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "beauty-template.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("shot-list").addEventListener("input", (event) => {
  const input = event.target;
  if (input.dataset.slot !== undefined) {
    const slot = state.template.slots[Number(input.dataset.slot)];
    slot[input.dataset.field] = input.value;
    if (input.dataset.field === "role") slot.role_source = "User reviewed";
  }
});
$("shot-list").addEventListener("change", (event) => {
  const input = event.target;
  if (input.dataset.duration !== undefined) {
    const index = Number(input.dataset.duration),
      value = Number(input.value);
    if (
      !Number.isFinite(value) ||
      value < 1 / state.template.fps ||
      value > 90
    ) {
      notice("Choose a duration between one frame and 90 seconds.");
      paintTemplate();
      return;
    }
    let cursor = 0;
    const updated = state.template.slots.map((slot, i) => {
      const duration =
        i === index
          ? Math.max(1, Math.round(value * state.template.fps)) /
            state.template.fps
          : slot.end - slot.start;
      const item = { ...slot, start: cursor, end: cursor + duration };
      cursor += duration;
      return item;
    });
    if (cursor > 90) {
      notice("Keep the total template under 90 seconds.");
      paintTemplate();
      return;
    }
    state.template.slots = updated;
    state.template.duration_seconds = cursor;
    paintTemplate();
  }
});
$("timeline").addEventListener("click", (event) => {
  const button = event.target.closest("[data-jump]");
  if (button) {
    const index = Number(button.dataset.jump);
    $(`shot-${index}`).scrollIntoView({ behavior: "smooth", block: "nearest" });
    if (state.session.reference)
      $("reference-video").currentTime = state.template.slots[index].start;
  }
});
for (const [button, box] of [
  ["caption-top", [0.07, 0.08, 0.93, 0.22]],
  ["caption-bottom", [0.07, 0.73, 0.93, 0.87]],
])
  $(button).onclick = () => {
    for (const slot of state.template.slots) slot.caption_box = [...box];
    notice("Caption position updated for all shots.", true);
  };
$("asset-list").addEventListener("change", async (event) => {
  const input = event.target;
  if (!input.dataset.assetRole) return;
  try {
    const product = $("product-name").value,
      points = $("selling-points").value;
    applySession(
      await api(
        `/api/sessions/${state.session.id}/assets/${input.dataset.assetRole}`,
        { method: "PATCH", body: { role: input.value } },
      ),
      true,
    );
    $("product-name").value = product;
    $("selling-points").value = points;
  } catch (error) {
    fail(error);
  }
});
$("asset-list").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-remove-asset]");
  if (!button) return;
  try {
    applySession(
      await api(
        `/api/sessions/${state.session.id}/assets/${button.dataset.removeAsset}`,
        { method: "DELETE" },
      ),
      true,
    );
  } catch (error) {
    fail(error);
  }
});
$("presentation-toggle").onclick = () => {
  if (!state.session?.last_render) {
    notice("Create a video first to open the side-by-side presentation view.");
    return;
  }
  document.body.classList.toggle("presentation");
  $("presentation-toggle").innerHTML = document.body.classList.contains(
    "presentation",
  )
    ? "Back to studio ↙"
    : "Presentation view ↗";
  window.scrollTo({ top: 0, behavior: "smooth" });
};
$("sync-play").onclick = () => {
  const reference = $("compare-reference"),
    result = $("result-video");
  reference.currentTime = 0;
  result.currentTime = 0;
  reference.muted = true;
  Promise.allSettled([reference.play(), result.play()]);
};
$("reset-session").onclick = async () => {
  if (state.busy) return;
  try {
    document.body.classList.remove("presentation");
    $("presentation-toggle").innerHTML = "Presentation view ↗";
    applySession(await api("/api/sessions", { method: "POST" }));
    notice("New session ready.", true);
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (error) {
    fail(error);
  }
};
async function init() {
  try {
    state.health = await api("/api/health");
    if (!state.health.sample_available) {
      $("sample-button").textContent = "Sample media not configured";
      $("sample-button").title =
        "Run scripts/prepare_studio_demo.py or upload your own files.";
    }
    const session =
      new URL(location.href).searchParams.get("session") ||
      localStorage.getItem("beauty-studio-session");
    if (session) {
      try {
        applySession(await api(`/api/sessions/${session}`));
      } catch {
        localStorage.removeItem("beauty-studio-session");
      }
    }
    updateButtons();
  } catch (error) {
    fail(error);
  }
}
init();
