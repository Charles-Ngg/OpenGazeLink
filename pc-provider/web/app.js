const $ = (id) => document.getElementById(id);
let state = null;
let activePage = "control";
let calibration = {
  active: false, phase: "idle", targets: [], lightTargets: [], poseTargets: [],
  index: 0, lightIndex: 0, poseIndex: 0, busy: false
};
let toastTimer = 0;
let actionBusy = false;
let configDirty = false;
let frameStreamActive = false;
let gazeStream = null;
let gazeArrivalTimes = [];
let artifactRenderKey = "";
let statusTimer = 0;
let statusRequestPending = false;
let lightingInventoryKey = "";
let lightingSelectionKey = "";
let lightingSelectionMode = "calibration";
let dismissedRecoveryKey = "";

async function request(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `${response.status}`);
  return payload;
}

function post(path, body = {}) {
  return request(path, { method: "POST", body: JSON.stringify(body) });
}

function toast(message) {
  window.clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").hidden = false;
  toastTimer = window.setTimeout(() => { $("toast").hidden = true; }, 3500);
}

function setPage(page) {
  activePage = page;
  document.querySelectorAll(".nav-button").forEach((button) => button.classList.toggle("active", button.dataset.page === page));
  document.querySelectorAll(".page").forEach((element) => element.classList.remove("active"));
  $(`${page}Page`).classList.add("active");
  if (state) renderStatus(state);
  syncStreams();
  refreshStatus();
}

function numericInput(id, label) {
  const input = $(id);
  const value = input.valueAsNumber;
  const minimum = input.min === "" ? -Infinity : Number(input.min);
  const maximum = input.max === "" ? Infinity : Number(input.max);
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new Error(`${label}填写无效`);
  }
  return value;
}

function configPayload() {
  const landmarker = document.querySelector('input[name="landmarker"]:checked');
  if (!state || !landmarker) throw new Error("配置尚未加载完成");
  return {
    landmarker: landmarker.value,
    input_source: $("inputSource").value,
    lighting_profile: $("lightingProfile").value,
    udp_port: numericInput("udpPort", "UDP端口"),
    rotate: $("rotate").value,
    mirror: $("mirror").checked,
    windows_camera_index: Number($("windowsCameraIndex").value),
    windows_camera_width: numericInput("windowsCameraWidth", "PC摄像头宽度"),
    windows_camera_height: numericInput("windowsCameraHeight", "PC摄像头高度"),
    windows_camera_fps: numericInput("windowsCameraFps", "PC摄像头 FPS"),
    windows_camera_backend: $("windowsCameraBackend").value,
    windows_camera_fov_x_degrees: numericInput("windowsCameraFov", "PC摄像头水平视场角"),
    screen_width: numericInput("screenWidth", "屏幕宽度"),
    screen_height: numericInput("screenHeight", "屏幕高度"),
    screen_diagonal_inches: numericInput("screenDiagonal", "屏幕尺寸"),
    camera_offset_x_cm: numericInput("cameraOffsetX", "相机水平位置"),
    camera_offset_y_cm: numericInput("cameraOffsetY", "相机垂直位置"),
    camera_offset_z_cm: numericInput("cameraOffsetZ", "相机深度位置"),
    geometry_configured: true,
    shared_memory_name: $("sharedMemory").value,
    one_euro_enabled: $("oneEuroEnabled").checked,
    one_euro_min_cutoff: numericInput("oneEuroMinCutoff", "最小截止频率"),
    one_euro_beta: numericInput("oneEuroBeta", "速度响应 beta"),
    one_euro_derivative_cutoff: numericInput("oneEuroDerivativeCutoff", "导数截止频率"),
    extrapolation_enabled: $("extrapolationEnabled").checked,
    extrapolation_horizon_ms: numericInput("extrapolationHorizon", "预测时域"),
    extrapolation_max_lead_fraction: numericInput(
      "extrapolationMaxLeadPercent", "最大外推距离"
    ) / 100,
    motion_diagnostics_enabled: $("motionDiagnosticsEnabled").checked
  };
}

function applyConfig(config) {
  document.querySelector(`input[name="landmarker"][value="${config.landmarker}"]`).checked = true;
  $("lightingProfile").value = config.lighting_profile || "reference";
  $("inputSource").value = config.input_source || "phone_udp";
  $("udpPort").value = config.udp_port;
  $("rotate").value = config.rotate;
  $("mirror").checked = config.mirror;
  $("windowsCameraIndex").value = config.windows_camera_index ?? 0;
  $("windowsCameraWidth").value = config.windows_camera_width ?? 640;
  $("windowsCameraHeight").value = config.windows_camera_height ?? 480;
  $("windowsCameraFps").value = config.windows_camera_fps ?? 30;
  $("windowsCameraBackend").value = config.windows_camera_backend || "auto";
  $("windowsCameraFov").value = config.windows_camera_fov_x_degrees ?? 60;
  $("screenWidth").value = config.screen_width;
  $("screenHeight").value = config.screen_height;
  $("screenDiagonal").value = config.screen_diagonal_inches;
  $("cameraOffsetX").value = config.camera_offset_x_cm;
  $("cameraOffsetY").value = config.camera_offset_y_cm;
  $("cameraOffsetZ").value = config.camera_offset_z_cm;
  $("sharedMemory").value = config.shared_memory_name;
  $("oneEuroEnabled").checked = config.one_euro_enabled;
  $("oneEuroMinCutoff").value = config.one_euro_min_cutoff;
  $("oneEuroBeta").value = config.one_euro_beta;
  $("oneEuroDerivativeCutoff").value = config.one_euro_derivative_cutoff;
  $("extrapolationEnabled").checked = config.extrapolation_enabled;
  $("extrapolationHorizon").value = config.extrapolation_horizon_ms;
  $("extrapolationMaxLeadPercent").value = Number(
    config.extrapolation_max_lead_fraction || 0
  ) * 100;
  $("motionDiagnosticsEnabled").checked = Boolean(config.motion_diagnostics_enabled);
  configDirty = false;
  renderConfigApplyStatus();
}

function renderConfigApplyStatus() {
  const element = $("configApplyStatus");
  if (!element) return;
  element.classList.toggle("dirty", configDirty);
  element.classList.toggle("applied", !configDirty && Boolean(state));
  element.textContent = configDirty
    ? "有未保存修改 · 保存后立即应用到控制台、预览和后台共享内存"
    : (state ? "已应用 · 保存配置会立即重载运行参数；未启动时不会向游戏发送数据" : "配置尚未加载");
}

function renderConfiguredProcessing(config) {
  if (!config) return;
  const filterText = config.one_euro_enabled
    ? `稳定 A 开启 · cutoff ${Number(config.one_euro_min_cutoff || 0).toFixed(2)} / β ${Number(config.one_euro_beta || 0).toFixed(2)}`
    : "稳定 A 关闭";
  const extrapolationText = config.extrapolation_enabled
    ? `补偿 B ${Number(config.extrapolation_horizon_ms || 0).toFixed(0)} ms`
    : "补偿 B 关闭";
  $("previewPostprocess").textContent = `${filterText} · ${extrapolationText}`;
  $("previewMotionMode").textContent = "模式 -";
}

function lightingProfileLabel(name) {
  return name === "reference" ? "参考光照" : name;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[character]));
}

function setLightingProfileSelection(name) {
  for (const id of ["lightingProfile", "calibrationLightingProfile"]) {
    const select = $(id);
    if (Array.from(select.options).some((option) => option.value === name)) {
      select.value = name;
    }
  }
}

function renderLightingProfiles(names, configuredName, inventory = []) {
  const profiles = names.length ? names : ["reference"];
  for (const id of ["lightingProfile", "calibrationLightingProfile"]) {
    const select = $(id);
    const currentOptions = Array.from(select.options).map((option) => option.value);
    if (JSON.stringify(currentOptions) !== JSON.stringify(profiles)) {
      select.innerHTML = profiles.map((name) =>
        `<option value="${name}">${lightingProfileLabel(name)}</option>`
      ).join("");
      select.value = profiles.includes(configuredName) ? configuredName : "reference";
    }
  }
  const activeName = profiles.includes(configuredName) ? configuredName : "reference";
  $("activeLightingProfile").textContent = `当前：${lightingProfileLabel(activeName)}`;
  $("lightingProfileList").textContent = `已保存：${profiles.map(lightingProfileLabel).join("、")}`;
  const profileItems = inventory.length ? inventory : profiles.map((name) => ({
    name, builtin: ["reference", "dark", "bright"].includes(name),
    deletable: name !== "reference", reusable: false, model_present: true,
    sample_frames: { legacy: 0, tasks: 0 }, geometry_matches: true,
  }));
  const inventoryKey = JSON.stringify(profileItems);
  if (inventoryKey !== lightingInventoryKey) {
    lightingInventoryKey = inventoryKey;
    $("lightingProfileItems").innerHTML = profileItems.map((item) => {
      const name = escapeHtml(item.name);
      const frames = item.sample_frames || {};
      const frameText = item.builtin
        ? "完整校准会重新生成"
        : item.reusable
          ? `${frames.legacy || 0}/${frames.tasks || 0} 帧 · 可在完整校准后重训${item.model_present ? "" : " · 当前模型未加载"}`
          : `${frames.legacy || 0}/${frames.tasks || 0} 帧${item.geometry_matches ? "" : " · 几何不匹配"}`;
      return `<div class="profile-item">
        <strong class="profile-name">${escapeHtml(lightingProfileLabel(item.name))}</strong>
        <span class="profile-meta">${frameText}</span>
        <button class="profile-delete" type="button" data-delete-profile="${name}"${item.deletable ? "" : " disabled"}>删除预设</button>
      </div>`;
    }).join("");
  }
}

function metric(label, value) {
  return `<div><dt>${label}</dt><dd>${value ?? "-"}</dd></div>`;
}

function describeRuntimeError(message) {
  const translations = {
    "landmarker produced no face": "未检测到人脸",
    "waiting for Camera2 intrinsics": "正在等待 Camera2 内参"
  };
  return translations[message] || message || "";
}

function renderArtifacts(artifacts) {
  const order = [
    ["legacy_cnn", "O Legacy + CNN"],
    ["tasks_cnn", "N Tasks + CNN"]
  ];
  const renderKey = JSON.stringify({
    models: order.map(([key]) => {
      const item = artifacts.models[key] || {};
      const holdout = item.holdout || {};
      return [key, item.ready, item.compatible, item.schema, item.created_at, holdout.validation_median_deg ?? holdout.median_deg];
    }),
    datasets: ["legacy", "tasks"].map((key) => {
      const item = artifacts.datasets[key] || {};
      return [key, item.ready, item.compatible, item.schema, item.samples, item.targets, item.created_at];
    })
  });
  if (renderKey === artifactRenderKey) return;
  artifactRenderKey = renderKey;
  $("artifactGrid").innerHTML = order.map(([key, label]) => {
    const item = artifacts.models[key] || {};
    const holdout = item.holdout || {};
    const median = holdout.validation_median_deg ?? holdout.median_deg;
    const usable = item.ready && item.compatible !== false;
    return `<div class="artifact ${usable ? "ready" : "missing"}">
      <strong>${label}</strong><span>${usable ? `已训练${median != null ? ` · ${Number(median).toFixed(2)} deg` : ""}` : "需要重新校准"}</span>
    </div>`;
  }).join("");
  $("datasetGrid").innerHTML = ["legacy", "tasks"].map((key) => {
    const item = artifacts.datasets[key] || {};
    const usable = item.ready && item.compatible !== false;
    return `<div class="artifact ${usable ? "ready" : "missing"}"><strong>${key === "legacy" ? "O Legacy" : "N Tasks"}</strong>
      <span>${usable ? `${item.samples || 0} 帧 · ${item.targets || 0} 个采样组` : "需要重新校准"}</span></div>`;
  }).join("");
}

function renderPairing(application = {}) {
  const pairing = application.pairing || {};
  const source = application.phone_source || {};
  const paired = Boolean(pairing.paired_phone_id);
  $("pairingSummary").textContent = paired
    ? `已配对：${pairing.paired_phone_name || pairing.paired_phone_id}`
    : "尚未配对";
  $("pairingDetail").textContent = paired
    ? `电脑 ${pairing.pc_name || "OpenGazeLink"} · 手机地址 ${source.allowed_source_ip || "等待重新发现"}`
    : `发现端口 ${pairing.discovery_port || 5006} · 在手机端点击“发现电脑”`;
  const pending = pairing.pending || [];
  $("pairingCandidates").innerHTML = pending.map((phone) => `
    <div class="profile-item">
      <strong class="profile-name">${escapeHtml(phone.name || "Android phone")}</strong>
      <span class="profile-meta">${escapeHtml(phone.address || "")}</span>
      <button class="button secondary" type="button" data-accept-phone="${escapeHtml(phone.phone_id)}">确认配对</button>
    </div>
  `).join("");
  $("forgetPairingButton").disabled = !paired || actionBusy;
}

function renderFirstRun({ paired, cameraReady, intrinsicsReady, geometryReady, modelReady, inputSource }) {
  const localCamera = inputSource === "windows_camera";
  $("pairingPanel").hidden = localCamera;
  $("setupPairing").querySelector("strong").textContent = localCamera ? "1. PC 摄像头" : "1. 手机连接";
  $("setupCamera").querySelector("strong").textContent = localCamera ? "2. 画面输入" : "2. 画面与内参";
  const update = (id, ready, readyText, missingText) => {
    const element = $(id);
    element.classList.toggle("ready", ready);
    element.classList.toggle("missing", !ready);
    element.querySelector("span").textContent = ready ? readyText : missingText;
  };
  update("setupPairing", paired || cameraReady, "手机已连接", "等待自动配对或手动地址");
  update("setupCamera", cameraReady && intrinsicsReady, "画面与 Camera2 内参有效", "在手机端开始传输");
  update("setupGeometry", geometryReady, "屏幕几何已保存", "填写屏幕尺寸与相机位置并保存");
  update("setupModel", modelReady, "所选模型可运行", "完成首次完整校准");
  if (localCamera) {
    update("setupPairing", true, "PC 摄像头已选择", "选择 PC 摄像头");
    update("setupCamera", cameraReady && intrinsicsReady, "PC 摄像头画面有效", "等待 PC 摄像头画面");
  }
  $("firstRunPanel").hidden = cameraReady && intrinsicsReady && geometryReady && modelReady;
}

function renderStatus(payload) {
  const first = state === null;
  state = payload;
  if (first) applyConfig(payload.config);
  const engine = payload.engine;
  const camera = engine.camera || {};
  const intrinsics = engine.intrinsics || {};
  const cameraReady = Number(camera.width) > 0 && Number(camera.height) > 0;
  if (cameraReady) {
    $("cameraStage").style.aspectRatio = `${Number(camera.width)} / ${Number(camera.height)}`;
  } else {
    $("cameraStage").style.removeProperty("aspect-ratio");
  }
  const input = engine.input || { ready: cameraReady, error: cameraReady ? "" : "未收到手机画面" };
  const geometryReady = payload.geometry?.configured === true;
  const modelKey = `${payload.config.landmarker}_cnn`;
  const selectedModel = payload.artifacts?.models?.[modelKey] || {};
  const modelReady = selectedModel.ready && selectedModel.compatible !== false;
  const intrinsicsReady = payload.config.input_source === "windows_camera"
    ? cameraReady && intrinsics.source === "estimated_windows_camera"
    : Boolean(intrinsics.source && intrinsics.source !== "estimated_frame_center");
  const calibrationState = payload.calibration || {};
  const motionDiagnostics = engine.motion_diagnostics || {};
  const availableLightingProfiles = selectedModel.lighting_profiles?.length
    ? selectedModel.lighting_profiles : ["reference"];
  renderLightingProfiles(
    availableLightingProfiles, payload.config.lighting_profile,
    payload.lighting_profiles || [],
  );
  const startError = input.error || (modelReady ? "" : "所选模型文件缺失或不兼容");
  $("cameraDot").className = `status-dot ${cameraReady ? "ok" : "warn"}`;
  const cameraLabel = payload.config.input_source === "windows_camera" ? "PC 摄像头" : "手机";
  $("cameraSummary").textContent = cameraReady ? `${cameraLabel} ${camera.width}x${camera.height} · ${Number(camera.fps || 0).toFixed(1)} FPS` : `等待${cameraLabel}`;
  if (cameraReady) {
    $("cameraSummary").textContent += payload.config.input_source === "windows_camera"
      ? ` · 覆盖旧帧 ${Number(camera.overwrittenFrames || 0)}`
      : ` · queue ${Number(camera.transportQueueMs || 0).toFixed(1)} ms · decode ${Number(camera.decodeMs || 0).toFixed(1)} ms`;
  }
  $("frameRate").textContent = activePage === "control"
    ? `${Number(camera.fps || 0).toFixed(1)} FPS 连续流`
    : "已暂停";
  $("runtimeDot").className = `status-dot ${engine.tracking ? "ok" : ""}`;
  const hostMode = payload.application?.mode === "runtime" ? "后台" : "控制";
  $("runtimeSummary").textContent = engine.tracking ? `${hostMode} · 共享内存运行中` : `${hostMode} · 未运行`;
  $("runtimeDetail").textContent = startError || describeRuntimeError(engine.error) || `${payload.config.shared_memory_name} · ${Number(engine.inference_ms?.mean || 0).toFixed(2)} ms`;
  const selection = `${payload.config.landmarker === "legacy" ? "O Legacy" : "N Tasks"} + CNN`;
  $("selectionLabel").textContent = selection;
  $("previewMode").textContent = selection;
  renderConfiguredProcessing(payload.config);
  $("motionDiagnosticsStatus").classList.toggle("active", Boolean(motionDiagnostics.active));
  $("motionDiagnosticsStatus").classList.toggle("warning", Number(motionDiagnostics.dropped_records || 0) > 0);
  $("motionDiagnosticsStatus").textContent = motionDiagnostics.active
    ? `正在记录 ${Number(motionDiagnostics.sample_count || 0)} 帧 · 丢弃 ${Number(motionDiagnostics.dropped_records || 0)} · ${motionDiagnostics.path || "正在创建文件"}`
    : (motionDiagnostics.path ? `最近记录 · ${motionDiagnostics.path}` : "未开始记录");
  $("intrinsicsSource").textContent = intrinsics.source || "未收到";
  $("intrinsicsGrid").innerHTML = [
    metric("宽度", intrinsics.width), metric("高度", intrinsics.height),
    metric("fx", intrinsics.fx != null ? Number(intrinsics.fx).toFixed(3) : "-"),
    metric("fy", intrinsics.fy != null ? Number(intrinsics.fy).toFixed(3) : "-"),
    metric("cx", intrinsics.cx != null ? Number(intrinsics.cx).toFixed(3) : "-"),
    metric("cy", intrinsics.cy != null ? Number(intrinsics.cy).toFixed(3) : "-"),
    metric("镜头", intrinsics.sourceMetadata?.lensFacing || "-"),
    metric("旋转", `${intrinsics.rotate ?? camera.rotation ?? "-"} (${camera.rotationSource === "phone" ? "手机" : "手动"})`)
  ].join("");
  renderArtifacts(payload.artifacts);
  renderPairing(payload.application);
  renderFirstRun({
    paired: Boolean(payload.application?.pairing?.paired_phone_id),
    cameraReady, intrinsicsReady, geometryReady, modelReady,
    inputSource: payload.config.input_source,
  });
  $("saveConfigButton").disabled = !configDirty;
  renderConfigApplyStatus();
  $("geometryStatus").textContent = geometryReady
    ? "屏幕与相机位置已确认"
    : "首次校准前必须确认并保存屏幕与相机位置";
  $("geometryStatus").classList.toggle("ready", geometryReady);
  $("startButton").disabled = engine.tracking || !input.ready || !modelReady || !geometryReady || actionBusy;
  $("stopButton").disabled = !engine.tracking;
  $("backgroundButton").disabled = !modelReady || !geometryReady || actionBusy;
  $("startCalibrationButton").disabled = !input.ready || !geometryReady || calibrationState.active || actionBusy;
  $("startLightingAdaptationButton").disabled = !input.ready || !modelReady || !geometryReady
    || !selectedModel.lighting_profiles?.includes("reference") || actionBusy;
  $("applyLightingProfileButton").disabled = !modelReady || actionBusy;
  $("previewStartButton").disabled = ((!input.ready || !modelReady) && !engine.tracking) || actionBusy;
  $("previewStartButton").textContent = actionBusy
    ? "正在启动"
    : (engine.tracking ? (document.fullscreenElement === $("previewPage") ? "停止预览" : "进入全屏") : "启动预览");
  $("calibrationStatus").textContent = calibrationState.active
    ? (calibrationState.phase === "lighting_profile_selection"
      ? "新基座已训练，等待选择保留的光照预设"
      : calibrationState.phase === "lighting_profile_training"
        ? "正在新基座上重训光照适配层"
        : calibrationState.phase === "head_pose"
      ? `交叉头姿 ${calibrationState.pose_index || 0}/${calibrationState.pose_total || 20}`
      : calibrationState.phase === "light_anchor"
        ? `光照锚点 ${calibrationState.light_index || 0}/${calibrationState.light_total || 10}`
        : `静态眼动 ${calibrationState.index || 0}/${calibrationState.total || 25}`)
    : (!geometryReady
      ? "请先在运行控制中确认并保存屏幕与相机位置"
      : (input.error || "25 点静态眼动 + 10 个光照锚点 + 4 头姿 × 5 注视点"));
  if (calibrationState.state === "awaiting_profiles") {
    showLightingProfileSelection(
      calibrationState.available_lighting_profiles || [], "calibration",
    );
  } else if (!calibrationState.active && !actionBusy) {
    const recoverable = (payload.lighting_profiles || []).filter(
      (item) => item.reusable && !item.model_present
    ).map((item) => ({name: item.name, sample_frames: item.sample_frames || {}}));
    const recoveryKey = `restore:${JSON.stringify(recoverable)}`;
    if (recoverable.length && recoveryKey !== dismissedRecoveryKey) {
      showLightingProfileSelection(recoverable, "restore");
    } else if (!recoverable.length && !$("lightingSelectionModal").hidden) {
      hideLightingProfileSelection();
    }
  }
  if (activePage === "preview" && !engine.tracking) {
    $("previewState").textContent = startError || "等待启动";
  }
  syncStreams();
}

async function refreshStatus() {
  if (statusRequestPending) return;
  statusRequestPending = true;
  try { renderStatus(await request("/api/status")); }
  catch (error) { toast(error.message); }
  finally {
    statusRequestPending = false;
    scheduleStatusRefresh();
  }
}

async function refreshWindowsCameras() {
  const button = $("refreshWindowsCamerasButton");
  button.disabled = true;
  try {
    const payload = await request("/api/windows-cameras");
    const select = $("windowsCameraIndex");
    const selected = String(select.value || state?.config?.windows_camera_index || 0);
    const cameras = payload.cameras || [];
    select.innerHTML = cameras.length
      ? cameras.map((camera) => `<option value="${Number(camera.index)}">${escapeHtml(camera.name || `Windows camera ${camera.index}`)} (${Number(camera.width || 0)}x${Number(camera.height || 0)})</option>`).join("")
      : `<option value="${selected}">未发现摄像头（设备 ${selected}）</option>`;
    select.value = Array.from(select.options).some((option) => option.value === selected)
      ? selected : String(cameras[0]?.index ?? selected);
  } finally {
    button.disabled = false;
  }
}

function scheduleStatusRefresh() {
  window.clearTimeout(statusTimer);
  statusTimer = 0;
  if (document.hidden || (activePage === "preview" && state?.engine?.tracking)) return;
  statusTimer = window.setTimeout(refreshStatus, state?.engine?.input?.ready ? 1000 : 500);
}

function startFrameStream() {
  if (frameStreamActive || Number(state?.engine?.camera?.width || 0) <= 0) return;
  frameStreamActive = true;
  $("cameraFrame").hidden = false;
  $("cameraPlaceholder").hidden = true;
  $("cameraFrame").src = `/api/frame.mjpg?t=${Date.now()}`;
}

function stopFrameStream() {
  if (!frameStreamActive) return;
  frameStreamActive = false;
  $("cameraFrame").removeAttribute("src");
  $("cameraFrame").hidden = true;
  $("cameraPlaceholder").hidden = false;
}

$("cameraFrame").addEventListener("error", () => {
  frameStreamActive = false;
  $("cameraFrame").hidden = true;
  $("cameraPlaceholder").hidden = false;
});

function syncStreams() {
  const cameraReady = Number(state?.engine?.camera?.width || 0) > 0;
  if (!document.hidden && activePage === "control" && cameraReady) startFrameStream();
  else stopFrameStream();
  if (!document.hidden && activePage === "preview") startGazeStream();
  else stopGazeStream();
}

function showCalibrationTarget() {
  const isPose = calibration.phase === "head_pose";
  const isQuickLight = calibration.phase === "light_adaptation";
  const isLight = calibration.phase === "light_anchor" || isQuickLight;
  const list = isPose ? calibration.poseTargets : (calibration.phase === "light_anchor" ? calibration.lightTargets : calibration.targets);
  const index = isPose ? calibration.poseIndex : (calibration.phase === "light_anchor" ? calibration.lightIndex : calibration.index);
  const target = list[index];
  if (!target) return;
  const width = Math.max(1, state.config.screen_width - 1);
  const height = Math.max(1, state.config.screen_height - 1);
  $("calibrationTarget").style.left = `${target.x / width * 100}%`;
  $("calibrationTarget").style.top = `${target.y / height * 100}%`;
  $("calibrationTarget").classList.toggle("pose-target", isPose);
  if (isPose) {
    $("calibrationMessage").textContent = "交叉头姿 " + (calibration.poseIndex + 1) + "/" + calibration.poseTargets.length + " · " + target.pose_instruction + "，眼睛盯住蓝点并保持头部不动";
  } else {
    const lighting = target.lighting || {};
    const lightText = isLight ? (" · 光照" + (lighting.name || "anchor")) : " · 参考光照";
    const current = index;
    $("calibrationMessage").textContent = "注视点 " + (current + 1) + "/" + list.length + " · 盯住蓝点，点击采集" + lightText;
  }
  const level = target.lighting?.start ?? 0.42;
  setCalibrationLight(level);
  $("calibrationTarget").classList.remove("collecting");
}

function setCalibrationLight(level) {
  const clamped = Math.max(0, Math.min(1, Number(level) || 0));
  const value = Math.round(255 * clamped);
  $("calibrationOverlay").style.backgroundColor = `rgb(${value},${value},${value})`;
}

function animateCalibrationLight(profile) {
  if (!profile || profile.mode !== "transition") {
    setCalibrationLight(profile?.end ?? profile?.start ?? 0.22);
    return Promise.resolve();
  }
  const duration = Math.max(300, Number(profile.duration_ms || 1800));
  const started = performance.now();
  return new Promise((resolve) => {
    const tick = (now) => {
      const progress = Math.min(1, (now - started) / duration);
      setCalibrationLight(Number(profile.start) + (Number(profile.end) - Number(profile.start)) * progress);
      if (progress < 1) window.requestAnimationFrame(tick);
      else resolve();
    };
    window.requestAnimationFrame(tick);
  });
}

function showLightingProfileSelection(items, mode = "calibration") {
  const profiles = Array.isArray(items) ? items : [];
  lightingSelectionMode = mode;
  const key = `${mode}:${JSON.stringify(profiles)}`;
  if (key !== lightingSelectionKey) {
    lightingSelectionKey = key;
    $("lightingSelectionItems").innerHTML = profiles.map((item) => {
      const name = escapeHtml(item.name);
      const frames = item.sample_frames || {};
      return `<div class="selection-item">
        <label><input type="checkbox" data-lighting-selection="${name}" checked><strong>${escapeHtml(lightingProfileLabel(item.name))}</strong></label>
        <span>${frames.legacy || 0}/${frames.tasks || 0} 帧</span>
      </div>`;
    }).join("");
  }
  $("lightingSelectionTitle").textContent = mode === "restore"
    ? "恢复光照预设到当前模型" : "保留光照预设";
  $("lightingSelectionDescription").textContent = mode === "restore"
    ? "检测到上次完整校准前保存的光照样本。选择要在当前新基座上补训的适配层，无需重新进行完整校准。"
    : "新基座已经训练完成。选择要用旧样本在新模型上重新训练的自定义光照适配层。";
  $("abandonCalibrationButton").textContent = mode === "restore" ? "稍后" : "放弃本次校准";
  $("completeLightingSelectionButton").textContent = mode === "restore"
    ? "重训所选适配层" : "保留所选并完成";
  $("lightingSelectionModal").hidden = false;
}

function hideLightingProfileSelection() {
  $("lightingSelectionModal").hidden = true;
  lightingSelectionKey = "";
}

async function completeLightingProfileSelection() {
  if (actionBusy) return;
  actionBusy = true;
  const selected = Array.from(
    $("lightingSelectionItems").querySelectorAll("[data-lighting-selection]:checked")
  ).map((input) => input.dataset.lightingSelection);
  $("completeLightingSelectionButton").disabled = true;
  $("completeLightingSelectionButton").textContent = "正在重训适配层";
  try {
    const endpoint = lightingSelectionMode === "restore"
      ? "/api/lighting-profiles/retrain" : "/api/calibration/lighting-profiles";
    const result = await post(endpoint, {
      profile_names: selected,
    });
    hideLightingProfileSelection();
    if (lightingSelectionMode === "calibration") {
      calibration.active = false;
      calibration.phase = "idle";
    }
    await refreshStatus();
    const retained = result.retained_lighting_profiles || result.profile_names || [];
    toast(retained.length
      ? `新模型已保留：${retained.map(lightingProfileLabel).join("、")}`
      : "新模型已保存，未保留自定义光照预设");
  } finally {
    actionBusy = false;
    $("completeLightingSelectionButton").disabled = false;
    $("completeLightingSelectionButton").textContent = lightingSelectionMode === "restore"
      ? "重训所选适配层" : "保留所选并完成";
    refreshStatus();
  }
}

async function abandonPendingCalibration() {
  if (actionBusy) return;
  if (lightingSelectionMode === "restore") {
    dismissedRecoveryKey = lightingSelectionKey;
    hideLightingProfileSelection();
    return;
  }
  if (!window.confirm("放弃这次已经训练但尚未提交的完整校准？当前模型不会被替换。")) return;
  actionBusy = true;
  try {
    await post("/api/calibration/cancel");
    hideLightingProfileSelection();
    calibration.active = false;
    calibration.phase = "idle";
    toast("已放弃本次完整校准，原模型保持不变");
  } finally {
    actionBusy = false;
    refreshStatus();
  }
}

async function startCalibration() {
  if (actionBusy) return;
  if (!state?.engine?.input?.ready) throw new Error(state?.engine?.input?.error || "输入尚未准备好");
  actionBusy = true;
  $("calibrationOverlay").hidden = false;
  $("calibrationTarget").hidden = true;
  $("calibrationMessage").textContent = "正在启动校准";
  try {
    await $("calibrationOverlay").requestFullscreen();
    await post("/api/config", configPayload());
    const result = await post("/api/calibration/start");
    if (result.calibration_schema !== "crossed-head-pose-4x5-v1") {
      await post("/api/calibration/cancel").catch(() => {});
      throw new Error("前端与校准后端版本不一致，请重启电脑端控制程序");
    }
    calibration = {
      active: true, phase: result.phase || "static",
      targets: result.targets, lightTargets: result.light_targets || [],
      poseTargets: result.pose_targets || [], index: 0, lightIndex: 0,
      poseIndex: 0, busy: false
    };
    $("calibrationTarget").hidden = false;
    showCalibrationTarget();
  } catch (error) {
    $("calibrationOverlay").hidden = true;
    if (document.fullscreenElement) await document.exitFullscreen().catch(() => {});
    throw error;
  } finally {
    actionBusy = false;
    refreshStatus();
  }
}

async function startLightingAdaptation() {
  if (actionBusy) return;
  if (!state?.engine?.input?.ready) throw new Error(state?.engine?.input?.error || "输入尚未准备好");
  actionBusy = true;
  $("calibrationOverlay").hidden = false;
  $("calibrationTarget").hidden = true;
  $("calibrationMessage").textContent = "正在启动光照适配";
  try {
    await $("calibrationOverlay").requestFullscreen();
    await post("/api/config", configPayload());
    const result = await post("/api/lighting-adaptation/start", {
      profile_name: $("lightingAdaptationName").value,
      screen_level: Number($("lightingAdaptationLevel").value || 0.42)
    });
    $("lightingAdaptationName").value = result.profile_name || $("lightingAdaptationName").value;
    calibration = {
      active: true, phase: "light_adaptation", targets: result.targets || [],
      lightTargets: [], poseTargets: [], index: 0, lightIndex: 0,
      poseIndex: 0, busy: false
    };
    $("calibrationTarget").hidden = false;
    showCalibrationTarget();
  } catch (error) {
    $("calibrationOverlay").hidden = true;
    if (document.fullscreenElement) await document.exitFullscreen().catch(() => {});
    throw error;
  } finally {
    actionBusy = false;
    refreshStatus();
  }
}

async function applyLightingProfile() {
  if (actionBusy) return;
  actionBusy = true;
  try {
    setLightingProfileSelection($("calibrationLightingProfile").value);
    await post("/api/config", configPayload());
    await refreshStatus();
    toast(`已应用光照预设：${lightingProfileLabel(state.config.lighting_profile)}`);
  } finally {
    actionBusy = false;
    refreshStatus();
  }
}

async function deleteLightingProfile(profileName) {
  if (actionBusy) return;
  if (!window.confirm(`删除光照预设“${profileName}”？活动数据会从当前数据集移除，原始诊断图片仍保留。`)) return;
  actionBusy = true;
  try {
    await post("/api/lighting-profiles/delete", { profile_name: profileName });
    if ($("lightingProfile").value === profileName) setLightingProfileSelection("reference");
    await refreshStatus();
    toast(`光照预设已删除：${profileName}`);
  } finally {
    actionBusy = false;
    refreshStatus();
  }
}

async function startPreview() {
  if (actionBusy) return;
  if (!state?.engine?.input?.ready && !state?.engine?.tracking) {
    throw new Error(state?.engine?.input?.error || "输入尚未准备好");
  }
  if (state?.engine?.tracking && document.fullscreenElement === $("previewPage")) {
    await post("/api/tracking/stop");
    await document.exitFullscreen();
    await refreshStatus();
    return;
  }
  actionBusy = true;
  $("previewStartButton").disabled = true;
  $("previewStartButton").textContent = "正在启动";
  try {
    await $("previewPage").requestFullscreen();
    if (configDirty || !state?.engine?.tracking) {
      await post("/api/config", configPayload());
      configDirty = false;
    }
    if (!state?.engine?.tracking) {
      await post("/api/tracking/start");
    }
    await refreshStatus();
  } catch (error) {
    if (document.fullscreenElement) await document.exitFullscreen().catch(() => {});
    throw error;
  } finally {
    actionBusy = false;
    refreshStatus();
  }
}

async function sampleTarget() {
  if (!calibration.active || calibration.busy) return;
  calibration.busy = true;
  $("calibrationTarget").classList.add("collecting");
  const phase = calibration.phase;
  const isPose = phase === "head_pose";
  const isLight = phase === "light_anchor";
  const isQuickLight = phase === "light_adaptation";
  const list = isPose ? calibration.poseTargets : (isLight ? calibration.lightTargets : calibration.targets);
  const index = isPose ? calibration.poseIndex : (isLight ? calibration.lightIndex : calibration.index);
  const target = list[index];
  $("calibrationMessage").textContent = isPose
    ? ((index + 1) + "/" + list.length + " · 正在采集固定头姿下的眼动")
    : ((index + 1) + "/" + list.length + " · 正在采集" + ((isLight || isQuickLight) ? "光照锚点" : "静态眼动"));
  try {
    if (isPose) {
      const result = await post("/api/calibration/pose", { index });
      calibration.poseIndex = result.pose_index;
      if (calibration.poseIndex >= calibration.poseTargets.length) {
        $("calibrationTarget").hidden = true;
        $("calibrationMessage").textContent = "正在训练 O/N 两套 CNN";
        const finished = await post("/api/calibration/finish");
        await closeCalibration(false);
        if (finished.requires_lighting_profile_selection) {
          calibration.active = true;
          calibration.phase = "lighting_profile_selection";
          showLightingProfileSelection(finished.available_lighting_profiles || []);
        } else {
          toast("新校准数据已训练并保存");
          await refreshStatus();
        }
      } else {
        showCalibrationTarget();
      }
    } else {
      const profile = target.lighting || {};
      const animation = animateCalibrationLight(profile);
      const result = await post("/api/calibration/sample", { index });
      await animation;
      if (isLight) calibration.lightIndex = result.light_index;
      else calibration.index = result.index;
      calibration.phase = result.phase || phase;
      if (isQuickLight && calibration.index >= calibration.targets.length) {
        $("calibrationTarget").hidden = true;
        $("calibrationMessage").textContent = "正在训练光照适配器";
        const finished = await post("/api/calibration/finish");
        await closeCalibration(false);
        await refreshStatus();
        setLightingProfileSelection(finished.profile_name);
        await post("/api/config", configPayload());
        toast(`光照配置 ${finished.profile_name} 已保存`);
        await refreshStatus();
        return;
      }
      if (calibration.phase === "light_anchor" && !isLight) calibration.lightIndex = 0;
      if (calibration.phase === "head_pose") {
        calibration.poseIndex = 0;
        showCalibrationTarget();
      } else {
        showCalibrationTarget();
      }
    }
  } catch (error) {
    $("calibrationMessage").textContent = `采集失败 · ${error.message}`;
    $("calibrationTarget").classList.remove("collecting");
  } finally {
    calibration.busy = false;
  }
}

async function closeCalibration(cancel = true) {
  if (cancel && calibration.active) await post("/api/calibration/cancel").catch(() => {});
  calibration.active = false;
  calibration.phase = "idle";
  $("calibrationOverlay").hidden = true;
  $("calibrationOverlay").style.backgroundColor = "#050605";
  $("calibrationTarget").hidden = false;
  if (document.fullscreenElement) await document.exitFullscreen().catch(() => {});
}

function placeDot(element, point, gaze) {
  if (!point || !gaze.valid) { element.hidden = true; return; }
  element.hidden = false;
  element.style.left = `${point[0] / Math.max(1, state.config.screen_width - 1) * 100}%`;
  element.style.top = `${point[1] / Math.max(1, state.config.screen_height - 1) * 100}%`;
}

function renderGaze(gaze) {
  placeDot($("rightDot"), gaze.right, gaze);
  placeDot($("leftDot"), gaze.left, gaze);
  placeDot($("combinedDot"), gaze.combined, gaze);
  const now = performance.now();
  gazeArrivalTimes.push(now);
  gazeArrivalTimes = gazeArrivalTimes.filter((value) => value >= now - 1000);
  const gazeFps = gazeArrivalTimes.length >= 2
    ? (gazeArrivalTimes.length - 1) * 1000 / (gazeArrivalTimes[gazeArrivalTimes.length - 1] - gazeArrivalTimes[0])
    : 0;
  const waitingMessage = state?.engine?.input?.error
    || describeRuntimeError(gaze.error || state?.engine?.error)
    || (state?.engine?.tracking ? "等待有效注视输出" : "等待启动");
  const extrapolation = gaze.postprocess?.extrapolation_state;
  if (gaze.postprocess) {
    const filterText = gaze.postprocess.one_euro ? "稳定 A 开启" : "稳定 A 关闭";
    const extrapolationText = gaze.postprocess.extrapolation
      ? `补偿 B ${Number(gaze.postprocess.extrapolation_horizon_ms || 0).toFixed(0)} ms`
      : "补偿 B 关闭";
    $("previewPostprocess").textContent = `${filterText} · ${extrapolationText}`;
  }
  const modeLabels = {
    fixation: "固定注视 · 不外推",
    continuous_motion: "连续运动 · 短期外推",
    jump_or_landing: "跳变/落地 · 跟随最新点",
    disabled: "外推未启用",
  };
  $("previewMotionMode").textContent = `模式 ${modeLabels[extrapolation?.mode] || "等待"}`;
  $("previewState").textContent = gaze.valid
    ? `${gazeFps.toFixed(1)} FPS · 推理 ${Number(gaze.processing_ms || 0).toFixed(1)} ms`
    : waitingMessage;
}

function startGazeStream() {
  if (gazeStream) return;
  gazeArrivalTimes = [];
  const stream = new EventSource("/api/gaze/stream");
  gazeStream = stream;
  stream.onmessage = (event) => {
    if (stream !== gazeStream) return;
    try { renderGaze(JSON.parse(event.data)); }
    catch (error) { $("previewState").textContent = `注视数据错误 · ${error.message}`; }
  };
  stream.onerror = () => {
    if (stream === gazeStream && activePage === "preview") {
      $("previewState").textContent = "注视数据连接中断";
    }
  };
}

function stopGazeStream() {
  if (!gazeStream) return;
  const stream = gazeStream;
  gazeStream = null;
  stream.close();
  gazeArrivalTimes = [];
}

document.querySelectorAll(".nav-button").forEach((button) => button.addEventListener("click", () => setPage(button.dataset.page)));
$("pairingCandidates").addEventListener("click", (event) => {
  const button = event.target.closest("[data-accept-phone]");
  if (!button) return;
  post("/api/pairing/accept", { phone_id: button.dataset.acceptPhone })
    .then(refreshStatus)
    .catch((error) => toast(error.message));
});
$("forgetPairingButton").addEventListener("click", () => {
  post("/api/pairing/forget").then(refreshStatus).catch((error) => toast(error.message));
});
$("lightingProfile").addEventListener("change", () => setLightingProfileSelection($("lightingProfile").value));
$("calibrationLightingProfile").addEventListener("change", () => setLightingProfileSelection($("calibrationLightingProfile").value));
$("lightingProfileItems").addEventListener("click", (event) => {
  const button = event.target.closest("[data-delete-profile]");
  if (button) deleteLightingProfile(button.dataset.deleteProfile).catch((error) => toast(error.message));
});
$("configForm").addEventListener("input", () => { configDirty = true; renderConfigApplyStatus(); });
$("configForm").addEventListener("change", () => { configDirty = true; renderConfigApplyStatus(); });
$("saveConfigButton").addEventListener("click", async () => {
  try {
    await post("/api/config", configPayload());
    configDirty = false;
    toast("配置已保存并立即应用");
    await refreshStatus();
  } catch (error) { toast(error.message); }
});
$("refreshWindowsCamerasButton").addEventListener("click", () => {
  refreshWindowsCameras().catch((error) => toast(error.message));
});
$("applyWindowsCameraButton").addEventListener("click", async () => {
  const button = $("applyWindowsCameraButton");
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = "正在打开摄像头...";
  try {
    await post("/api/config", configPayload());
    configDirty = false;
    toast("摄像头设置已保存并应用");
    await refreshStatus();
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
});
$("startButton").addEventListener("click", async () => {
  try {
    await post("/api/config", configPayload());
    configDirty = false;
    await post("/api/tracking/start");
    await refreshStatus();
  } catch (error) { toast(error.message); }
});
$("stopButton").addEventListener("click", async () => { await post("/api/tracking/stop"); await refreshStatus(); });
$("exitButton").addEventListener("click", async () => {
  try {
    await post("/api/application/exit");
    document.body.innerHTML = '<main class="background-confirmation"><h1>OpenGazeLink 已退出</h1><p>现在可以关闭此页面。</p></main>';
  } catch (error) {
    toast(error.message);
  }
});
$("backgroundButton").addEventListener("click", async () => {
  try {
    await post("/api/config", configPayload());
    configDirty = false;
    await post("/api/application/background");
    document.body.innerHTML = '<main class="background-confirmation"><h1>OpenGazeLink 已转入后台运行</h1><p>可以关闭此页面。再次打开“OpenGazeLink 控制中心”即可恢复控制页面。</p></main>';
  } catch (error) {
    toast(error.message);
  }
});
$("previewStartButton").addEventListener("click", () => startPreview().catch((error) => toast(error.message)));
$("startCalibrationButton").addEventListener("click", () => startCalibration().catch((error) => toast(error.message)));
$("startLightingAdaptationButton").addEventListener("click", () => startLightingAdaptation().catch((error) => toast(error.message)));
$("applyLightingProfileButton").addEventListener("click", () => applyLightingProfile().catch((error) => toast(error.message)));
$("completeLightingSelectionButton").addEventListener("click", () => completeLightingProfileSelection().catch((error) => toast(error.message)));
$("abandonCalibrationButton").addEventListener("click", () => abandonPendingCalibration().catch((error) => toast(error.message)));
$("calibrationTarget").addEventListener("click", sampleTarget);
$("cancelCalibrationButton").addEventListener("click", () => closeCalibration(true));
document.addEventListener("fullscreenchange", () => { if (!document.fullscreenElement && calibration.active) closeCalibration(true); });
document.addEventListener("fullscreenchange", () => { if (activePage === "preview") refreshStatus(); });
document.addEventListener("visibilitychange", () => {
  syncStreams();
  if (document.hidden) {
    window.clearTimeout(statusTimer);
    statusTimer = 0;
  } else {
    refreshStatus();
  }
});

refreshStatus();
