const $ = id => document.getElementById(id);
let state = null;
let activePage = "control";
let actionBusy = false;
let configDirty = false;
let toastTimer = 0;
let statusTimer = 0;
let statusRequestPending = false;
let applicationClosed = false;
let frameStreamActive = false;
let gazeStream = null;
let gazeStreamIncludesPerformance = false;
let gazeArrivalTimes = [];
let latencyHistory = [];
let lastLatencySequence = null;
let lastPerformance = null;

async function request(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store", ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) }
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || String(response.status));
  return payload;
}
function post(path, body = {}) {
  return request(path, { method: "POST", body: JSON.stringify(body) });
}
function toast(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = describeRuntimeError(message);
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 5000);
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}
function describeRuntimeError(message) {
  const messages = {
    "landmarker produced no face": ["未检测到人脸，请看向相机", "No face detected. Look toward the camera."],
    "waiting for Camera2 intrinsics": ["等待手机发送相机内参", "Waiting for camera intrinsics from the phone"],
    "waiting for phone frames": ["等待手机画面", "Waiting for phone frames"],
    "calibration is active": ["请先结束当前校准", "Finish the current calibration first"],
    "cannot change configuration during calibration": ["校准期间不能修改设置", "Settings cannot change during calibration"],
    "Failed to fetch": ["无法连接电脑服务，请确认程序正在运行", "Cannot connect to the PC service. Check that it is running."],
  };
  const pair = messages[message];
  return pair ? tr(...pair) : (message || "");
}
function numericInput(id) {
  const input = $(id);
  const value = input.valueAsNumber;
  const minimum = input.min === "" ? -Infinity : Number(input.min);
  const maximum = input.max === "" ? Infinity : Number(input.max);
  const step = Number(input.step || 1), origin = Number(input.min || 0);
  // Commands temporarily disable controls; native checkValidity then skips them.
  const stepValid = input.step === "any" || Math.abs((value - origin) / step - Math.round((value - origin) / step)) < 1e-6;
  if (!input.checkValidity() || !Number.isFinite(value) || value < minimum || value > maximum || !stepValid) {
    input.closest("details")?.setAttribute("open", "");
    input.focus();
    throw new Error(tr("请填写有效数值：", "Enter a valid value: ") + (input.closest("label")?.querySelector("span")?.textContent || id));
  }
  return value;
}
const numberFields = {
  udpPort: "udp_port", windowsCameraWidth: "windows_camera_width",
  windowsCameraHeight: "windows_camera_height", windowsCameraFps: "windows_camera_fps",
  windowsCameraFov: "windows_camera_fov_x_degrees", screenWidth: "screen_width",
  screenHeight: "screen_height", screenDiagonal: "screen_diagonal_inches",
  cameraOffsetX: "camera_offset_x_cm", cameraOffsetY: "camera_offset_y_cm", cameraOffsetZ: "camera_offset_z_cm"
};
const textFields = {
  inputSource: "input_source", rotate: "rotate", windowsCameraIndex: "windows_camera_index",
  windowsCameraBackend: "windows_camera_backend", sharedMemory: "shared_memory_name"
};
function configPayload() {
  if (!state) throw new Error(tr("配置尚未加载", "Settings have not loaded"));
  const payload = { geometry_configured: true };
  for (const [id, field] of Object.entries(numberFields)) payload[field] = numericInput(id);
  for (const [id, field] of Object.entries(textFields)) payload[field] = $(id).value;
  payload.windows_camera_index = Number(payload.windows_camera_index);
  if (!Number.isInteger(payload.windows_camera_index) || payload.windows_camera_index < 0) {
    throw new Error(tr("请选择摄像头", "Select a camera"));
  }
  if (!payload.shared_memory_name.trim()) throw new Error(tr("共享内存名称不能为空", "Shared memory name is required"));
  payload.mirror = $("mirror").checked;
  // This is the only prediction preference. Stability remains automatic.
  payload.event_temporal_enabled = $("eventTemporalEnabled").checked;
  return payload;
}
function applyConfig(config) {
  for (const [id, field] of Object.entries(numberFields)) $(id).value = config[field];
  const cameraIndex = String(config.windows_camera_index ?? 0);
  if (!Array.from($("windowsCameraIndex").options).some(o => o.value === cameraIndex)) {
    $("windowsCameraIndex").add(new Option("Camera " + cameraIndex, cameraIndex));
  }
  for (const [id, field] of Object.entries(textFields)) $(id).value = config[field];
  $("mirror").checked = Boolean(config.mirror);
  $("eventTemporalEnabled").checked = config.event_temporal_enabled !== false;
  $("windowsCameraSettings").hidden = config.input_source !== "windows_camera";
  configDirty = false;
}
function renderConfigApplyStatus() {
  $("configApplyStatus").classList.toggle("dirty", configDirty);
  $("configApplyStatus").textContent = !state ? tr("正在连接…", "Connecting…")
    : configDirty ? tr("有未保存的修改", "You have unsaved changes")
    : !state.config.geometry_configured ? tr("请确认屏幕与相机位置，再保存", "Confirm screen and camera position, then save")
    : tr("设置已保存", "Settings saved");
  $("saveConfigButton").disabled = !state || actionBusy || Boolean(state?.calibration?.active)
    || (!configDirty && Boolean(state?.config?.geometry_configured));
}
async function saveConfig() {
  const result = await post("/api/config", configPayload());
  applyConfig(result.config);
  if (state) state.config = result.config;
}
function setPage(page) {
  if (!["control", "calibration", "preview", "performance"].includes(page)) return;
  activePage = page;
  document.querySelectorAll(".nav-button").forEach(button => {
    const selected = button.dataset.page === page;
    button.classList.toggle("active", selected);
    if (selected) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
  });
  document.querySelectorAll(".page").forEach(element => element.classList.toggle("active", element.id === page + "Page"));
  if (state) renderStatus(state);
  syncStreams();
  refreshStatus();
}
function modelReady(payload) {
  const model = payload.artifacts?.models?.tasks_conditioned_video || {};
  return Boolean(model.ready && model.compatible !== false);
}
function readinessMessage(payload, needsModel = true) {
  if (!payload.engine?.input?.ready) return tr("请先连接相机并开始传输画面。", "Connect a camera and start streaming first.");
  if (!payload.geometry?.configured) return tr("请在运行页确认屏幕与相机位置，并保存设置。", "Confirm and save the screen and camera position on Connect & run.");
  if (needsModel && !modelReady(payload)) return tr("请先完成第一阶段位置校准。", "Complete stage 1 position calibration first.");
  return "";
}
function renderPairing(application = {}) {
  const pairing = application.pairing || {};
  const paired = Boolean(pairing.paired_phone_id);
  $("pairingSummary").textContent = paired ? tr("已配对", "Paired") : tr("未配对", "Not paired");
  $("pairingDetail").textContent = paired
    ? (pairing.paired_phone_name || "Android") + " · " + (application.phone_source?.allowed_source_ip || tr("等待手机", "Waiting for phone"))
    : tr("手机与电脑接入同一网络或 USB 共享网络，在手机端点击“发现电脑”。", "Use the same network or USB tethering, then tap “Find PC” on the phone.");
  $("pairingCandidates").innerHTML = (pairing.pending || []).map(phone =>
    '<div class="pairing-item"><div><strong>' + escapeHtml(phone.name || "Android") + '</strong><br><span>' +
    escapeHtml(phone.address || "") + '</span></div><button type="button" class="button secondary" data-accept-phone="' +
    escapeHtml(phone.phone_id) + '">' + tr("确认配对", "Pair") + "</button></div>"
  ).join("");
  $("forgetPairingButton").disabled = !paired || actionBusy || Boolean(state?.calibration?.active);
}
function renderStability(payload) {
  const profile = payload.stability || {};
  $("stabilityStatus").textContent = profile.calibrated
    ? tr("自动稳定 · 已使用第二阶段采集结果", "Automatic stability · using stage 2 data")
    : tr("自动稳定 · 使用默认判断，可在第二阶段完善", "Automatic stability · defaults active; refine in stage 2");
}
function renderCalibration(calibration = {}) {
  const phases = {
    unified_capture: ["正在采集", "Capturing"],
    unified_training: ["正在准备训练", "Preparing training"],
    unified_replay: ["正在处理采集画面", "Processing captured frames"],
    unified_spatial: ["正在训练位置模型", "Training position model"],
    unified_prediction: ["正在检查时序", "Checking temporal behavior"],
    video_alignment: ["正在对齐采集数据", "Aligning captured data"],
    personal_eye_and_binocular_fusion: ["正在准备个人双眼模型训练", "Preparing personal binocular training"],
    video_joint_current_temporal_prediction: ["正在导出位置模型", "Exporting position model"],
    unified_complete: ["位置校准完成，新模型已启用", "Position calibration complete; new model is active"],
    unified_kept_existing: ["已完成检查，继续使用原模型", "Evaluation complete; keeping the existing model"],
    event_replay: ["正在检查眼跳与注视", "Evaluating saccades and fixations"],
    event_evaluation_complete: ["第二阶段完成，位置模型保持不变", "Stage 2 complete; position model unchanged"],
    idle: ["可以开始校准", "Ready to calibrate"],
  };
  const phase = calibration.state === "paused" ? tr("采集已暂停", "Capture paused")
    : calibration.state === "cancelled" ? tr("本次任务已结束，数据已保留", "Run ended; data kept")
    : tr(...(phases[calibration.phase] || [calibration.phase || "可以开始校准", calibration.phase || "Ready to calibrate"]));
  $("videoStatus").textContent = describeRuntimeError(calibration.error) || phase +
    (calibration.valid_frames ? " · " + calibration.valid_frames + tr(" 个有效帧", " valid frames") : "");
  const training = calibration.state === "training";
  const bar = $("trainingProgress"), progress = calibration.progress || {};
  bar.hidden = !training;
  if (training && progress.total > 0) {
    bar.max = progress.total; bar.value = progress.completed;
    $("videoStatus").textContent = progress.stage === "replay"
      ? tr(`回放处理：${progress.completed}/${progress.total} 段 · ${progress.workers} 路并行`, `Replay: ${progress.completed}/${progress.total} sequences · ${progress.workers} workers`)
      : tr(`位置模型训练：第 ${progress.completed}/${progress.total} 轮`, `Position model: epoch ${progress.completed}/${progress.total}`);
  } else bar.removeAttribute("value");
  const model = state?.artifacts?.models?.tasks_conditioned_video || {};
  const created = model.created_at ? new Date(model.created_at) : null;
  const stamp = created && !Number.isNaN(created.getTime()) ? created.toLocaleString(language === "en" ? "en-US" : "zh-CN") : tr("时间未知", "Unknown time");
  $("activeModelInfo").textContent = model.ready
    ? tr("当前模型训练于：", "Current model trained: ") + stamp + (model.compatible === false ? tr("（当前设置不兼容）", " (incompatible with current settings)") : "")
    : tr("尚无已训练的位置模型", "No trained position model yet");
}
function renderStatus(payload) {
  if (applicationClosed) return;
  const first = state === null;
  state = payload;
  if (first || (!configDirty && !actionBusy)) applyConfig(payload.config);
  if (first && !payload.geometry?.configured) $("geometrySettings").open = true;
  const engine = payload.engine || {};
  const camera = engine.camera || {};
  const calibration = payload.calibration || {};
  const busy = actionBusy || Boolean(calibration.active);
  const ready = modelReady(payload);
  const inputReady = Boolean(engine.input?.ready);
  const geometryReady = Boolean(payload.geometry?.configured);
  $("runtimeDot").classList.toggle("ok", Boolean(engine.tracking));
  $("runtimeDot").classList.toggle("warn", Boolean(engine.error));
  $("runtimeSummary").textContent = engine.tracking ? tr("运行中", "Running") : tr("已停止", "Stopped");
  $("runtimeDetail").textContent = describeRuntimeError(engine.error) || (engine.tracking
    ? tr("正在向共享内存输出注视位置，关闭网页也会继续运行。", "Sending gaze to shared memory. Tracking continues when this page is closed.")
    : readinessMessage(payload) || tr("准备就绪，可以启动追踪。", "Ready to start tracking."));
  const width = Number(camera.width) || 0, height = Number(camera.height) || 0;
  $("cameraSummary").textContent = width && height
    ? width + " × " + height + " · " + (inputReady ? tr("画面已连接", "Camera connected") : tr("等待新画面", "Waiting for a fresh frame"))
    : tr("相机尚未连接", "No camera connected");
  $("frameRate").textContent = Number(camera.fps || camera.receiveFps || 0).toFixed(1) + " FPS";
  $("pairingPanel").hidden = payload.config.input_source === "windows_camera";
  const hint = readinessMessage(payload);
  $("firstRunPanel").hidden = !hint;
  $("firstRunPanel").textContent = hint;
  $("calibrationReadiness").textContent = readinessMessage(payload, false) ||
    (ready ? tr("相机和位置模型已就绪，可重做第一阶段或开始第二阶段。", "Camera and position model are ready. Repeat stage 1 or start stage 2.")
      : tr("相机已就绪，请先完成第一阶段。", "Camera is ready. Complete stage 1 first."));
  $("calibrationReadiness").classList.toggle("ready", inputReady && geometryReady);
  $("geometryStatus").textContent = geometryReady
    ? tr("已保存。调整位置后请重新校准。", "Saved. Recalibrate after moving the camera or screen.")
    : tr("首次使用必须确认这些数值。", "Confirm these values before your first calibration.");
  $("modelStatus").textContent = ready ? tr("当前模型可用", "Current model ready") : tr("需要位置校准", "Position calibration needed");
  $("startButton").disabled = Boolean(engine.tracking) || !inputReady || !ready || !geometryReady || busy;
  $("stopButton").disabled = !engine.tracking || actionBusy;
  $("backgroundButton").disabled = !inputReady || !ready || !geometryReady || busy;
  $("exitButton").disabled = busy;
  $("startVideoButton").disabled = !inputReady || !geometryReady || busy;
  $("startEventButton").disabled = !inputReady || !geometryReady || !ready || busy;
  $("startEventButton").title = ready ? "" : tr("请先完成第一阶段", "Complete stage 1 first");
  $("previewStartButton").disabled = busy || (!engine.tracking && (!inputReady || !ready || !geometryReady));
  $("previewStartButton").textContent = document.fullscreenElement === $("previewPage")
    ? tr("退出全屏", "Exit fullscreen") : tr("全屏预览", "Fullscreen preview");
  $("configForm").querySelectorAll("input, select, button").forEach(element => { element.disabled = busy; });
  $("language").disabled = actionBusy;
  renderConfigApplyStatus();
  renderPairing(payload.application);
  renderStability(payload);
  renderCalibration(calibration);
  const intrinsics = engine.intrinsics || {};
  $("intrinsicsSource").textContent = intrinsics.source === "estimated_windows_camera"
    ? tr("由电脑摄像头视场角估算", "Estimated from PC camera field of view")
    : intrinsics.source ? tr("来自手机 Camera2", "Provided by phone Camera2") : tr("等待内参", "Waiting for intrinsics");
  $("intrinsicsGrid").innerHTML = ["width", "height", "fx", "fy", "cx", "cy"].map(key =>
    "<div><dt>" + ({ width: tr("宽度", "Width"), height: tr("高度", "Height") }[key] || key) +
    "</dt><dd>" + (Number.isFinite(Number(intrinsics[key])) && intrinsics[key] != null ? Number(intrinsics[key]).toFixed(key.length === 2 ? 2 : 0) : "—") + "</dd></div>"
  ).join("");
  if (!engine.tracking) {
    $("previewState").textContent = tr("追踪未启动", "Tracking is stopped");
    $("combinedDot").hidden = true;
  }
  $("previewPostprocess").textContent = payload.config.event_temporal_enabled !== false
    ? tr("自动稳定 · 眼跳预测开", "Automatic stability · prediction on")
    : tr("自动稳定 · 眼跳预测关", "Automatic stability · prediction off");
  renderPerformance(engine.performance || lastPerformance);
  syncStreams();
}
async function refreshStatus() {
  if (statusRequestPending || applicationClosed) return;
  statusRequestPending = true;
  try { renderStatus(await request("/api/status")); }
  catch (error) {
    $("runtimeSummary").textContent = tr("连接中断", "Disconnected");
    $("runtimeDetail").textContent = describeRuntimeError(error.message);
    $("runtimeDot").classList.remove("ok");
  } finally {
    statusRequestPending = false;
    clearTimeout(statusTimer);
    if (!document.hidden && !applicationClosed) statusTimer = setTimeout(refreshStatus,
      ["preview", "performance"].includes(activePage) && state?.engine?.tracking ? 3000 : 1000);
  }
}
async function refreshWindowsCameras() {
  const button = $("refreshWindowsCamerasButton");
  button.disabled = true;
  try {
    const result = await request("/api/windows-cameras");
    const select = $("windowsCameraIndex"), previous = select.value;
    select.replaceChildren();
    (result.cameras || []).forEach(camera => select.add(new Option(camera.name || "Camera " + camera.index, String(camera.index))));
    if (!Array.from(select.options).some(o => o.value === previous)) {
      select.add(new Option(tr("设备 ", "Device ") + previous + tr("（未检测到）", " (not detected)"), previous));
    }
    select.value = previous;
  } finally { button.disabled = Boolean(state?.calibration?.active) || actionBusy; }
}
function stopFrameStream() {
  frameStreamActive = false;
  $("cameraFrame").removeAttribute("src");
  $("cameraFrame").hidden = true;
  $("cameraPlaceholder").hidden = false;
}
function syncStreams() {
  const showFrame = !document.hidden && !applicationClosed && activePage === "control" && state?.engine?.input?.ready;
  if (showFrame && !frameStreamActive) {
    frameStreamActive = true;
    $("cameraFrame").src = "/api/frame.mjpg?t=" + Date.now();
    $("cameraFrame").hidden = false;
    $("cameraPlaceholder").hidden = true;
  } else if (!showFrame && frameStreamActive) stopFrameStream();
  const showGaze = !document.hidden && !applicationClosed && ["preview", "performance"].includes(activePage);
  if (showGaze) startGazeStream(); else stopGazeStream();
}
function placeDot(element, point, gaze) {
  if (!point || !gaze.valid || !state) { element.hidden = true; return; }
  element.hidden = false;
  element.style.left = point[0] / Math.max(1, state.config.screen_width - 1) * 100 + "%";
  element.style.top = point[1] / Math.max(1, state.config.screen_height - 1) * 100 + "%";
}
function renderGaze(gaze) {
  if (applicationClosed) return;
  placeDot($("combinedDot"), gaze.combined, gaze);
  const now = performance.now();
  gazeArrivalTimes.push(now);
  gazeArrivalTimes = gazeArrivalTimes.filter(value => value >= now - 1000);
  const fps = gazeArrivalTimes.length > 1 ? (gazeArrivalTimes.length - 1) * 1000 / (now - gazeArrivalTimes[0]) : 0;
  $("previewState").textContent = gaze.valid ? fps.toFixed(1) + " FPS · " + Number(gaze.processing_ms || 0).toFixed(1) + " ms"
    : describeRuntimeError(gaze.error) || tr("等待有效注视输出", "Waiting for valid gaze");
  const modes = {
    event_fixation: ["注视 · 自动稳定", "Fixation · automatic stability"],
    event_pursuit: ["追视 · 跟随观测", "Pursuit · following observed motion"],
    event_saccade_observed: ["眼跳 · 跟随观测", "Saccade · following observed motion"],
    event_saccade_velocity_prediction: ["眼跳 · 短时预测", "Saccade · short-horizon prediction"],
    event_saccade_landing_prediction: ["眼跳 · 落点预测", "Saccade · landing prediction"],
    event_landing: ["落定 · 恢复稳定", "Landing · resuming stability"]
  };
  const mode = gaze.postprocess?.extrapolation_state?.mode;
  $("previewMotionMode").textContent = modes[mode] ? tr(...modes[mode]) : "";
  renderPerformance(gaze.performance, gaze.output_seq);
}
function startGazeStream() {
  const include = activePage === "performance";
  if (gazeStream && gazeStreamIncludesPerformance === include) return;
  stopGazeStream();
  const stream = new EventSource("/api/gaze/stream?performance=" + (include ? "1" : "0"));
  gazeStream = stream;
  gazeStreamIncludesPerformance = include;
  stream.onmessage = event => {
    if (stream !== gazeStream) return;
    try { renderGaze(JSON.parse(event.data)); } catch (_) { /* A malformed frame must not break the stream. */ }
  };
  stream.onerror = () => {
    if (stream !== gazeStream) return;
    $("previewState").textContent = tr("数据连接中断，正在重连", "Data interrupted; reconnecting");
    $("latencyLiveState").textContent = tr("正在重连", "Reconnecting");
    $("latencyLiveState").classList.remove("live");
  };
}
function stopGazeStream() {
  gazeStream?.close();
  gazeStream = null;
  gazeArrivalTimes = [];
}
const latencyStages = [
  ["phone_capture_to_send_ms", "手机采集 → 发包", "Phone capture → send"],
  ["transport_to_first_packet_ms", "发送 → 首包（估计）", "Send → first packet (estimate)"],
  ["receive_decode_ms", "接收与解码（以下三项小计）", "Receive & decode (subtotal of next three)"],
  ["packet_assembly_ms", "　收齐分包", "　Packet assembly"],
  ["decode_queue_ms", "　等待解码", "　Decode queue"],
  ["jpeg_decode_ms", "　解码与旋转", "　Decode & rotate"],
  ["pc_queue_ms", "等待电脑处理", "PC processing queue"],
  ["face_landmarks_ms", "人脸关键点", "Face landmarks"],
  ["eye_normalization_ms", "眼部归一化", "Eye normalization"],
  ["backend_overhead_ms", "视觉后端其他", "Vision backend overhead"],
  ["gaze_model_ms", "注视模型", "Gaze model"],
  ["projection_fusion_ms", "投影与双眼融合", "Projection & eye fusion"],
  ["prediction_postprocess_ms", "自动稳定与眼跳预测", "Automatic stability & saccade prediction"],
  ["shared_memory_write_ms", "写入共享内存", "Shared memory write"],
  ["other_processing_ms", "其他处理", "Other processing"]
];
function latencyText(value) {
  return value != null && Number.isFinite(Number(value)) ? Number(value).toFixed(2) + " ms" : "—";
}
function drawLatencyChart() {
  if (activePage !== "performance") return;
  const canvas = $("latencyChart"), width = Math.max(240, canvas.clientWidth || 900), height = Math.max(150, canvas.clientHeight || 205);
  const ratio = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  if (!context) return;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  const left = 56, top = 12, pw = width - left - 10, ph = height - 36;
  const values = latencyHistory.flatMap(item => [item.total, item.sum]).filter(Number.isFinite);
  const max = Math.max(20, Math.ceil(Math.max(...values, 20) * 1.1 / 10) * 10);
  context.font = "12px Segoe UI";
  for (let i = 0; i <= 4; i++) {
    const y = top + ph * i / 4;
    context.strokeStyle = "#344039"; context.lineWidth = 1;
    context.beginPath(); context.moveTo(left, y); context.lineTo(width - 10, y); context.stroke();
    context.fillStyle = "#a2b0a7"; context.textAlign = "right";
    context.fillText((max * (1 - i / 4)).toFixed(0) + " ms", left - 8, y + 4);
  }
  for (const [field, color] of [["total", "#79d4a0"], ["sum", "#e6bf73"]]) {
    context.strokeStyle = color; context.lineWidth = 1.7; context.beginPath();
    let drawing = false;
    latencyHistory.forEach((item, i) => {
      if (!Number.isFinite(item[field])) { drawing = false; return; }
      const x = left + pw * i / Math.max(1, latencyHistory.length - 1), y = top + ph * (1 - item[field] / max);
      if (drawing) context.lineTo(x, y); else context.moveTo(x, y);
      drawing = true;
    });
    context.stroke();
  }
}
function renderPerformance(data, sequence = null) {
  if (data?.schema === "opengazelink-live-latency-v1") lastPerformance = data;
  if (activePage !== "performance") return;
  data = lastPerformance || {};
  const current = data.current || {}, metric = field => data.metrics?.[field] || {};
  const total = metric("source_to_shared_memory_ms_proxy"), processing = metric("pc_processing_ms");
  $("latencyEndToEnd").textContent = latencyText(current.source_to_shared_memory_ms_proxy);
  $("latencyEndToEndStats").textContent = tr("均值 ", "Mean ") + latencyText(total.mean) + " · P95 " + latencyText(total.p95);
  $("latencyStageSum").textContent = latencyText(current.stage_sum_ms);
  $("latencyToDisplay").textContent = latencyText(current.source_to_display_ms_estimate);
  $("latencyProcessing").textContent = latencyText(current.pc_processing_ms);
  $("latencyProcessingStats").textContent = tr("均值 ", "Mean ") + latencyText(processing.mean) + " · P95 " + latencyText(processing.p95);
  $("latencySumCheck").textContent = tr("与全流程差 ", "Difference from total: ") + latencyText(current.sum_error_ms);
  $("latencyDisplayNote").textContent = tr("含显示预算 ", "Includes display budget: ") + latencyText(current.display_delay_ms_assumed);
  $("latencySamples").textContent = Number(data.sample_count || 0) + tr(" 个样本 · 后台每秒汇总", " samples · aggregated each second");
  $("latencyWindow").textContent = Number(data.window_seconds || 30) + " s";
  $("latencyHorizon").textContent = latencyText(current.prediction_horizon_ms);
  $("latencyDisplayBudget").textContent = latencyText(current.display_delay_ms_assumed);
  const probe = current.clock_basis === "phone_roundtrip_alignment", local = current.clock_basis === "camera_read_completion_proxy";
  $("latencyBasis").textContent = probe ? tr("双向探测校时", "Round-trip clock alignment")
    : local ? tr("以电脑读帧完成时刻为基准", "PC frame-read completion reference")
    : current.clock_basis === "phone_minimum_transit_proxy" ? tr("最小传输时间代理", "Minimum-transit proxy") : tr("时间基准尚不可用", "Timing reference unavailable");
  $("latencyMeasurementNote").textContent = probe
    ? tr("最近探测 RTT ", "Latest probe RTT ") + latencyText(current.clock_probe_rtt_ms) + tr("，校时不确定度约 ±", ", alignment uncertainty approximately ±") +
      latencyText(current.clock_probe_uncertainty_ms) + tr("。路径不对称与设备调度会影响单向延迟估算。显示预算不是实测。", ". Path asymmetry and scheduling affect one-way estimates. Display budget is not a measurement.")
    : local ? tr("不包含驱动缓存、曝光和真正显示出光。", "Excludes driver buffering, exposure and actual display emission.")
    : tr("手机与电脑没有共同时钟；代理值可能偏低，无法单独识别固定网络延迟。", "Phone and PC clocks differ. Proxy values may be low and cannot isolate fixed network latency.");
  $("latencyStageRows").innerHTML = latencyStages.map(([field, zh, en]) =>
    "<tr><td>" + tr(zh, en) + "</td>" + [current[field], metric(field).mean, metric(field).p95, metric(field).max].map(v => "<td>" + latencyText(v) + "</td>").join("") + "</tr>"
  ).join("");
  const live = Boolean(state?.engine?.tracking && data.sample_count);
  $("latencyLiveState").classList.toggle("live", live);
  $("latencyLiveState").textContent = live ? tr("实时统计", "Live") : tr("等待有效输出", "Waiting for output");
  if (sequence != null && sequence !== lastLatencySequence) {
    lastLatencySequence = sequence;
    latencyHistory.push({ total: current.source_to_shared_memory_ms_proxy, sum: current.stage_sum_ms });
    if (latencyHistory.length > 180) latencyHistory.shift();
  }
  drawLatencyChart();
}
async function runAction(action) {
  if (actionBusy) return;
  actionBusy = true;
  if (state) renderStatus(state);
  try { await action(); } catch (error) { toast(error.message); }
  finally { actionBusy = false; if (!applicationClosed) await refreshStatus(); }
}
function leaveApplication(heading, detail) {
  applicationClosed = true;
  clearTimeout(statusTimer);
  stopGazeStream(); stopFrameStream();
  document.body.replaceChildren();
  const main = document.createElement("main"); main.className = "background-confirmation";
  const h1 = document.createElement("h1"); h1.textContent = heading;
  const p = document.createElement("p"); p.textContent = detail;
  main.append(h1, p); document.body.append(main);
}
document.querySelectorAll(".nav-button").forEach(button => button.addEventListener("click", () => setPage(button.dataset.page)));
$("language").addEventListener("change", () => applyLanguage($("language").value));
document.addEventListener("languagechange", () => { if (state) renderStatus(state); });
for (const type of ["input", "change"]) $("configForm").addEventListener(type, () => {
  configDirty = true;
  $("windowsCameraSettings").hidden = $("inputSource").value !== "windows_camera";
  renderConfigApplyStatus();
});
$("saveConfigButton").addEventListener("click", () => runAction(async () => { await saveConfig(); toast(tr("设置已保存并应用", "Settings saved and applied")); }));
$("refreshWindowsCamerasButton").addEventListener("click", () => refreshWindowsCameras().catch(error => toast(error.message)));
$("pairingCandidates").addEventListener("click", event => {
  const button = event.target.closest("[data-accept-phone]");
  if (button) runAction(() => post("/api/pairing/accept", { phone_id: button.dataset.acceptPhone }));
});
$("forgetPairingButton").addEventListener("click", () => {
  if (confirm(tr("解除手机配对？传输中的手机需要重新连接。", "Unpair this phone? An active phone stream will need to reconnect."))) {
    runAction(() => post("/api/pairing/forget"));
  }
});
$("startButton").addEventListener("click", () => runAction(async () => { await saveConfig(); await post("/api/tracking/start"); }));
$("stopButton").addEventListener("click", () => runAction(() => post("/api/tracking/stop")));
$("backgroundButton").addEventListener("click", () => runAction(async () => {
  await saveConfig(); await post("/api/application/background");
  leaveApplication(tr("正在后台运行", "Running in background"), tr("可以关闭此页面。再次打开控制中心即可恢复。", "You can close this page. Open the control center to return."));
}));
$("exitButton").addEventListener("click", () => runAction(async () => {
  await post("/api/application/exit");
  leaveApplication(tr("OpenGazeLink 已退出", "OpenGazeLink has exited"), tr("现在可以关闭此页面。", "You can close this page."));
}));
$("previewStartButton").addEventListener("click", () => {
  if (document.fullscreenElement === $("previewPage")) { document.exitFullscreen().catch(error => toast(error.message)); return; }
  // Request fullscreen in the click gesture, before asynchronous network calls.
  const fullscreen = $("previewPage").requestFullscreen();
  runAction(async () => {
    try {
      await fullscreen;
      if (!state?.engine?.tracking) { await saveConfig(); await post("/api/tracking/start"); }
    } catch (error) {
      if (document.fullscreenElement === $("previewPage")) await document.exitFullscreen();
      throw error;
    }
  });
});
$("cameraFrame").addEventListener("error", stopFrameStream);
document.addEventListener("fullscreenchange", () => { if (activePage === "preview") refreshStatus(); });
document.addEventListener("visibilitychange", () => {
  syncStreams(); clearTimeout(statusTimer);
  if (!document.hidden) refreshStatus();
});
window.addEventListener("resize", drawLatencyChart);
applyLanguage(language);
refreshStatus();
