package com.eyetracing.android

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.content.res.Resources
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import android.os.Bundle
import android.text.InputType
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.view.Gravity
import android.view.Surface
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import org.json.JSONObject
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.net.SocketTimeoutException
import java.util.Locale
import java.util.UUID
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private val prefs by lazy { getSharedPreferences("eyetracing_yuv_sender", MODE_PRIVATE) }
    private var uiLanguage = "en"
    private lateinit var localized: Resources
    private val textBindings = mutableListOf<Pair<TextView, Int>>()
    private lateinit var hostInput: EditText
    private lateinit var portInput: EditText
    private lateinit var languageInput: Spinner
    private lateinit var cameraInput: Spinner
    private lateinit var resolutionInput: Spinner
    private lateinit var frameRateInput: Spinner
    private lateinit var rotationInput: Spinner
    private lateinit var jpegOptions: LinearLayout
    private lateinit var jpegQualityInput: Spinner
    private lateinit var cropInputs: List<EditText>
    private lateinit var outputSizeText: TextView
    private lateinit var modeGroup: RadioGroup
    private lateinit var highSpeedButton: RadioButton
    private lateinit var cameraModeButton: RadioButton
    private lateinit var modeHelp: TextView
    private lateinit var formatHelp: TextView
    private lateinit var statusText: TextView
    private lateinit var connectionText: TextView
    private lateinit var statsSummary: TextView
    private lateinit var captureRate: TextView
    private lateinit var sendRate: TextView
    private lateinit var videoRate: TextView
    private lateinit var encodingAge: TextView
    private lateinit var exposure: TextView
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var refreshButton: Button
    private lateinit var discoveryButton: Button
    private lateinit var changePcButton: Button
    private lateinit var connectionOptions: LinearLayout
    private val catalogWorker = Executors.newSingleThreadExecutor()
    private var cameras = emptyList<CameraChoice>()
    private var options = emptyList<CaptureOption>()
    private var resolutions = emptyList<CaptureResolution>()
    private var frameRates = emptyList<Int>()
    private var selectedCameraId = ""
    private var selectedOptionKey: String? = null
    private var selectedMode = SessionMode.HIGH_SPEED
    private var loadingCameras = false
    private var updatingUi = false
    private var catalogGeneration = 0
    private val excludedCombinations = mutableSetOf<String>()
    private var stream: CameraStream? = null
    private var streamGeneration = 0
    private var currentRates: StreamRates? = null
    private var statusId = R.string.ready
    private var statusArgs = emptyArray<Any>()
    private var connectionId = R.string.connection_idle
    private var connectionArgs = emptyArray<Any>()
    @Volatile private var discoverySocket: DatagramSocket? = null
    @Volatile private var discoveryGeneration = 0
    private var startAfterPermission = false
    private var destroyed = false

    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) refreshCameraOptions(startAfterPermission)
        else setStatus(R.string.permission_denied)
        startAfterPermission = false
        updateEnabled()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        uiLanguage = prefs.getString("ui_language", Locale.getDefault().language).let { if (it == "zh") "zh" else "en" }
        localize()
        selectedCameraId = prefs.getString("camera_id", "").orEmpty()
        selectedMode = when (prefs.getString("session_mode", null)) {
            "CAMERA" -> SessionMode.CAMERA
            "HIGH_SPEED" -> SessionMode.HIGH_SPEED
            else -> if (prefs.contains("encoding_mode") && prefs.getString("encoding_mode", "") != "avc") SessionMode.CAMERA else SessionMode.HIGH_SPEED
        }
        migratePreferences()
        WindowCompat.setDecorFitsSystemWindows(window, false)
        buildUi()
        refreshCameraOptions()
        startPcDiscovery()
    }

    private fun migratePreferences() {
        // New crop keys deliberately do not reuse legacy sensor-oriented insets.
        // Display-oriented crop settings survive subsequent launches.
        val edit = prefs.edit().putString("session_mode", selectedMode.name)
        if (!prefs.contains("rotation_mode")) {
            edit.putInt("rotation_mode", if (prefs.contains("frame_rotation")) prefs.getInt("frame_rotation", 270) else -1)
        }
        listOf("crop_left", "crop_right", "crop_top", "crop_bottom", "encoding_mode",
            "processing_mode", "jpeg_quality", "chunk_payload_size", "low_latency_wifi").forEach { edit.remove(it) }
        edit.apply()
    }

    override fun onDestroy() {
        destroyed = true
        discoveryGeneration++
        discoverySocket?.close()
        catalogGeneration++
        catalogWorker.shutdownNow()
        stopStreaming()
        super.onDestroy()
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()
    // Plain labels may contain literal '%' characters (e.g. crop percentages).
    // Only invoke Resources' formatter when substitution arguments are supplied.
    private fun s(id: Int, vararg args: Any): String =
        if (args.isEmpty()) localized.getString(id) else localized.getString(id, *args)
    private fun localize() {
        val config = Configuration(resources.configuration).apply { setLocale(Locale(uiLanguage)) }
        localized = createConfigurationContext(config).resources
    }
    private fun bind(view: TextView, id: Int): TextView {
        textBindings += view to id
        view.text = s(id)
        return view
    }
    private fun text(id: Int, size: Float = 15f): TextView = TextView(this).apply {
        bind(this, id); textSize = size; setTextColor(0xFFBCCBC0.toInt())
        setLineSpacing(dp(3).toFloat(), 1f)
        setPadding(0, dp(6), 0, dp(8))
    }
    private fun button(id: Int, action: () -> Unit): Button = Button(this).apply {
        bind(this, id); isAllCaps = false; textSize = 15f; minHeight = dp(48)
        setOnClickListener { action() }
    }
    private fun spinner(): Spinner = Spinner(this).apply { minimumHeight = dp(48) }
    private fun setItems(view: Spinner, items: List<String>, position: Int = 0) {
        view.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, items).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        view.setSelection(position.coerceIn(0, (items.size - 1).coerceAtLeast(0)))
    }
    private fun selected(action: (Int) -> Unit) = object : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
            if (!updatingUi) action(position)
        }
        override fun onNothingSelected(parent: AdapterView<*>?) = Unit
    }
    private fun column() = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
    private fun card(titleId: Int) = column().apply {
        setPadding(dp(16), dp(12), dp(16), dp(16))
        background = GradientDrawable().apply { setColor(0xFF1B241E.toInt()); cornerRadius = dp(8).toFloat() }
        addView(text(titleId, 18f).apply { setTextColor(0xFFF0F6F2.toInt()); setTypeface(null, Typeface.BOLD) })
        layoutParams = LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(16) }
    }

    private fun buildUi() {
        val root = column().apply { setBackgroundColor(0xFF111813.toInt()) }
        ViewCompat.setOnApplyWindowInsetsListener(root) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.ime())
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
        val header = LinearLayout(this).apply {
            gravity = Gravity.CENTER_VERTICAL; setPadding(dp(18), dp(10), dp(12), dp(6))
            addView(TextView(this@MainActivity).apply {
                setText(R.string.app_name); textSize = 21f; setTypeface(null, Typeface.BOLD)
                setTextColor(0xFFF0F6F2.toInt())
            }, LinearLayout.LayoutParams(0, -2, 1f))
        }
        languageInput = spinner()
        setItems(languageInput, listOf("中文", "English"), if (uiLanguage == "zh") 0 else 1)
        languageInput.contentDescription = "Language / 语言"
        languageInput.onItemSelectedListener = selected { index ->
            val next = if (index == 0) "zh" else "en"
            if (next != uiLanguage) {
                uiLanguage = next; localize(); prefs.edit().putString("ui_language", next).apply()
                updateLanguage()
            }
        }
        header.addView(languageInput, LinearLayout.LayoutParams(dp(112), -2))
        root.addView(header)
        val content = column().apply { setPadding(dp(16), dp(8), dp(16), dp(8)) }
        content.addView(text(R.string.app_intro))

        val connection = card(R.string.connection_title)
        connection.addView(text(R.string.host_label, 14f))
        hostInput = EditText(this).apply {
            setSingleLine(true); textSize = 16f; hint = s(R.string.host_hint)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
            setText(prefs.getString("host", "")); setSelectAllOnFocus(true)
        }
        connection.addView(hostInput)
        discoveryButton = button(R.string.find_pc) { startPcDiscovery(manual = true) }
        connection.addView(discoveryButton)
        connectionText = text(R.string.connection_idle, 14f)
        connection.addView(connectionText)
        connectionOptions = column().apply { visibility = View.GONE }
        connection.addView(button(R.string.connection_options) {
            connectionOptions.visibility = if (connectionOptions.visibility == View.GONE) View.VISIBLE else View.GONE
        })
        connectionOptions.addView(text(R.string.port_label, 14f))
        portInput = EditText(this).apply {
            setSingleLine(true); inputType = InputType.TYPE_CLASS_NUMBER
            setText(prefs.getString("port", "5007")); textSize = 16f
        }
        connectionOptions.addView(portInput)
        connectionOptions.addView(text(R.string.rotation_label, 14f))
        rotationInput = spinner()
        setItems(rotationInput, rotationLabels(), rotationValues.indexOf(prefs.getInt("rotation_mode", -1)).coerceAtLeast(0))
        connectionOptions.addView(rotationInput)
        rotationInput.onItemSelectedListener = selected { updateOutputSize() }
        changePcButton = button(R.string.change_pc) {
            prefs.edit().remove("paired_instance_id").apply()
            hostInput.setText(""); startPcDiscovery(manual = true, restart = true)
        }
        connectionOptions.addView(changePcButton)
        connection.addView(connectionOptions)
        content.addView(connection)

        val capture = card(R.string.capture_title)
        modeGroup = RadioGroup(this).apply { orientation = RadioGroup.HORIZONTAL }
        highSpeedButton = RadioButton(this).apply { id = View.generateViewId(); bind(this, R.string.high_speed); textSize = 15f; minHeight = dp(52) }
        cameraModeButton = RadioButton(this).apply { id = View.generateViewId(); bind(this, R.string.camera_mode); textSize = 15f; minHeight = dp(52) }
        modeGroup.addView(highSpeedButton, RadioGroup.LayoutParams(0, -2, 1f))
        modeGroup.addView(cameraModeButton, RadioGroup.LayoutParams(0, -2, 1f))
        modeGroup.check(if (selectedMode == SessionMode.HIGH_SPEED) highSpeedButton.id else cameraModeButton.id)
        modeGroup.setOnCheckedChangeListener { _, checked ->
            if (!updatingUi && stream == null) {
                selectedMode = if (checked == highSpeedButton.id) SessionMode.HIGH_SPEED else SessionMode.CAMERA
                selectedOptionKey = null; rebuildOptions()
            }
        }
        capture.addView(modeGroup)
        modeHelp = text(R.string.high_speed_help, 14f); capture.addView(modeHelp)
        capture.addView(text(R.string.camera_label, 14f))
        cameraInput = spinner(); capture.addView(cameraInput)
        cameraInput.onItemSelectedListener = selected { index ->
            cameras.getOrNull(index)?.let { choice ->
                if (choice.id != selectedCameraId && stream == null) {
                    selectedCameraId = choice.id; selectedOptionKey = null; rebuildOptions()
                }
            }
        }
        resolutionInput = Spinner(this, Spinner.MODE_DIALOG).apply {
            id = View.generateViewId(); minimumHeight = dp(48)
        }
        capture.addView(text(R.string.resolution_label, 14f).apply { labelFor = resolutionInput.id })
        capture.addView(resolutionInput)
        resolutionInput.onItemSelectedListener = selected { index ->
            val resolution = resolutions.getOrNull(index)
            val previous = currentOption()
            if (stream == null && !loadingCameras && resolution != null && resolution != previous?.resolution) {
                CaptureOptions.selectResolution(options, resolution, previous)?.let { chosen ->
                    chooseOption(chosen)
                    if (previous != null && chosen.fps != previous.fps) setStatus(R.string.frame_rate_adjusted, chosen.fps)
                    else setStatus(R.string.ready)
                }
            }
        }
        frameRateInput = Spinner(this, Spinner.MODE_DIALOG).apply {
            id = View.generateViewId(); minimumHeight = dp(48)
        }
        capture.addView(text(R.string.frame_rate_label, 14f).apply { labelFor = frameRateInput.id })
        capture.addView(frameRateInput)
        frameRateInput.onItemSelectedListener = selected { index ->
            val fps = frameRates.getOrNull(index)
            val previous = currentOption()
            if (stream == null && !loadingCameras && fps != null && fps != previous?.fps) {
                CaptureOptions.selectFrameRate(options, fps, previous)?.let { chosen ->
                    chooseOption(chosen)
                    if (previous != null && chosen.resolution != previous.resolution) setStatus(R.string.resolution_adjusted, chosen.resolution.label)
                    else setStatus(R.string.ready)
                }
            }
        }
        formatHelp = text(R.string.reading_cameras, 14f); capture.addView(formatHelp)
        refreshButton = button(R.string.refresh_formats) {
            if (hasCameraPermission()) refreshCameraOptions()
            else { startAfterPermission = false; permission.launch(Manifest.permission.CAMERA) }
        }
        capture.addView(refreshButton)
        jpegOptions = column()
        jpegOptions.addView(text(R.string.jpeg_quality_label, 14f))
        jpegQualityInput = spinner()
        setItems(jpegQualityInput, JpegQuality.levels.map { "Q$it" },
            JpegQuality.levels.indexOf(prefs.getInt("camera_jpeg_quality", JpegQuality.DEFAULT)).coerceAtLeast(0))
        jpegOptions.addView(jpegQualityInput)
        jpegOptions.addView(text(R.string.jpeg_quality_help, 13f))
        jpegOptions.addView(text(R.string.crop_label, 14f))
        cropInputs = listOf(R.string.crop_left to "left", R.string.crop_right to "right",
            R.string.crop_top to "top", R.string.crop_bottom to "bottom").map { (label, edge) ->
            val row = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL }
            val input = EditText(this).apply {
                id = View.generateViewId(); setSingleLine(true); setSelectAllOnFocus(true)
                inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
                setText(prefs.getString("display_crop_$edge", "0")); textSize = 16f
            }
            row.addView(text(label, 14f).apply { labelFor = input.id }, LinearLayout.LayoutParams(0, -2, 1f))
            row.addView(input, LinearLayout.LayoutParams(dp(100), -2)); jpegOptions.addView(row)
            input
        }
        jpegOptions.addView(text(R.string.crop_help, 13f))
        outputSizeText = text(R.string.not_available, 14f); jpegOptions.addView(outputSizeText)
        cropInputs.forEach { input -> input.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) { updateOutputSize() }
            override fun afterTextChanged(s: Editable?) = Unit
        }) }
        capture.addView(jpegOptions)
        content.addView(capture)

        val monitor = card(R.string.performance_title)
        statsSummary = text(R.string.not_available); monitor.addView(statsSummary)
        val metrics = column().apply { visibility = View.GONE }
        monitor.addView(button(R.string.performance_details) {
            metrics.visibility = if (metrics.visibility == View.GONE) View.VISIBLE else View.GONE
        })
        fun metric(id: Int): TextView {
            val row = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL }
            row.addView(text(id, 14f), LinearLayout.LayoutParams(0, -2, 1f))
            val value = TextView(this).apply { text = "—"; textSize = 15f; setTextColor(0xFFF0F6F2.toInt()) }
            row.addView(value); metrics.addView(row)
            return value
        }
        captureRate = metric(R.string.capture_fps); sendRate = metric(R.string.sent_fps)
        videoRate = metric(R.string.video_rate); encodingAge = metric(R.string.encode_age); exposure = metric(R.string.exposure)
        metrics.addView(text(R.string.performance_note, 13f))
        monitor.addView(metrics); content.addView(monitor)
        root.addView(ScrollView(this).apply { isFillViewport = true; addView(content) }, LinearLayout.LayoutParams(-1, 0, 1f))

        val footer = column().apply { setPadding(dp(16), dp(8), dp(16), dp(10)); setBackgroundColor(0xFF1B241E.toInt()) }
        statusText = text(R.string.ready).apply { setTextColor(0xFFF0F6F2.toInt()); accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE }
        footer.addView(statusText)
        startButton = button(R.string.start_stream) { ensureCameraAndStart() }.apply {
            backgroundTintList = android.content.res.ColorStateList.valueOf(0xFF2F8050.toInt())
        }
        stopButton = button(R.string.stop_stream) { stopStreaming(); setStatus(R.string.stopped) }
        val commands = LinearLayout(this).apply {
            addView(stopButton, LinearLayout.LayoutParams(0, -2, 1f))
            addView(startButton, LinearLayout.LayoutParams(0, -2, 2f))
        }
        footer.addView(commands); footer.addView(text(R.string.stream_hint, 12f))
        root.addView(footer)
        setContentView(root)
        updateEnabled()
    }

    private val rotationValues = listOf(-1, 0, 90, 180, 270)
    private fun rotationLabels() = listOf(s(R.string.rotation_auto), "0°", "90°", "180°", "270°")
    private fun cameraName(camera: CameraChoice) = s(when (camera.facing) {
        CameraCharacteristics.LENS_FACING_FRONT -> R.string.front_camera
        CameraCharacteristics.LENS_FACING_BACK -> R.string.rear_camera
        else -> R.string.external_camera
    }, camera.id)

    private fun updateLanguage() {
        textBindings.forEach { (view, id) -> view.text = s(id) }
        hostInput.hint = s(R.string.host_hint)
        val rotation = rotationInput.selectedItemPosition
        updatingUi = true
        setItems(rotationInput, rotationLabels(), rotation)
        setItems(cameraInput, cameras.map(::cameraName), cameras.indexOfFirst { it.id == selectedCameraId }.coerceAtLeast(0))
        updatingUi = false
        rebuildOptions()
        statusText.text = s(statusId, *statusArgs)
        connectionText.text = s(connectionId, *connectionArgs)
        renderRates()
    }

    private fun hasCameraPermission() = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    private fun refreshCameraOptions(startWhenReady: Boolean = false) {
        if (stream != null || loadingCameras || destroyed) return
        if (!hasCameraPermission()) { setStatus(R.string.no_cameras); updateEnabled(); return }
        loadingCameras = true; excludedCombinations.clear()
        formatHelp.text = s(R.string.reading_cameras)
        updateEnabled()
        val generation = ++catalogGeneration
        catalogWorker.execute {
            val result = runCatching { CameraCatalog.read(getSystemService(Context.CAMERA_SERVICE) as CameraManager) }
            runOnUiThread {
                if (destroyed || generation != catalogGeneration) return@runOnUiThread
                loadingCameras = false
                cameras = result.getOrDefault(emptyList())
                if (cameras.none { it.id == selectedCameraId }) {
                    val facing = if (prefs.getString("camera_facing", "front") == "back") CameraCharacteristics.LENS_FACING_BACK else CameraCharacteristics.LENS_FACING_FRONT
                    val available = cameras.filter { camera -> camera.options.any { it.mode == selectedMode } }
                    selectedCameraId = (available.firstOrNull { it.facing == facing } ?: available.firstOrNull()
                        ?: cameras.firstOrNull { it.facing == facing } ?: cameras.firstOrNull())?.id.orEmpty()
                }
                updatingUi = true
                setItems(cameraInput, cameras.map(::cameraName).ifEmpty { listOf(s(R.string.no_cameras)) },
                    cameras.indexOfFirst { it.id == selectedCameraId }.coerceAtLeast(0))
                updatingUi = false
                rebuildOptions()
                if (result.isFailure) setStatus(R.string.camera_read_failed)
                else if (cameras.isEmpty()) setStatus(R.string.no_cameras)
                else setStatus(R.string.ready)
                if (startWhenReady && currentOption() != null) startStreaming()
            }
        }
    }

    private fun optionPreferenceKey() = "capture_option_" + selectedCameraId + "_" + selectedMode.name
    private fun rebuildOptions() {
        options = cameras.firstOrNull { it.id == selectedCameraId }?.options.orEmpty()
            .filter { it.mode == selectedMode && it.key !in excludedCombinations }
        val saved = selectedOptionKey ?: prefs.getString(optionPreferenceKey(), null)
        val chosen = CaptureOptions.selected(options, saved)
        selectedOptionKey = chosen?.key
        resolutions = CaptureOptions.resolutions(options)
        frameRates = CaptureOptions.frameRates(options)
        renderCaptureOptions()
    }
    private fun chooseOption(option: CaptureOption) {
        selectedOptionKey = option.key
        prefs.edit().putString(optionPreferenceKey(), option.key).apply()
        renderCaptureOptions()
    }
    private fun renderCaptureOptions() {
        val chosen = currentOption()
        updatingUi = true
        resolutionInput.prompt = s(R.string.resolution_label)
        frameRateInput.prompt = s(R.string.frame_rate_label)
        setItems(resolutionInput, resolutions.map { it.label }.ifEmpty { listOf(s(R.string.no_combinations)) },
            resolutions.indexOf(chosen?.resolution).coerceAtLeast(0))
        setItems(frameRateInput, frameRates.map { "$it FPS" }.ifEmpty { listOf(s(R.string.no_combinations)) },
            frameRates.indexOf(chosen?.fps).coerceAtLeast(0))
        updatingUi = false
        modeHelp.text = s(if (selectedMode == SessionMode.HIGH_SPEED) R.string.high_speed_help else R.string.camera_help)
        formatHelp.text = if (options.isNotEmpty()) s(R.string.formats_help, resolutions.size, frameRates.size)
            else s(if (selectedMode == SessionMode.HIGH_SPEED) R.string.no_high_speed else R.string.no_camera_options)
        updateEnabled()
    }
    private fun currentOption(): CaptureOption? = options.firstOrNull { it.key == selectedOptionKey }
    private fun selectedCrop(): CropPercent? {
        val values = cropInputs.map { it.text.toString().trim().replace(',', '.').toDoubleOrNull() ?: return null }
        return runCatching { CropPercent(values[0], values[1], values[2], values[3]) }.getOrNull()
    }
    private fun updateOutputSize() {
        if (!::outputSizeText.isInitialized) return
        val option = currentOption() ?: return
        val crop = selectedCrop()
        val rotation = runCatching { frameRotation(option) }.getOrNull() ?: return
        if (crop == null) { outputSizeText.text = s(R.string.crop_invalid); return }
        val pixels = crop.pixels(option.width, option.height, rotation)
        val swap = rotation == 90 || rotation == 270
        outputSizeText.text = s(R.string.crop_output,
            if (swap) pixels.height else pixels.width, if (swap) pixels.width else pixels.height, rotation)
    }
    private fun updateEnabled() {
        if (!::startButton.isInitialized) return
        val idle = stream == null
        listOf(hostInput, portInput, cameraInput, rotationInput, highSpeedButton, cameraModeButton, changePcButton)
            .forEach { it.isEnabled = idle && !loadingCameras }
        resolutionInput.isEnabled = idle && !loadingCameras && resolutions.isNotEmpty()
        frameRateInput.isEnabled = idle && !loadingCameras && frameRates.isNotEmpty()
        jpegOptions.visibility = if (selectedMode == SessionMode.CAMERA) View.VISIBLE else View.GONE
        jpegQualityInput.isEnabled = idle && !loadingCameras
        cropInputs.forEach { it.isEnabled = idle && !loadingCameras }
        updateOutputSize()
        refreshButton.isEnabled = idle && !loadingCameras
        discoveryButton.isEnabled = idle
        startButton.isEnabled = idle && !loadingCameras && (currentOption() != null || !hasCameraPermission())
        stopButton.isEnabled = !idle
    }
    private fun setStatus(id: Int, vararg args: Any) {
        statusId = id; statusArgs = arrayOf(*args)
        if (::statusText.isInitialized) statusText.text = s(id, *args)
    }
    private fun setConnection(id: Int, vararg args: Any) {
        connectionId = id; connectionArgs = arrayOf(*args); connectionText.text = s(id, *args)
    }
    private fun ensureCameraAndStart() {
        if (!hasCameraPermission()) {
            startAfterPermission = true; permission.launch(Manifest.permission.CAMERA)
        } else startStreaming()
    }

    @Suppress("DEPRECATION")
    private fun frameRotation(option: CaptureOption): Int {
        val override = rotationValues.getOrElse(rotationInput.selectedItemPosition) { -1 }
        if (override >= 0) return override
        val chars = (getSystemService(Context.CAMERA_SERVICE) as CameraManager).getCameraCharacteristics(option.cameraId)
        val sensor = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
        val display = when (windowManager.defaultDisplay.rotation) {
            Surface.ROTATION_90 -> 90; Surface.ROTATION_180 -> 180; Surface.ROTATION_270 -> 270; else -> 0
        }
        return if (chars.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT) (sensor + display) % 360
        else (sensor - display + 360) % 360
    }
    private fun startStreaming() {
        if (stream != null || destroyed) return
        val option = currentOption() ?: return
        val host = hostInput.text.toString().trim()
        if (host.isBlank() || host.any { it.isWhitespace() }) { setStatus(R.string.host_required); hostInput.requestFocus(); return }
        val port = portInput.text.toString().toIntOrNull()
        if (port == null || port !in 1..65535) { connectionOptions.visibility = View.VISIBLE; setStatus(R.string.port_invalid); portInput.requestFocus(); return }
        val rotation = runCatching { frameRotation(option) }.getOrElse { setStatus(R.string.camera_read_failed); return }
        val crop = if (option.mode == SessionMode.CAMERA) selectedCrop() else CropPercent()
        if (crop == null) { setStatus(R.string.crop_invalid); return }
        val quality = JpegQuality.levels.getOrElse(jpegQualityInput.selectedItemPosition) { JpegQuality.DEFAULT }
        val cropPrefs = prefs.edit().putInt("camera_jpeg_quality", quality)
        listOf("left", "right", "top", "bottom").zip(cropInputs).forEach { (edge, input) ->
            cropPrefs.putString("display_crop_$edge", input.text.toString().trim())
        }
        cropPrefs.apply()
        prefs.edit().putString("host", host).putString("port", port.toString())
            .putString("camera_id", option.cameraId).putString("session_mode", selectedMode.name)
            .putString(optionPreferenceKey(), option.key)
            .putInt("rotation_mode", rotationValues.getOrElse(rotationInput.selectedItemPosition) { -1 }).apply()
        val generation = ++streamGeneration
        currentRates = null; renderRates()
        stream = CameraStream(this, option, host, port, rotation, crop, quality,
            onReady = { if (!destroyed && generation == streamGeneration) setStatus(R.string.streaming, option.label) },
            onRates = { rates -> if (!destroyed && generation == streamGeneration) { currentRates = rates; renderRates() } },
            onFailure = { failure, detail ->
                if (!destroyed && generation == streamGeneration) {
                    Log.w("OpenGazeLink", failure.name + ": " + detail)
                    stopStreaming()
                    if (failure == StreamFailure.COMBINATION) { excludedCombinations += option.key; rebuildOptions() }
                    setStatus(when (failure) {
                        StreamFailure.COMBINATION -> R.string.combination_failed
                        StreamFailure.CONNECTION -> R.string.connection_failed
                        StreamFailure.INTRINSICS -> R.string.intrinsics_failed
                        StreamFailure.CAMERA -> R.string.camera_failed
                    })
                }
            })
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setStatus(R.string.opening, option.label)
        updateEnabled()
        stream?.start()
    }
    private fun stopStreaming() {
        streamGeneration++
        stream?.close(); stream = null
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        if (!destroyed) { currentRates = null; renderRates(); updateEnabled() }
    }
    private fun renderRates() {
        if (!::statsSummary.isInitialized) return
        val rates = currentRates
        statsSummary.text = if (rates == null) s(R.string.not_available) else s(R.string.performance_summary, rates.captureFps, rates.sentFps)
        fun number(value: Double?, unit: String): String = value?.let { String.format(Locale.getDefault(), "%.1f %s", it, unit) } ?: "—"
        captureRate.text = number(rates?.captureFps, "FPS"); sendRate.text = number(rates?.sentFps, "FPS")
        videoRate.text = number(rates?.mbps, "Mbps")
        encodingAge.text = number(rates?.encodingAgeMs, "ms"); exposure.text = number(rates?.exposureMs, "ms")
    }

    private fun phoneId(): String {
        val id = prefs.getString("phone_id", "").orEmpty()
        if (id.isNotBlank()) return id
        return UUID.randomUUID().toString().also { prefs.edit().putString("phone_id", it).apply() }
    }
    private fun startPcDiscovery(manual: Boolean = false, restart: Boolean = false) {
        if (destroyed) return
        if (!restart && discoverySocket?.isClosed == false) {
            if (manual) setConnection(R.string.discovery_active)
            return
        }
        discoverySocket?.close()
        val generation = ++discoveryGeneration
        val id = phoneId()
        val nonce = UUID.randomUUID().toString()
        if (manual) setConnection(R.string.discovery_active)
        Thread({
            try {
                DatagramSocket().use { socket ->
                    if (generation != discoveryGeneration) return@Thread
                    discoverySocket = socket; socket.broadcast = true; socket.soTimeout = 900
                    while (!destroyed && generation == discoveryGeneration) {
                        val instance = prefs.getString("paired_instance_id", "").orEmpty()
                        val query = JSONObject().put("magic", DISCOVERY_MAGIC).put("version", 1).put("type", "discover")
                            .put("nonce", nonce).put("phone_id", id).put("phone_name", (Build.MANUFACTURER + " " + Build.MODEL).trim())
                            .put("instance_id", instance).toString().toByteArray(Charsets.UTF_8)
                        socket.send(DatagramPacket(query, query.size, InetAddress.getByName("255.255.255.255"), 5006))
                        // Broadcast can be filtered on tethering/VPN interfaces.
                        // Refresh the saved pairing directly as well after PC restart.
                        val savedHost = prefs.getString("host", "").orEmpty()
                        if (savedHost.isNotBlank() && savedHost != "0.0.0.0") runCatching {
                            socket.send(DatagramPacket(query, query.size, InetAddress.getByName(savedHost), 5006))
                        }
                        val response = DatagramPacket(ByteArray(4096), 4096)
                        try { socket.receive(response) } catch (_: SocketTimeoutException) { continue }
                        val offer = runCatching { JSONObject(String(response.data, response.offset, response.length, Charsets.UTF_8)) }.getOrNull() ?: continue
                        if (offer.optString("magic") != DISCOVERY_MAGIC || offer.optInt("version") != 1 ||
                            offer.optString("type") != "offer" || offer.optString("nonce") != nonce) continue
                        val instanceId = offer.optString("instance_id")
                        if (instance.isNotEmpty() && instance != instanceId) continue
                        val address = response.address.hostAddress ?: continue
                        val name = offer.optString("pc_name", "OpenGazeLink")
                        val port = offer.optInt("data_port", 5007)
                        if (port !in 1..65535) continue
                        runOnUiThread {
                            if (destroyed || generation != discoveryGeneration) return@runOnUiThread
                            if (!offer.optBoolean("accepted")) setConnection(R.string.pc_found, name, address)
                            else if (stream != null && (hostInput.text.toString().trim() != address || portInput.text.toString() != port.toString())) {
                                setConnection(R.string.pairing_changed)
                            } else {
                                if (hostInput.text.toString() != address) hostInput.setText(address)
                                if (portInput.text.toString() != port.toString()) portInput.setText(String.format(Locale.ROOT, "%d", port))
                                prefs.edit().putString("paired_instance_id", instanceId).putString("host", address).putString("port", port.toString()).apply()
                                setConnection(R.string.paired, name, address, port)
                            }
                        }
                        Thread.sleep(3000)
                    }
                }
            } catch (error: Exception) {
                if (!destroyed && generation == discoveryGeneration) runOnUiThread { if (!destroyed) setConnection(R.string.discovery_failed) }
            }
        }, "phone-pc-discovery").apply { isDaemon = true; start() }
    }

    companion object { private const val DISCOVERY_MAGIC = "EYETRACING_DISCOVERY_V1" }
}
