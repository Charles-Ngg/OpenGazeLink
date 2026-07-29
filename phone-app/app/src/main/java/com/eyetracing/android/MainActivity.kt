package com.eyetracing.android

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.Image
import android.media.ImageReader
import android.os.Bundle
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.text.InputType
import android.util.Range
import android.util.Size
import android.util.SizeF
import android.view.Gravity
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import android.widget.ArrayAdapter
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.net.SocketTimeoutException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt
import java.util.ArrayDeque
import java.util.UUID

private data class CropPercent(
    val left: Double,
    val right: Double,
    val top: Double,
    val bottom: Double,
) {
    fun validated(): CropPercent = CropPercent(
        left.coerceIn(0.0, 45.0),
        right.coerceIn(0.0, 45.0),
        top.coerceIn(0.0, 45.0),
        bottom.coerceIn(0.0, 45.0),
    )

    fun pixels(width: Int, height: Int): PixelCrop {
        fun even(value: Int): Int = value and -2
        val x = even((width * left / 100.0).roundToInt()).coerceIn(0, width - 4)
        val y = even((height * top / 100.0).roundToInt()).coerceIn(0, height - 4)
        val rightInset = even((width * right / 100.0).roundToInt()).coerceIn(0, width - x - 4)
        val bottomInset = even((height * bottom / 100.0).roundToInt()).coerceIn(0, height - y - 4)
        val outputWidth = even(width - x - rightInset).coerceAtLeast(4)
        val outputHeight = even(height - y - bottomInset).coerceAtLeast(4)
        return PixelCrop(x, y, outputWidth, outputHeight)
    }
}

private data class PixelCrop(val x: Int, val y: Int, val width: Int, val height: Int)
private data class Nv21Frame(
    val bytes: ByteArray,
    val width: Int,
    val height: Int,
    val crop: PixelCrop,
    val sensorTimeNs: Long = 0L,
)
private data class TransportFrame(
    val bytes: ByteArray,
    val width: Int,
    val height: Int,
    val format: Byte,
    val sensorTimeNs: Long,
)
private data class FrameSendStats(val bytes: Int, val chunks: Int)

private data class WindowSummary(
    val count: Long,
    val mean: Double,
    val standardDeviation: Double,
    val minimum: Double,
    val maximum: Double,
)

private class NumericWindow {
    private var count = 0L
    private var sum = 0.0
    private var sumSquares = 0.0
    private var minimum = Double.POSITIVE_INFINITY
    private var maximum = Double.NEGATIVE_INFINITY

    @Synchronized
    fun add(value: Double) {
        if (!value.isFinite()) return
        count += 1
        sum += value
        sumSquares += value * value
        minimum = min(minimum, value)
        maximum = max(maximum, value)
    }

    @Synchronized
    fun reset() {
        count = 0L
        sum = 0.0
        sumSquares = 0.0
        minimum = Double.POSITIVE_INFINITY
        maximum = Double.NEGATIVE_INFINITY
    }

    @Synchronized
    fun drain(): WindowSummary? {
        if (count == 0L) return null
        val mean = sum / count
        val variance = max(0.0, sumSquares / count - mean * mean)
        val result = WindowSummary(count, mean, sqrt(variance), minimum, maximum)
        reset()
        return result
    }
}

private class ReusableByteArrayOutputStream(initialSize: Int) : ByteArrayOutputStream(initialSize) {
    val backingBuffer: ByteArray get() = buf
}

class MainActivity : ComponentActivity() {
    private lateinit var hostInput: EditText
    private lateinit var portInput: EditText
    private lateinit var widthInput: EditText
    private lateinit var heightInput: EditText
    private lateinit var cameraFacingInput: Spinner
    private lateinit var rotationInput: Spinner
    private lateinit var fpsInput: Spinner
    private lateinit var packetModeInput: Spinner
    private lateinit var encodingModeInput: Spinner
    private lateinit var processingModeInput: Spinner
    private lateinit var jpegQualityInput: EditText
    private lateinit var cropLeftInput: EditText
    private lateinit var cropRightInput: EditText
    private lateinit var cropTopInput: EditText
    private lateinit var cropBottomInput: EditText
    private lateinit var statusText: TextView
    private lateinit var statsText: TextView
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var capsButton: Button
    private lateinit var intrinsicsButton: Button
    private lateinit var discoveryButton: Button
    private lateinit var cropLabel: TextView

    private var cameraThread: HandlerThread? = null
    private var cameraHandler: Handler? = null
    private var transportThread: HandlerThread? = null
    private var transportHandler: Handler? = null
    private var imageReader: ImageReader? = null
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var sender: UdpYuvSender? = null
    private var running = false
    private var activeCameraId: String? = null
    private var activeCameraCharacteristics: CameraCharacteristics? = null
    private var activeTimestampSource: Int? = null
    private var activeCropRegion: Rect? = null
    private var activeStreamWidth = 0
    private var activeStreamHeight = 0
    private var activeOutputWidth = 0
    private var activeOutputHeight = 0
    private var activeJpegQuality = 95
    private var activeTargetFps = 30
    private var activeFrameRotation = 270
    private var activeLensFacing = CameraCharacteristics.LENS_FACING_FRONT
    private var activeChunkPayloadSize = 1400
    private var activeHardwareJpeg = false
    private var activeLowLatencyProcessing = true
    private var activeHardwareCropRegion: Rect? = null
    private var activeSoftwareCrop = CropPercent(left = 45.0, right = 0.0, top = 0.0, bottom = 0.0)
    private var planarURow = ByteArray(0)
    private var planarVRow = ByteArray(0)
    private var useInterleavedVuFastPath: Boolean? = null
    private var yuvConversionPath = "uninitialized"
    private var jpegOutput = ReusableByteArrayOutputStream(256 * 1024)
    private var nextSendSensorTimeNs = 0L
    private val frameQueueLock = Any()
    private val frameBufferPool = ArrayDeque<ByteArray>()
    private var pendingFrame: TransportFrame? = null
    private var transportWorkScheduled = false
    @Volatile private var discoveryStopped = false
    private var discoveryThread: Thread? = null

    private var frameCount = 0L
    private var byteCount = 0L
    private var dropCount = 0L
    private var throttledFrameCount = 0L
    private var latestOnlyDropCount = 0L
    private var convertedFrameCount = 0L
    private var chunkCount = 0L
    private var conversionTimeNs = 0L
    private var encodeTimeNs = 0L
    private var sendTimeNs = 0L
    private val cameraAgeWindow = NumericWindow()
    private val exposureWindow = NumericWindow()
    private val frameDurationWindow = NumericWindow()
    private val rollingShutterWindow = NumericWindow()
    private val pipelineDepthWindow = NumericWindow()
    private var lastStatsAtMs = 0L
    private var lastStatsFrameCount = 0L
    private var lastStatsByteCount = 0L
    private var lastStatsConversionTimeNs = 0L
    private var lastStatsEncodeTimeNs = 0L
    private var lastStatsSendTimeNs = 0L
    private var lastStatsChunkCount = 0L
    private var lastStatsThrottledFrameCount = 0L
    private var lastStatsConvertedFrameCount = 0L

    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startStreaming() else setStatus("Camera permission denied.")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        buildUi()
        loadPrefs()
        startPcDiscovery(manual = false)
    }

    override fun onDestroy() {
        discoveryStopped = true
        stopStreaming()
        super.onDestroy()
    }

    private fun buildUi() {
        hostInput = editText("Windows USB IP, e.g. 192.168.42.xxx").apply {
            setText("")
        }
        portInput = editText("5007").apply {
            inputType = InputType.TYPE_CLASS_NUMBER
            setText("5007")
        }
        widthInput = editText("640").apply {
            inputType = InputType.TYPE_CLASS_NUMBER
            setText("640")
        }
        heightInput = editText("480").apply {
            inputType = InputType.TYPE_CLASS_NUMBER
            setText("480")
        }
        cameraFacingInput = spinner(listOf("Front camera", "Rear / main camera"))
        rotationInput = spinner(listOf("0 deg", "90 deg", "180 deg", "270 deg"), 3)
        fpsInput = spinner(listOf("15 FPS", "20 FPS", "30 FPS", "45 FPS", "60 FPS"), 2)
        packetModeInput = spinner(listOf("Fast LAN (1400-byte payload)", "Compatible (1200-byte payload)"))
        encodingModeInput = spinner(listOf("Software YUV crop + JPEG", "Camera2 hardware JPEG"))
        processingModeInput = spinner(listOf("Low-latency realtime", "Camera defaults"))
        jpegQualityInput = editText("95").apply {
            inputType = InputType.TYPE_CLASS_NUMBER
            setText("95")
        }
        cropLeftInput = decimalEditText("45")
        cropRightInput = decimalEditText("0")
        cropTopInput = decimalEditText("0")
        cropBottomInput = decimalEditText("0")
        statusText = TextView(this).apply {
            text = "Idle. Start after Windows receiver is listening."
            setTextColor(0xFFEFF6F0.toInt())
            textSize = 14f
        }
        statsText = TextView(this).apply {
            text = "-"
            setTextColor(0xFFB5C0B4.toInt())
            textSize = 13f
        }
        startButton = Button(this).apply {
            text = "Start UDP Camera"
            setOnClickListener { ensureCameraPermissionAndStart() }
        }
        stopButton = Button(this).apply {
            text = "Stop"
            setOnClickListener { stopStreaming() }
        }
        capsButton = Button(this).apply {
            text = "Show Camera Caps"
            setOnClickListener { showCameraCaps() }
        }
        intrinsicsButton = Button(this).apply {
            text = "Send Camera Intrinsics"
            setOnClickListener { sendCurrentIntrinsics(manual = true) }
        }
        discoveryButton = Button(this).apply {
            text = "Discover / pair PC"
            setOnClickListener { startPcDiscovery(manual = true) }
        }

        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 24, 24, 24)
            setBackgroundColor(0xFF101411.toInt())
            addView(title("OpenGazeLink Phone Camera v0.11"))
            addView(help("Camera2 -> selectable software or hardware JPEG -> paced UDP. Rotation is applied on the PC."))
            addView(label("Windows host"))
            addView(hostInput)
            addView(discoveryButton)
            addView(label("UDP port"))
            addView(portInput)
            addView(rowWithLabels("Width", widthInput, "Height", heightInput))
            addView(label("Camera"))
            addView(cameraFacingInput)
            addView(label("Output rotation hint"))
            addView(rotationInput)
            addView(label("Transmission frame rate"))
            addView(fpsInput)
            addView(label("UDP packet mode"))
            addView(packetModeInput)
            addView(label("JPEG encoder"))
            addView(encodingModeInput)
            addView(label("Camera processing"))
            addView(processingModeInput)
            addView(label("JPEG quality (1-100, 0 = raw NV21)"))
            addView(jpegQualityInput)
            cropLabel = label("Crop (%) - software copy or hardware sensor crop")
            addView(cropLabel)
            addView(rowWithLabels("Left", cropLeftInput, "Right", cropRightInput))
            addView(rowWithLabels("Top", cropTopInput, "Bottom", cropBottomInput))
            addView(row(startButton, stopButton))
            addView(row(capsButton, intrinsicsButton))
            addView(statusText)
            addView(statsText)
        }

        setContentView(ScrollView(this).apply { addView(controls) })
    }

    private fun editText(hintText: String): EditText {
        return EditText(this).apply {
            hint = hintText
            setSingleLine(true)
            setTextColor(0xFFEFF6F0.toInt())
            setHintTextColor(0xFF9EAA9E.toInt())
        }
    }

    private fun decimalEditText(defaultValue: String): EditText = editText(defaultValue).apply {
        inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
        setText(defaultValue)
    }

    private fun spinner(items: List<String>, selection: Int = 0): Spinner {
        return Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                items,
            )
            setSelection(selection)
        }
    }

    private fun title(text: String): TextView {
        return TextView(this).apply {
            this.text = text
            setTextColor(0xFFEFF6F0.toInt())
            textSize = 20f
            gravity = Gravity.CENTER_HORIZONTAL
        }
    }

    private fun help(text: String): TextView {
        return TextView(this).apply {
            this.text = text
            setTextColor(0xFFB5C0B4.toInt())
            textSize = 13f
            setPadding(0, 12, 0, 16)
        }
    }

    private fun label(text: String): TextView {
        return TextView(this).apply {
            this.text = text
            setTextColor(0xFFB5C0B4.toInt())
            textSize = 12f
        }
    }

    private fun row(left: Button, right: Button): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(left, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(right, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        }
    }

    private fun rowWithLabels(leftLabel: String, left: EditText, rightLabel: String, right: EditText): LinearLayout {
        val leftBox = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(label(leftLabel))
            addView(left)
        }
        val rightBox = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(label(rightLabel))
            addView(right)
        }
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(leftBox, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(rightBox, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        }
    }

    private fun loadPrefs() {
        val prefs = getSharedPreferences("eyetracing_yuv_sender", MODE_PRIVATE)
        hostInput.setText(prefs.getString("host", hostInput.text.toString()))
        portInput.setText(prefs.getString("port", portInput.text.toString()))
        widthInput.setText(prefs.getString("width", widthInput.text.toString()))
        heightInput.setText(prefs.getString("height", heightInput.text.toString()))
        cameraFacingInput.setSelection(if (prefs.getString("camera_facing", "front") == "back") 1 else 0)
        rotationInput.setSelection(listOf(0, 90, 180, 270).indexOf(prefs.getInt("frame_rotation", 270)).coerceAtLeast(0))
        val savedFps = prefs.getString("fps", "30")?.toIntOrNull() ?: 30
        fpsInput.setSelection(listOf(15, 20, 30, 45, 60).indexOf(savedFps).coerceAtLeast(0))
        packetModeInput.setSelection(if (prefs.getInt("chunk_payload_size", 1400) == 1200) 1 else 0)
        encodingModeInput.setSelection(if (prefs.getString("encoding_mode", "software") == "software") 0 else 1)
        processingModeInput.setSelection(if (prefs.getString("processing_mode", "low_latency") == "default") 1 else 0)
        jpegQualityInput.setText(prefs.getString("jpeg_quality", jpegQualityInput.text.toString()))
        cropLeftInput.setText(prefs.getString("crop_left", cropLeftInput.text.toString()))
        cropRightInput.setText(prefs.getString("crop_right", cropRightInput.text.toString()))
        cropTopInput.setText(prefs.getString("crop_top", cropTopInput.text.toString()))
        cropBottomInput.setText(prefs.getString("crop_bottom", cropBottomInput.text.toString()))
    }

    private fun savePrefs() {
        getSharedPreferences("eyetracing_yuv_sender", MODE_PRIVATE)
            .edit()
            .putString("host", hostInput.text.toString().trim())
            .putString("port", portInput.text.toString().trim())
            .putString("width", widthInput.text.toString().trim())
            .putString("height", heightInput.text.toString().trim())
            .putString("camera_facing", if (cameraFacingInput.selectedItemPosition == 1) "back" else "front")
            .putInt("frame_rotation", selectedRotation())
            .putString("fps", selectedFps().toString())
            .putInt("chunk_payload_size", selectedChunkPayloadSize())
            .putString("encoding_mode", if (encodingModeInput.selectedItemPosition == 1) "hardware" else "software")
            .putString("processing_mode", if (processingModeInput.selectedItemPosition == 1) "default" else "low_latency")
            .putString("jpeg_quality", jpegQualityInput.text.toString().trim())
            .putString("crop_left", cropLeftInput.text.toString().trim())
            .putString("crop_right", cropRightInput.text.toString().trim())
            .putString("crop_top", cropTopInput.text.toString().trim())
            .putString("crop_bottom", cropBottomInput.text.toString().trim())
            .apply()
    }

    private fun selectedFps(): Int = listOf(15, 20, 30, 45, 60)
        .getOrElse(fpsInput.selectedItemPosition) { 30 }

    private fun selectedRotation(): Int = listOf(0, 90, 180, 270)
        .getOrElse(rotationInput.selectedItemPosition) { 270 }

    private fun selectedChunkPayloadSize(): Int = if (packetModeInput.selectedItemPosition == 1) 1200 else 1400

    private fun phoneId(): String {
        val prefs = getSharedPreferences("eyetracing_yuv_sender", MODE_PRIVATE)
        val existing = prefs.getString("phone_id", "").orEmpty()
        if (existing.isNotBlank()) return existing
        val generated = UUID.randomUUID().toString()
        prefs.edit().putString("phone_id", generated).apply()
        return generated
    }

    private fun startPcDiscovery(manual: Boolean) {
        if (discoveryThread?.isAlive == true) {
            if (manual) setStatus("PC discovery is already running.")
            return
        }
        discoveryStopped = false
        val prefs = getSharedPreferences("eyetracing_yuv_sender", MODE_PRIVATE)
        var knownInstance = prefs.getString("paired_instance_id", "").orEmpty()
        val phoneId = phoneId()
        val nonce = UUID.randomUUID().toString()
        discoveryThread = Thread({
            var lastAcceptedAddress = ""
            try {
                DatagramSocket().use { socket ->
                    socket.broadcast = true
                    socket.soTimeout = 700
                    while (!discoveryStopped) {
                        val request = JSONObject()
                            .put("magic", DISCOVERY_MAGIC)
                            .put("type", "discover")
                            .put("version", DISCOVERY_VERSION)
                            .put("nonce", nonce)
                            .put("phone_id", phoneId)
                            .put("phone_name", "${Build.MANUFACTURER} ${Build.MODEL}".trim())
                            .put("instance_id", knownInstance)
                            .toString().toByteArray(Charsets.UTF_8)
                        socket.send(DatagramPacket(
                            request, request.size,
                            InetAddress.getByName("255.255.255.255"), DISCOVERY_PORT,
                        ))
                        val buffer = ByteArray(4096)
                        val responsePacket = DatagramPacket(buffer, buffer.size)
                        try {
                            socket.receive(responsePacket)
                        } catch (_: SocketTimeoutException) {
                            continue
                        }
                        val response = JSONObject(String(
                            responsePacket.data, responsePacket.offset,
                            responsePacket.length, Charsets.UTF_8,
                        ))
                        if (response.optString("magic") != DISCOVERY_MAGIC
                            || response.optString("type") != "offer"
                            || response.optInt("version") != DISCOVERY_VERSION
                            || response.optString("nonce") != nonce) continue
                        val instanceId = response.optString("instance_id")
                        if (knownInstance.isNotBlank() && instanceId != knownInstance) continue
                        val pcName = response.optString("pc_name", "OpenGazeLink PC")
                        val address = responsePacket.address.hostAddress ?: continue
                        if (!response.optBoolean("accepted", false)) {
                            setStatus("Found $pcName at $address. Confirm this phone in the PC control page.")
                            continue
                        }
                        val dataPort = response.optInt("data_port", 5007)
                        knownInstance = instanceId
                        if (address != lastAcceptedAddress) {
                            lastAcceptedAddress = address
                            runOnUiThread {
                                val oldHost = hostInput.text.toString().trim()
                                val restart = running && oldHost != address
                                if (restart) stopStreaming()
                                hostInput.setText(address)
                                portInput.setText(dataPort.toString())
                                getSharedPreferences("eyetracing_yuv_sender", MODE_PRIVATE).edit()
                                    .putString("paired_instance_id", instanceId)
                                    .putString("host", address)
                                    .putString("port", dataPort.toString())
                                    .apply()
                                setStatus("Paired with $pcName at $address:$dataPort.")
                                if (restart) ensureCameraPermissionAndStart()
                            }
                        }
                        Thread.sleep(4_000L)
                    }
                }
            } catch (error: Exception) {
                if (manual && !discoveryStopped) setStatus("PC discovery failed: ${error.message}")
            }
        }, "pc-discovery").apply { start() }
    }

    private fun ensureCameraPermissionAndStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startStreaming()
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startStreaming() {
        if (running) return
        val host = hostInput.text.toString().trim()
        val port = portInput.text.toString().trim().toIntOrNull() ?: 5007
        val width = widthInput.text.toString().trim().toIntOrNull() ?: 640
        val height = heightInput.text.toString().trim().toIntOrNull() ?: 480
        val fps = selectedFps()
        val lensFacing = if (cameraFacingInput.selectedItemPosition == 1) {
            CameraCharacteristics.LENS_FACING_BACK
        } else {
            CameraCharacteristics.LENS_FACING_FRONT
        }
        val frameRotation = selectedRotation()
        val chunkPayloadSize = selectedChunkPayloadSize()
        val jpegQuality = (jpegQualityInput.text.toString().trim().toIntOrNull() ?: 95).coerceIn(0, 100)
        val preferHardwareJpeg = encodingModeInput.selectedItemPosition == 1 && jpegQuality > 0
        val softwareCrop = CropPercent(
            left = cropLeftInput.text.toString().trim().toDoubleOrNull() ?: 45.0,
            right = cropRightInput.text.toString().trim().toDoubleOrNull() ?: 0.0,
            top = cropTopInput.text.toString().trim().toDoubleOrNull() ?: 0.0,
            bottom = cropBottomInput.text.toString().trim().toDoubleOrNull() ?: 0.0,
        ).validated()
        if (host.isBlank()) {
            setStatus("Windows host is empty.")
            return
        }
        savePrefs()
        frameCount = 0L
        byteCount = 0L
        dropCount = 0L
        throttledFrameCount = 0L
        latestOnlyDropCount = 0L
        convertedFrameCount = 0L
        chunkCount = 0L
        conversionTimeNs = 0L
        encodeTimeNs = 0L
        sendTimeNs = 0L
        cameraAgeWindow.reset()
        exposureWindow.reset()
        frameDurationWindow.reset()
        rollingShutterWindow.reset()
        pipelineDepthWindow.reset()
        lastStatsAtMs = SystemClock.elapsedRealtime()
        lastStatsFrameCount = 0L
        lastStatsByteCount = 0L
        lastStatsConversionTimeNs = 0L
        lastStatsEncodeTimeNs = 0L
        lastStatsSendTimeNs = 0L
        lastStatsChunkCount = 0L
        lastStatsThrottledFrameCount = 0L
        lastStatsConvertedFrameCount = 0L
        nextSendSensorTimeNs = 0L
        useInterleavedVuFastPath = null
        yuvConversionPath = "detecting"
        running = true
        activeStreamWidth = width
        activeStreamHeight = height
        activeJpegQuality = jpegQuality
        activeTargetFps = fps
        activeFrameRotation = frameRotation
        activeLensFacing = lensFacing
        activeChunkPayloadSize = chunkPayloadSize
        activeSoftwareCrop = softwareCrop
        activeHardwareJpeg = preferHardwareJpeg
        activeLowLatencyProcessing = processingModeInput.selectedItemPosition == 0
        activeHardwareCropRegion = null
        activeOutputWidth = 0
        activeOutputHeight = 0
        cameraThread = HandlerThread("camera2-yuv-sender").also { it.start() }
        cameraHandler = Handler(cameraThread!!.looper)
        transportThread = HandlerThread("jpeg-udp-transport").also { it.start() }
        transportHandler = Handler(transportThread!!.looper)
        val transport = when {
            preferHardwareJpeg -> "Camera2 hardware JPEG Q$jpegQuality"
            jpegQuality > 0 -> "software JPEG Q$jpegQuality"
            else -> "raw NV21"
        }
        setStatus(
            "Opening ${lensFacingName(lensFacing)} camera $width x $height @ $fps, $transport to $host:$port ...\n" +
                "rotation=$frameRotation, UDP payload=$chunkPayloadSize bytes\n" +
                "YUV crop L${softwareCrop.left}% R${softwareCrop.right}% " +
                "T${softwareCrop.top}% B${softwareCrop.bottom}%"
        )
        cameraHandler?.post {
            runCatching {
                sender = UdpYuvSender(host, port, chunkPayloadSize)
                openCamera(width, height, fps, lensFacing, preferHardwareJpeg)
            }.onFailure { error ->
                setStatus("Start streaming failed: ${error.javaClass.simpleName}: ${error.message}")
                stopStreaming()
            }
        }
    }

    private fun openCamera(width: Int, height: Int, fps: Int, lensFacing: Int, preferHardwareJpeg: Boolean) {
        val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val cameraId = findCamera(manager, lensFacing) ?: run {
            setStatus("No ${lensFacingName(lensFacing)} camera found.")
            stopStreaming()
            return
        }
        val characteristics = manager.getCameraCharacteristics(cameraId)
        activeCameraId = cameraId
        activeCameraCharacteristics = characteristics
        activeTimestampSource = characteristics.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)
        activeCropRegion = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
        val fpsRange = chooseFpsRange(characteristics, fps)
        val jpegSize = if (preferHardwareJpeg) {
            chooseHardwareJpegSize(characteristics, activeSoftwareCrop.pixels(width, height))
        } else null
        activeHardwareJpeg = preferHardwareJpeg && jpegSize != null
        if (activeHardwareJpeg) {
            val active = activeCropRegion ?: error("Camera has no active array")
            val size = jpegSize!!
            activeHardwareCropRegion = hardwareCropRegion(
                active,
                size,
                activeSoftwareCrop,
            )
            activeOutputWidth = size.width
            activeOutputHeight = size.height
            yuvConversionPath = "camera2-hardware-jpeg"
            imageReader = ImageReader.newInstance(size.width, size.height, ImageFormat.JPEG, 2).apply {
                setOnImageAvailableListener({ reader ->
                    val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                    handleHardwareJpeg(image)
                }, cameraHandler)
            }
            setStatus(
                "Using Camera2 hardware JPEG ${size.width}x${size.height}; " +
                    "sensor crop=${activeHardwareCropRegion?.width()}x${activeHardwareCropRegion?.height()}"
            )
        } else {
            activeHardwareCropRegion = null
            yuvConversionPath = "detecting"
            if (preferHardwareJpeg) {
                setStatus("No Camera2 JPEG output size; falling back to software YUV JPEG.")
            }
            createSoftwareImageReader(width, height)
        }
        try {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return
            manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraDevice = camera
                    createCaptureSession(camera, fpsRange)
                }

                override fun onDisconnected(camera: CameraDevice) {
                    camera.close()
                    setStatus("Camera disconnected.")
                    stopStreaming()
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    camera.close()
                    setStatus("Camera error $error")
                    stopStreaming()
                }
            }, cameraHandler)
        } catch (error: Exception) {
            stopStreaming()
            setStatus("Open camera failed: ${error.message}")
        }
    }

    private fun createCaptureSession(camera: CameraDevice, fpsRange: Range<Int>) {
        val reader = imageReader ?: return
        try {
            val template = if (activeHardwareJpeg) {
                CameraDevice.TEMPLATE_STILL_CAPTURE
            } else {
                CameraDevice.TEMPLATE_RECORD
            }
            val requestBuilder = camera.createCaptureRequest(template).apply {
                addTarget(reader.surface)
                set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
                set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fpsRange)
                if (activeLowLatencyProcessing) {
                    applyLowLatencyProcessing(this, activeCameraCharacteristics)
                }
                if (activeHardwareJpeg) {
                    activeHardwareCropRegion?.let { set(CaptureRequest.SCALER_CROP_REGION, it) }
                    set(CaptureRequest.JPEG_QUALITY, activeJpegQuality.toByte())
                }
            }
            camera.createCaptureSession(listOf(reader.surface), object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    captureSession = session
                    session.setRepeatingRequest(
                        requestBuilder.build(),
                        object : CameraCaptureSession.CaptureCallback() {
                            override fun onCaptureCompleted(
                                session: CameraCaptureSession,
                                request: CaptureRequest,
                                result: TotalCaptureResult,
                            ) {
                                result.get(CaptureResult.SENSOR_EXPOSURE_TIME)
                                    ?.takeIf { it > 0L }
                                    ?.let { exposureWindow.add(it / 1_000_000.0) }
                                result.get(CaptureResult.SENSOR_FRAME_DURATION)
                                    ?.takeIf { it > 0L }
                                    ?.let { frameDurationWindow.add(it / 1_000_000.0) }
                                result.get(CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW)
                                    ?.takeIf { it > 0L }
                                    ?.let { rollingShutterWindow.add(it / 1_000_000.0) }
                                result.get(CaptureResult.REQUEST_PIPELINE_DEPTH)
                                    ?.toInt()
                                    ?.takeIf { it > 0 }
                                    ?.let { pipelineDepthWindow.add(it.toDouble()) }
                            }
                        },
                        cameraHandler,
                    )
                    setStatus("Streaming with AE FPS range ${fpsRange.lower}..${fpsRange.upper}.")
                    sendCurrentIntrinsics(manual = false)
                }

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    if (activeHardwareJpeg) {
                        setStatus("Hardware JPEG session failed; retrying software YUV JPEG.")
                        activeHardwareJpeg = false
                        activeHardwareCropRegion = null
                        activeOutputWidth = 0
                        activeOutputHeight = 0
                        runCatching {
                            imageReader?.close()
                            createSoftwareImageReader(activeStreamWidth, activeStreamHeight)
                            createCaptureSession(camera, fpsRange)
                        }.onFailure { error ->
                            stopStreaming()
                            setStatus("Software fallback failed: ${error.message}")
                        }
                    } else {
                        stopStreaming()
                        setStatus("Capture session configure failed.")
                    }
                }
            }, cameraHandler)
        } catch (error: Exception) {
            stopStreaming()
            setStatus("Create capture session failed: ${error.message}")
        }
    }

    private fun applyLowLatencyProcessing(
        request: CaptureRequest.Builder,
        chars: CameraCharacteristics?,
    ) {
        if (chars == null) return

        request.set(CaptureRequest.CONTROL_ENABLE_ZSL, false)
        setFirstSupportedMode(
            request,
            CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
            chars.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES),
            CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF,
        )
        setFirstSupportedMode(
            request,
            CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
            chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION),
            CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF,
        )
        setFirstSupportedMode(
            request,
            CaptureRequest.STATISTICS_FACE_DETECT_MODE,
            chars.get(CameraCharacteristics.STATISTICS_INFO_AVAILABLE_FACE_DETECT_MODES),
            CaptureRequest.STATISTICS_FACE_DETECT_MODE_OFF,
        )
        setFirstSupportedMode(
            request,
            CaptureRequest.NOISE_REDUCTION_MODE,
            chars.get(CameraCharacteristics.NOISE_REDUCTION_AVAILABLE_NOISE_REDUCTION_MODES),
            CaptureRequest.NOISE_REDUCTION_MODE_MINIMAL,
            CaptureRequest.NOISE_REDUCTION_MODE_OFF,
            CaptureRequest.NOISE_REDUCTION_MODE_FAST,
        )
        setFirstSupportedMode(
            request,
            CaptureRequest.EDGE_MODE,
            chars.get(CameraCharacteristics.EDGE_AVAILABLE_EDGE_MODES),
            CaptureRequest.EDGE_MODE_FAST,
            CaptureRequest.EDGE_MODE_OFF,
        )
        setFirstSupportedMode(
            request,
            CaptureRequest.TONEMAP_MODE,
            chars.get(CameraCharacteristics.TONEMAP_AVAILABLE_TONE_MAP_MODES),
            CaptureRequest.TONEMAP_MODE_FAST,
        )
        setFirstSupportedMode(
            request,
            CaptureRequest.SHADING_MODE,
            chars.get(CameraCharacteristics.SHADING_AVAILABLE_MODES),
            CaptureRequest.SHADING_MODE_FAST,
            CaptureRequest.SHADING_MODE_OFF,
        )
        setFirstSupportedMode(
            request,
            CaptureRequest.COLOR_CORRECTION_ABERRATION_MODE,
            chars.get(CameraCharacteristics.COLOR_CORRECTION_AVAILABLE_ABERRATION_MODES),
            CaptureRequest.COLOR_CORRECTION_ABERRATION_MODE_FAST,
            CaptureRequest.COLOR_CORRECTION_ABERRATION_MODE_OFF,
        )
        setFirstSupportedMode(
            request,
            CaptureRequest.HOT_PIXEL_MODE,
            chars.get(CameraCharacteristics.HOT_PIXEL_AVAILABLE_HOT_PIXEL_MODES),
            CaptureRequest.HOT_PIXEL_MODE_FAST,
            CaptureRequest.HOT_PIXEL_MODE_OFF,
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            setFirstSupportedMode(
                request,
                CaptureRequest.DISTORTION_CORRECTION_MODE,
                chars.get(CameraCharacteristics.DISTORTION_CORRECTION_AVAILABLE_MODES),
                CaptureRequest.DISTORTION_CORRECTION_MODE_FAST,
                CaptureRequest.DISTORTION_CORRECTION_MODE_OFF,
            )
        }
    }

    private fun setFirstSupportedMode(
        request: CaptureRequest.Builder,
        key: CaptureRequest.Key<Int>,
        available: IntArray?,
        vararg preferred: Int,
    ) {
        val selected = preferred.firstOrNull { mode -> available?.contains(mode) == true } ?: return
        request.set(key, selected)
    }

    private fun createSoftwareImageReader(width: Int, height: Int) {
        imageReader = ImageReader.newInstance(width, height, ImageFormat.YUV_420_888, 2).apply {
            setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                handleImage(image)
            }, cameraHandler)
        }
    }

    private fun findCamera(manager: CameraManager, lensFacing: Int): String? {
        val candidates = manager.cameraIdList.filter { id ->
            val chars = manager.getCameraCharacteristics(id)
            chars.get(CameraCharacteristics.LENS_FACING) == lensFacing
        }
        if (lensFacing == CameraCharacteristics.LENS_FACING_BACK) {
            candidates.firstOrNull { id ->
                manager.getCameraCharacteristics(id)
                    .get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
                    ?.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA) == true
            }?.let { return it }
        }
        return candidates.firstOrNull()
    }

    private fun lensFacingName(lensFacing: Int?): String = when (lensFacing) {
        CameraCharacteristics.LENS_FACING_FRONT -> "front"
        CameraCharacteristics.LENS_FACING_BACK -> "rear"
        CameraCharacteristics.LENS_FACING_EXTERNAL -> "external"
        else -> "unknown"
    }

    private fun chooseFpsRange(chars: CameraCharacteristics, targetFps: Int): Range<Int> {
        val ranges = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
            ?: return Range(targetFps, targetFps)
        return ranges.minWithOrNull(
            compareBy<Range<Int>>(
                { if (targetFps in it.lower..it.upper) 0 else 1 },
                { kotlin.math.abs(it.upper - targetFps) },
                { kotlin.math.abs(it.lower - targetFps) },
                { -it.upper }
            )
        ) ?: Range(targetFps, targetFps)
    }

    private fun chooseHardwareJpegSize(chars: CameraCharacteristics, requested: PixelCrop): Size? {
        val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return null
        val sizes = map.getOutputSizes(ImageFormat.JPEG)?.filter { it.width >= 2 && it.height >= 2 } ?: return null
        if (sizes.isEmpty()) return null
        val targetArea = requested.width.toDouble() * requested.height.toDouble()
        val targetAspect = requested.width.toDouble() / requested.height.toDouble()
        return sizes.minByOrNull { size ->
            val aspectError = kotlin.math.abs(kotlin.math.ln(
                (size.width.toDouble() / size.height.toDouble()) / targetAspect
            ))
            val areaError = kotlin.math.abs(kotlin.math.ln(
                (size.width.toDouble() * size.height.toDouble()) / targetArea
            ))
            aspectError * 4.0 + areaError
        }
    }

    private fun hardwareCropRegion(
        active: Rect,
        output: Size,
        crop: CropPercent,
    ): Rect {
        val left = active.left + (active.width() * crop.left / 100.0).roundToInt()
        val top = active.top + (active.height() * crop.top / 100.0).roundToInt()
        val right = active.right - (active.width() * crop.right / 100.0).roundToInt()
        val bottom = active.bottom - (active.height() * crop.bottom / 100.0).roundToInt()
        val requested = Rect(left, top, right, bottom)
        val fitted = fitCropToAspect(requested, output.width, output.height)
        return Rect(
            fitted[0].roundToInt().coerceIn(active.left, active.right - 2),
            fitted[1].roundToInt().coerceIn(active.top, active.bottom - 2),
            (fitted[0] + fitted[2]).roundToInt().coerceIn(active.left + 2, active.right),
            (fitted[1] + fitted[3]).roundToInt().coerceIn(active.top + 2, active.bottom),
        )
    }

    private fun sendCurrentIntrinsics(manual: Boolean) {
        val host = hostInput.text.toString().trim()
        val port = portInput.text.toString().trim().toIntOrNull() ?: 5007
        if (host.isBlank()) {
            if (manual) setStatus("Windows host is empty.")
            return
        }
        runCatching {
            val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cameraId = activeCameraId ?: findCamera(manager, activeLensFacing)
                ?: error("No ${lensFacingName(activeLensFacing)} camera found.")
            val chars = activeCameraCharacteristics ?: manager.getCameraCharacteristics(cameraId)
            val width = activeOutputWidth.takeIf { it > 0 }
                ?: widthInput.text.toString().trim().toIntOrNull() ?: 640
            val height = activeOutputHeight.takeIf { it > 0 }
                ?: heightInput.text.toString().trim().toIntOrNull() ?: 480
            val payload = buildIntrinsicsPayload(
                cameraId,
                chars,
                width,
                height,
                activeHardwareCropRegion ?: activeCropRegion,
            )
            Thread {
                runCatching {
                    val activeSender = sender
                    if (activeSender != null) {
                        activeSender.sendIntrinsics(payload)
                    } else {
                        UdpYuvSender(host, port, selectedChunkPayloadSize()).use { it.sendIntrinsics(payload) }
                    }
                    val stream = payload.getJSONObject("streamIntrinsics")
                    setStatus(
                        "Sent ${payload.getString("source")} intrinsics to $host:$port\n" +
                            "${stream.getInt("width")}x${stream.getInt("height")}: " +
                            "fx=${"%.2f".format(stream.getDouble("fx"))} " +
                            "fy=${"%.2f".format(stream.getDouble("fy"))} " +
                            "cx=${"%.2f".format(stream.getDouble("cx"))} " +
                            "cy=${"%.2f".format(stream.getDouble("cy"))}"
                    )
                }.onFailure { error ->
                    setStatus("Send intrinsics failed: ${error.message}")
                }
            }.start()
        }.onFailure { error ->
            setStatus("Read camera intrinsics failed: ${error.message}")
        }
    }

    private fun buildIntrinsicsPayload(
        cameraId: String,
        chars: CameraCharacteristics,
        streamWidth: Int,
        streamHeight: Int,
        captureCrop: Rect?,
    ): JSONObject {
        val active = chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            ?: error("Camera has no active array")
        val preCorrection = chars.get(CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE)
            ?: active
        val pixelArray = chars.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
            ?: Size(preCorrection.width(), preCorrection.height())
        val factory = chars.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
        val focalMm = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull()
        val physical = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)

        val derived = if (focalMm != null && physical != null && physical.width > 0f && physical.height > 0f) {
            floatArrayOf(
                focalMm / physical.width * pixelArray.width,
                focalMm / physical.height * pixelArray.height,
                preCorrection.exactCenterX(),
                preCorrection.exactCenterY(),
                0f,
            )
        } else null
        val factoryValid = factory != null && factory.size >= 5 &&
            factory[0].isFinite() && factory[1].isFinite() && factory[0] > 0f && factory[1] > 0f
        val principalMarginX = preCorrection.width() * 0.1f
        val principalMarginY = preCorrection.height() * 0.1f
        val factoryPrincipalPlausible = if (factoryValid) {
            val values = factory!!
            !(abs(values[2]) < 1f && abs(values[3]) < 1f) &&
                values[2] in (preCorrection.left - principalMarginX)..(preCorrection.right + principalMarginX) &&
                values[3] in (preCorrection.top - principalMarginY)..(preCorrection.bottom + principalMarginY)
        } else false
        val factoryFocalPlausible = if (factoryValid) {
            val values = factory!!
            derived == null || (
                values[0] / derived[0] in 0.5f..2.0f &&
                    values[1] / derived[1] in 0.5f..2.0f
                )
        } else false
        val factoryPlausible = factoryValid && factoryPrincipalPlausible && factoryFocalPlausible
        val focalSource = if (factoryFocalPlausible) factory!! else derived
            ?: error("Neither factory calibration nor physical-sensor focal derivation is available")
        val principalSource = if (factoryPrincipalPlausible) factory!! else derived
        val sensor = floatArrayOf(
            focalSource[0], focalSource[1],
            principalSource?.get(2) ?: preCorrection.exactCenterX(),
            principalSource?.get(3) ?: preCorrection.exactCenterY(),
            focalSource.getOrElse(4) { 0f },
        )
        val source = when {
            factoryPlausible -> "camera2_factory_calibration"
            factoryFocalPlausible -> "camera2_factory_focal_derived_principal"
            else -> "camera2_focal_sensor_derived"
        }
        val crop = captureCrop ?: active
        val fittedCrop = fitCropToAspect(crop, streamWidth, streamHeight)
        val scaleX = streamWidth / fittedCrop[2]
        val scaleY = streamHeight / fittedCrop[3]
        val streamFx = sensor[0] * scaleX
        val streamFy = sensor[1] * scaleY
        val streamCx = (sensor[2] - fittedCrop[0]) * scaleX
        val streamCy = (sensor[3] - fittedCrop[1]) * scaleY
        val softwareCrop = if (activeHardwareJpeg) {
            PixelCrop(0, 0, streamWidth, streamHeight)
        } else {
            activeSoftwareCrop.pixels(streamWidth, streamHeight)
        }
        val outputCx = streamCx - softwareCrop.x
        val outputCy = streamCy - softwareCrop.y
        val distortion = chars.get(CameraCharacteristics.LENS_DISTORTION)
        val hardwareLevel = chars.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
        val sensorOrientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0

        return JSONObject().apply {
            put("schema", "eyetracing-camera2-intrinsics-v1")
            put("cameraId", cameraId)
            put("lensFacing", lensFacingName(chars.get(CameraCharacteristics.LENS_FACING)))
            put("frameRotation", activeFrameRotation)
            put("source", source)
            put("factoryCalibrationPresent", factoryValid)
            put("factoryCalibrationPlausible", factoryPlausible)
            put("factoryPrincipalPlausible", factoryPrincipalPlausible)
            put("factoryFocalPlausible", factoryFocalPlausible)
            put("hardwareLevel", hardwareLevelName(hardwareLevel))
            put("sensorOrientation", sensorOrientation)
            put("sensorTimestampSource", timestampSourceName(
                chars.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)
            ))
            put("timestampNs", SystemClock.elapsedRealtimeNanos())
            put("activeArray", rectJson(active))
            put("preCorrectionActiveArray", rectJson(preCorrection))
            put("captureCropRegion", rectJson(crop))
            put("effectiveStreamCrop", JSONObject().apply {
                put("left", fittedCrop[0].toDouble()); put("top", fittedCrop[1].toDouble())
                put("width", fittedCrop[2].toDouble()); put("height", fittedCrop[3].toDouble())
            })
            put("softwareCrop", JSONObject().apply {
                put("left", softwareCrop.x); put("top", softwareCrop.y)
                put("width", softwareCrop.width); put("height", softwareCrop.height)
                put("right", streamWidth - softwareCrop.x - softwareCrop.width)
                put("bottom", streamHeight - softwareCrop.y - softwareCrop.height)
                put("requestedPercent", JSONObject().apply {
                    put("left", activeSoftwareCrop.left); put("right", activeSoftwareCrop.right)
                    put("top", activeSoftwareCrop.top); put("bottom", activeSoftwareCrop.bottom)
                })
            })
            put("factoryIntrinsic", floatArrayJson(factory))
            put("derivedIntrinsic", floatArrayJson(derived))
            put("selectedSensorIntrinsic", floatArrayJson(sensor))
            put("focalLengthsMm", floatArrayJson(chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)))
            put("sensorPhysicalSizeMm", sizeFJson(physical))
            put("pixelArraySize", sizeJson(pixelArray))
            put("distortion", floatArrayJson(distortion))
            put("streamIntrinsics", JSONObject().apply {
                put("width", softwareCrop.width); put("height", softwareCrop.height)
                put("fx", streamFx.toDouble()); put("fy", streamFy.toDouble())
                put("cx", outputCx.toDouble()); put("cy", outputCy.toDouble())
                put("skew", (sensor.getOrElse(4) { 0f } * scaleX).toDouble())
            })
        }
    }

    private fun fitCropToAspect(crop: Rect, width: Int, height: Int): FloatArray {
        var left = crop.left.toFloat()
        var top = crop.top.toFloat()
        var cropWidth = crop.width().toFloat()
        var cropHeight = crop.height().toFloat()
        val outputAspect = width.toFloat() / height.toFloat()
        val cropAspect = cropWidth / cropHeight
        if (cropAspect > outputAspect) {
            val fittedWidth = cropHeight * outputAspect
            left += (cropWidth - fittedWidth) * 0.5f
            cropWidth = fittedWidth
        } else if (cropAspect < outputAspect) {
            val fittedHeight = cropWidth / outputAspect
            top += (cropHeight - fittedHeight) * 0.5f
            cropHeight = fittedHeight
        }
        return floatArrayOf(left, top, cropWidth, cropHeight)
    }

    private fun rectJson(rect: Rect): JSONObject = JSONObject().apply {
        put("left", rect.left); put("top", rect.top)
        put("right", rect.right); put("bottom", rect.bottom)
        put("width", rect.width()); put("height", rect.height())
    }

    private fun sizeJson(size: Size?): Any = size?.let {
        JSONObject().apply { put("width", it.width); put("height", it.height) }
    } ?: JSONObject.NULL

    private fun sizeFJson(size: SizeF?): Any = size?.let {
        JSONObject().apply { put("width", it.width.toDouble()); put("height", it.height.toDouble()) }
    } ?: JSONObject.NULL

    private fun floatArrayJson(values: FloatArray?): Any = values?.let {
        JSONArray().apply { it.forEach { value -> put(value.toDouble()) } }
    } ?: JSONObject.NULL

    private fun hardwareLevelName(level: Int?): String = when (level) {
        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY -> "LEGACY"
        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> "LIMITED"
        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> "FULL"
        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_3 -> "LEVEL_3"
        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_EXTERNAL -> "EXTERNAL"
        else -> "UNKNOWN"
    }

    private fun timestampSourceName(source: Int?): String = when (source) {
        CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME -> "REALTIME"
        CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_UNKNOWN -> "UNKNOWN"
        else -> "UNREPORTED"
    }

    private fun showCameraCaps() {
        runCatching {
            val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val lensFacing = if (cameraFacingInput.selectedItemPosition == 1) {
                CameraCharacteristics.LENS_FACING_BACK
            } else {
                CameraCharacteristics.LENS_FACING_FRONT
            }
            val cameraId = findCamera(manager, lensFacing) ?: error("No ${lensFacingName(lensFacing)} camera found.")
            val chars = manager.getCameraCharacteristics(cameraId)
            val ranges = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
                ?.sortedWith(compareBy<Range<Int>>({ it.upper }, { it.lower }))
                ?.joinToString(", ") { "${it.lower}..${it.upper}" }
                ?: "-"
            val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            val sizes = map?.getOutputSizes(ImageFormat.YUV_420_888)
                ?.sortedWith(compareBy<Size>({ it.width * it.height }, { it.width }))
                ?.joinToString(", ") { "${it.width}x${it.height}" }
                ?: "-"
            val jpegSizes = map?.getOutputSizes(ImageFormat.JPEG)
                ?.sortedWith(compareBy<Size>({ it.width * it.height }, { it.width }))
                ?.joinToString(", ") { "${it.width}x${it.height}" }
                ?: "-"
            val timestampSource = timestampSourceName(
                chars.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)
            )
            setStatus(
                "${lensFacingName(lensFacing)} camera=$cameraId\n" +
                    "timestamp source: $timestampSource\n" +
                    "AE FPS ranges: $ranges\nYUV sizes: $sizes\nJPEG sizes: $jpegSizes"
            )
        }.onFailure { error ->
            setStatus("Camera caps failed: ${error.message}")
        }
    }

    private fun handleImage(image: Image) {
        try {
            if (!shouldTransmit(image.timestamp)) return
            val callbackNs = SystemClock.elapsedRealtimeNanos()
            if (image.timestamp > 0L && callbackNs >= image.timestamp) {
                cameraAgeWindow.add((callbackNs - image.timestamp) / 1_000_000.0)
            }
            val conversionStartedNs = SystemClock.elapsedRealtimeNanos()
            val crop = activeSoftwareCrop.pixels(image.width, image.height)
            val outputSize = crop.width * crop.height * 3 / 2
            val output = obtainFrameBuffer(outputSize)
            val nv21 = cropYuv420ToNv21(image, crop, output)
            val conversionFinishedNs = SystemClock.elapsedRealtimeNanos()
            activeOutputWidth = nv21.width
            activeOutputHeight = nv21.height
            conversionTimeNs += conversionFinishedNs - conversionStartedNs
            convertedFrameCount += 1
            updateStats()
            publishLatestFrame(
                TransportFrame(
                    bytes = nv21.bytes,
                    width = nv21.width,
                    height = nv21.height,
                    format = UdpYuvSender.FORMAT_NV21,
                    sensorTimeNs = image.timestamp,
                )
            )
        } catch (error: Exception) {
            dropCount += 1
            if (dropCount % 30L == 1L) setStatus("Frame send failed: ${error.message}")
        } finally {
            image.close()
        }
    }

    private fun shouldTransmit(sensorTimeNs: Long): Boolean {
        val minimumIntervalNs = 1_000_000_000L / activeTargetFps.coerceAtLeast(1)
        if (nextSendSensorTimeNs == 0L) nextSendSensorTimeNs = sensorTimeNs
        if (sensorTimeNs + 1_000_000L < nextSendSensorTimeNs) {
            throttledFrameCount += 1
            updateStats()
            return false
        }
        nextSendSensorTimeNs += minimumIntervalNs
        if (nextSendSensorTimeNs <= sensorTimeNs) {
            nextSendSensorTimeNs = sensorTimeNs + minimumIntervalNs
        }
        return true
    }

    private fun handleHardwareJpeg(image: Image) {
        try {
            if (!shouldTransmit(image.timestamp)) return
            val callbackNs = SystemClock.elapsedRealtimeNanos()
            if (image.timestamp > 0L && callbackNs >= image.timestamp) {
                cameraAgeWindow.add((callbackNs - image.timestamp) / 1_000_000.0)
            }
            val conversionStartedNs = SystemClock.elapsedRealtimeNanos()
            val source = image.planes.firstOrNull()?.buffer?.duplicate()
                ?: error("Hardware JPEG image has no data plane")
            val payload = ByteArray(source.remaining())
            source.get(payload)
            val conversionFinishedNs = SystemClock.elapsedRealtimeNanos()
            activeOutputWidth = image.width
            activeOutputHeight = image.height
            conversionTimeNs += conversionFinishedNs - conversionStartedNs
            convertedFrameCount += 1
            updateStats()
            publishLatestFrame(
                TransportFrame(
                    bytes = payload,
                    width = image.width,
                    height = image.height,
                    format = UdpYuvSender.FORMAT_JPEG,
                    sensorTimeNs = image.timestamp,
                )
            )
        } catch (error: Exception) {
            dropCount += 1
            if (dropCount % 30L == 1L) setStatus("Hardware JPEG frame failed: ${error.message}")
        } finally {
            image.close()
        }
    }

    private fun obtainFrameBuffer(size: Int): ByteArray {
        synchronized(frameQueueLock) {
            val iterator = frameBufferPool.iterator()
            while (iterator.hasNext()) {
                val candidate = iterator.next()
                if (candidate.size == size) {
                    iterator.remove()
                    return candidate
                }
            }
        }
        return ByteArray(size)
    }

    private fun recycleFrameBuffer(buffer: ByteArray) {
        synchronized(frameQueueLock) {
            if (frameBufferPool.size < 3) frameBufferPool.addLast(buffer)
        }
    }

    private fun publishLatestFrame(frame: TransportFrame) {
        var schedule = false
        synchronized(frameQueueLock) {
            pendingFrame?.let {
                recycleTransportFrame(it)
                latestOnlyDropCount += 1
            }
            pendingFrame = frame
            if (!transportWorkScheduled) {
                transportWorkScheduled = true
                schedule = true
            }
        }
        if (schedule) {
            try {
                val posted = transportHandler?.post { processPendingFrames() } ?: false
                if (!posted) error("transport thread is not accepting frames")
            } catch (error: Exception) {
                synchronized(frameQueueLock) {
                    pendingFrame?.let { recycleTransportFrame(it) }
                    pendingFrame = null
                    transportWorkScheduled = false
                }
                throw error
            }
        }
    }

    private fun processPendingFrames() {
        while (true) {
            val frame = synchronized(frameQueueLock) {
                val next = pendingFrame
                pendingFrame = null
                if (next == null) transportWorkScheduled = false
                next
            } ?: return
            try {
                if (running) sendFrame(frame)
            } catch (error: Exception) {
                dropCount += 1
                if (dropCount % 30L == 1L) setStatus("Frame send failed: ${error.message}")
            } finally {
                recycleTransportFrame(frame)
            }
        }
    }

    private fun recycleTransportFrame(frame: TransportFrame) {
        if (frame.format == UdpYuvSender.FORMAT_NV21) recycleFrameBuffer(frame.bytes)
    }

    private fun sendFrame(frame: TransportFrame) {
        val encodeStartedNs = SystemClock.elapsedRealtimeNanos()
        val format: Byte
        val payload: ByteArray? = if (frame.format == UdpYuvSender.FORMAT_JPEG) {
            format = UdpYuvSender.FORMAT_JPEG
            frame.bytes
        } else if (activeJpegQuality > 0) {
            format = UdpYuvSender.FORMAT_JPEG
            jpegOutput.reset()
            val compressed = YuvImage(frame.bytes, ImageFormat.NV21, frame.width, frame.height, null)
                .compressToJpeg(Rect(0, 0, frame.width, frame.height), activeJpegQuality, jpegOutput)
            if (!compressed) error("JPEG compression returned false")
            null
        } else {
            format = UdpYuvSender.FORMAT_NV21
            frame.bytes
        }
        val encodeFinishedNs = SystemClock.elapsedRealtimeNanos()
        val sendStartedNs = encodeFinishedNs
        val sendPayload = if (payload != null) payload else jpegOutput.backingBuffer
        val sendLength = if (payload != null) payload.size else jpegOutput.size()
        val sendStats = sender?.sendFrame(
            sendPayload,
            sendLength,
            frame.width,
            frame.height,
            frame.sensorTimeNs,
            format,
        ) ?: FrameSendStats(0, 0)
        val sendFinishedNs = SystemClock.elapsedRealtimeNanos()
        frameCount += 1
        byteCount += sendStats.bytes.toLong()
        chunkCount += sendStats.chunks.toLong()
        encodeTimeNs += encodeFinishedNs - encodeStartedNs
        sendTimeNs += sendFinishedNs - sendStartedNs
        updateStats()
    }

    @Synchronized
    private fun updateStats() {
        val now = SystemClock.elapsedRealtime()
        if (now - lastStatsAtMs < 1000L) return
        val elapsed = max(1L, now - lastStatsAtMs) / 1000.0
        val fps = (frameCount - lastStatsFrameCount) / elapsed
        val mbps = (byteCount - lastStatsByteCount) * 8.0 / elapsed / 1_000_000.0
        val sampleFrames = max(1L, frameCount - lastStatsFrameCount)
        val convertedFrames = max(1L, convertedFrameCount - lastStatsConvertedFrameCount)
        val payloadKb = (byteCount - lastStatsByteCount).toDouble() / sampleFrames / 1024.0
        val conversionMs = (conversionTimeNs - lastStatsConversionTimeNs).toDouble() / convertedFrames / 1_000_000.0
        val encodeMs = (encodeTimeNs - lastStatsEncodeTimeNs).toDouble() / sampleFrames / 1_000_000.0
        val sendMs = (sendTimeNs - lastStatsSendTimeNs).toDouble() / sampleFrames / 1_000_000.0
        val cameraAge = cameraAgeWindow.drain()
        val exposure = exposureWindow.drain()
        val frameDuration = frameDurationWindow.drain()
        val rollingShutter = rollingShutterWindow.drain()
        val pipelineDepth = pipelineDepthWindow.drain()
        val chunksPerFrame = (chunkCount - lastStatsChunkCount).toDouble() / sampleFrames
        val maxPipelineDepth = activeCameraCharacteristics
            ?.get(CameraCharacteristics.REQUEST_PIPELINE_MAX_DEPTH)
            ?.toInt() ?: 0
        val throttled = throttledFrameCount - lastStatsThrottledFrameCount
        lastStatsAtMs = now
        lastStatsFrameCount = frameCount
        lastStatsByteCount = byteCount
        lastStatsConversionTimeNs = conversionTimeNs
        lastStatsEncodeTimeNs = encodeTimeNs
        lastStatsSendTimeNs = sendTimeNs
        lastStatsChunkCount = chunkCount
        lastStatsThrottledFrameCount = throttledFrameCount
        lastStatsConvertedFrameCount = convertedFrameCount
        runOnUiThread {
            val format = when {
                activeHardwareJpeg -> "HW JPEG Q$activeJpegQuality"
                activeJpegQuality > 0 -> "JPEG Q$activeJpegQuality"
                else -> "NV21"
            }
            val cameraStage = if (activeHardwareJpeg) "sensor->JPEG" else "sensor->YUV"
            val cameraAgeText = cameraAge?.let {
                "${"%.2f".format(it.mean)}+/-${"%.2f".format(it.standardDeviation)}ms " +
                    "[${"%.2f".format(it.minimum)}..${"%.2f".format(it.maximum)}] n=${it.count}"
            } ?: "unavailable"
            val exposureText = exposure?.let {
                "${"%.2f".format(it.mean)} [${"%.2f".format(it.minimum)}..${"%.2f".format(it.maximum)}]"
            } ?: "-"
            val frameText = frameDuration?.let {
                "${"%.2f".format(it.mean)} [${"%.2f".format(it.minimum)}..${"%.2f".format(it.maximum)}]"
            } ?: "-"
            val rollingText = rollingShutter?.let {
                "${"%.2f".format(it.mean)} [${"%.2f".format(it.minimum)}..${"%.2f".format(it.maximum)}]"
            } ?: "-"
            val depthText = pipelineDepth?.let {
                "${it.minimum.roundToInt()}..${it.maximum.roundToInt()}/$maxPipelineDepth n=${it.count}"
            } ?: "-/$maxPipelineDepth"
            statsText.text =
                "$format ${activeOutputWidth}x$activeOutputHeight sent=$frameCount " +
                    "fps=${"%.1f".format(fps)} mbps=${"%.1f".format(mbps)} " +
                    "KB/frame=${"%.1f".format(payloadKb)} drops=$dropCount\n" +
                    "copy=${"%.2f".format(conversionMs)}ms appEncode=${"%.2f".format(encodeMs)}ms " +
                    "send=${"%.2f".format(sendMs)}ms " +
                    "$cameraStage=$cameraAgeText chunks=${"%.1f".format(chunksPerFrame)} " +
                    "throttled=$throttled latestDrops=$latestOnlyDropCount yuv=$yuvConversionPath\n" +
                    "capture exp=$exposureText ms frame=$frameText ms\n" +
                    "rolling=$rollingText ms pipelineDepth=$depthText\n" +
                    "${lensFacingName(activeLensFacing)} camera rotation=$activeFrameRotation " +
                    "requested=${activeStreamWidth}x$activeStreamHeight " +
                    "processing=${if (activeLowLatencyProcessing) "low-latency" else "default"} " +
                    "timestamp=${timestampSourceName(activeTimestampSource)} UDP=$activeChunkPayloadSize"
        }
    }

    private fun stopStreaming() {
        running = false
        runCatching { captureSession?.stopRepeating() }
        runCatching { captureSession?.close() }
        runCatching { cameraDevice?.close() }
        runCatching { imageReader?.close() }
        captureSession = null
        cameraDevice = null
        imageReader = null
        sender?.close()
        sender = null
        synchronized(frameQueueLock) {
            pendingFrame?.let { recycleTransportFrame(it) }
            pendingFrame = null
            transportWorkScheduled = false
        }
        transportThread?.quitSafely()
        transportThread = null
        transportHandler = null
        cameraThread?.quitSafely()
        cameraThread = null
        cameraHandler = null
        setStatus("Stopped.")
    }

    private fun setStatus(text: String) {
        runOnUiThread { statusText.text = text }
    }

    private fun cropYuv420ToNv21(image: Image, crop: PixelCrop, out: ByteArray): Nv21Frame {
        val sourceWidth = image.width
        val sourceHeight = image.height
        val width = crop.width
        val height = crop.height
        require(sourceWidth % 2 == 0 && sourceHeight % 2 == 0) {
            "YUV dimensions must be even, got ${sourceWidth}x$sourceHeight"
        }
        val outputSize = width * height * 3 / 2
        require(out.size == outputSize) {
            "NV21 output buffer is ${out.size} bytes, expected $outputSize"
        }
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer.duplicate()
        val yBase = yBuffer.position()
        if (yPlane.pixelStride == 1) {
            for (row in 0 until height) {
                val source = yBase + (crop.y + row) * yPlane.rowStride + crop.x
                require(source + width <= yBuffer.limit()) { "Y plane row $row is truncated" }
                yBuffer.position(source)
                yBuffer.get(out, row * width, width)
            }
        } else {
            for (row in 0 until height) {
                val rowStart = yBase + (crop.y + row) * yPlane.rowStride + crop.x * yPlane.pixelStride
                for (col in 0 until width) {
                    out[row * width + col] = yBuffer.get(rowStart + col * yPlane.pixelStride)
                }
            }
        }

        val chromaHeight = height / 2
        val chromaWidth = width / 2
        val sourceChromaHeight = sourceHeight / 2
        val sourceChromaWidth = sourceWidth / 2
        val chromaX = crop.x / 2
        val chromaY = crop.y / 2
        val fastPath = useInterleavedVuFastPath ?: isInterleavedVu(
            uPlane, vPlane, sourceChromaWidth, sourceChromaHeight
        ).also { useInterleavedVuFastPath = it }
        if (fastPath && copyInterleavedVu(
                vPlane, uPlane, out, width * height,
                chromaX, chromaY, chromaWidth, chromaHeight,
            )
        ) {
            yuvConversionPath = "interleaved-vu-fast-crop"
            return Nv21Frame(out, width, height, crop)
        }

        val uBuffer = uPlane.buffer.duplicate()
        val vBuffer = vPlane.buffer.duplicate()
        val uBase = uBuffer.position()
        val vBase = vBuffer.position()
        var outPos = width * height
        if (uPlane.pixelStride == 1 && vPlane.pixelStride == 1) {
            if (planarURow.size != chromaWidth) planarURow = ByteArray(chromaWidth)
            if (planarVRow.size != chromaWidth) planarVRow = ByteArray(chromaWidth)
            for (row in 0 until chromaHeight) {
                val uRowStart = uBase + (chromaY + row) * uPlane.rowStride + chromaX
                val vRowStart = vBase + (chromaY + row) * vPlane.rowStride + chromaX
                require(uRowStart + chromaWidth <= uBuffer.limit()) { "U plane row $row is truncated" }
                require(vRowStart + chromaWidth <= vBuffer.limit()) { "V plane row $row is truncated" }
                uBuffer.position(uRowStart)
                vBuffer.position(vRowStart)
                uBuffer.get(planarURow, 0, chromaWidth)
                vBuffer.get(planarVRow, 0, chromaWidth)
                for (col in 0 until chromaWidth) {
                    out[outPos++] = planarVRow[col]
                    out[outPos++] = planarURow[col]
                }
            }
            yuvConversionPath = "planar-row-fast-crop"
            return Nv21Frame(out, width, height, crop)
        }

        for (row in 0 until chromaHeight) {
            val uRowStart = uBase + (chromaY + row) * uPlane.rowStride + chromaX * uPlane.pixelStride
            val vRowStart = vBase + (chromaY + row) * vPlane.rowStride + chromaX * vPlane.pixelStride
            for (col in 0 until chromaWidth) {
                out[outPos++] = vBuffer.get(vRowStart + col * vPlane.pixelStride)
                out[outPos++] = uBuffer.get(uRowStart + col * uPlane.pixelStride)
            }
        }
        yuvConversionPath = "generic-slow-crop"
        return Nv21Frame(out, width, height, crop)
    }

    private fun isInterleavedVu(
        uPlane: Image.Plane,
        vPlane: Image.Plane,
        chromaWidth: Int,
        chromaHeight: Int,
    ): Boolean {
        val rowBytes = chromaWidth * 2
        if (
            uPlane.pixelStride != 2 || vPlane.pixelStride != 2 ||
            uPlane.rowStride < rowBytes - 1 || vPlane.rowStride < rowBytes ||
            chromaWidth < 2 || chromaHeight < 1
        ) {
            return false
        }
        val u = uPlane.buffer.duplicate()
        val v = vPlane.buffer.duplicate()
        val uBase = u.position()
        val vBase = v.position()
        val rows = intArrayOf(0, chromaHeight / 2, chromaHeight - 1).distinct()
        val columnStep = max(1, (chromaWidth - 1) / 16)
        var compared = 0
        for (row in rows) {
            var col = 0
            while (col < chromaWidth - 1) {
                val uIndex = uBase + row * uPlane.rowStride + col * uPlane.pixelStride
                val vGapIndex = vBase + row * vPlane.rowStride + col * vPlane.pixelStride + 1
                if (uIndex >= u.limit() || vGapIndex >= v.limit() || u.get(uIndex) != v.get(vGapIndex)) {
                    return false
                }
                compared += 1
                col += columnStep
            }
        }
        return compared >= min(16, chromaWidth - 1)
    }

    private fun copyInterleavedVu(
        vPlane: Image.Plane,
        uPlane: Image.Plane,
        out: ByteArray,
        outputOffset: Int,
        chromaX: Int,
        chromaY: Int,
        chromaWidth: Int,
        chromaHeight: Int,
    ): Boolean {
        val v = vPlane.buffer.duplicate()
        val u = uPlane.buffer.duplicate()
        val vBase = v.position()
        val uBase = u.position()
        val rowBytes = chromaWidth * 2
        for (row in 0 until chromaHeight) {
            val sourceRow = chromaY + row
            val vRowStart = vBase + sourceRow * vPlane.rowStride + chromaX * vPlane.pixelStride
            val available = v.limit() - vRowStart
            if (available < rowBytes - 1) return false
            val copyLength = min(rowBytes, available)
            v.position(vRowStart)
            v.get(out, outputOffset + row * rowBytes, copyLength)
            if (copyLength < rowBytes) {
                val finalU = uBase + sourceRow * uPlane.rowStride +
                    (chromaX + chromaWidth - 1) * uPlane.pixelStride
                if (finalU >= u.limit()) return false
                out[outputOffset + (row + 1) * rowBytes - 1] = u.get(finalU)
            }
        }
        return true
    }
}

private class UdpYuvSender(host: String, port: Int, private val chunkPayloadSize: Int) : AutoCloseable {
    private val socket = DatagramSocket().apply {
        // Keep back-pressure close to the camera. A multi-megabyte UDP queue
        // makes send() look fast while displaying frames hundreds of ms late.
        sendBufferSize = 128 * 1024
        connect(InetAddress.getByName(host), port)
    }
    private var frameSeq = 0
    private val packetBytes = ByteArray(HEADER_SIZE + chunkPayloadSize)
    private val buffer = ByteBuffer.wrap(packetBytes).order(ByteOrder.LITTLE_ENDIAN)
    private val packet = DatagramPacket(packetBytes, packetBytes.size)

    fun sendFrame(
        payload: ByteArray,
        payloadLength: Int,
        width: Int,
        height: Int,
        sensorTimeNs: Long,
        format: Byte,
    ): FrameSendStats {
        require(payloadLength in 0..payload.size)
        val seq = frameSeq++
        val chunkCount = ceil(payloadLength / chunkPayloadSize.toDouble()).toInt()
        val frameSendTimeNs = SystemClock.elapsedRealtimeNanos()
        var offset = 0
        var sent = 0
        for (chunkIndex in 0 until chunkCount) {
            val payloadSize = min(chunkPayloadSize, payloadLength - offset)
            buffer.clear()
            buffer.putInt(MAGIC)
            buffer.putShort(VERSION.toShort())
            buffer.putShort(HEADER_SIZE.toShort())
            buffer.putInt(seq)
            buffer.putShort(chunkIndex.toShort())
            buffer.putShort(chunkCount.toShort())
            buffer.putShort(width.toShort())
            buffer.putShort(height.toShort())
            buffer.put(format)
            buffer.put(0)
            buffer.putLong(sensorTimeNs)
            buffer.putLong(frameSendTimeNs)
            buffer.putInt(payloadSize)
            System.arraycopy(payload, offset, packetBytes, HEADER_SIZE, payloadSize)
            packet.length = HEADER_SIZE + payloadSize
            socket.send(packet)
            offset += payloadSize
            sent += payloadSize
        }
        return FrameSendStats(sent, chunkCount)
    }

    fun sendIntrinsics(message: JSONObject) {
        val payload = message.toString().toByteArray(Charsets.UTF_8)
        val packetBytes = ByteArray(INTRINSICS_HEADER_SIZE + payload.size)
        val buffer = ByteBuffer.wrap(packetBytes).order(ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(INTRINSICS_MAGIC)
        buffer.putShort(INTRINSICS_VERSION.toShort())
        buffer.putShort(INTRINSICS_HEADER_SIZE.toShort())
        buffer.putInt(payload.size)
        buffer.put(payload)
        socket.send(DatagramPacket(packetBytes, packetBytes.size))
    }

    override fun close() {
        socket.close()
    }

    companion object {
        private const val MAGIC = 0x56555945
        private const val VERSION = 1
        private const val HEADER_SIZE = 42
        const val FORMAT_NV21: Byte = 1
        const val FORMAT_JPEG: Byte = 2
        private const val INTRINSICS_MAGIC = 0x49435945
        private const val INTRINSICS_VERSION = 1
        private const val INTRINSICS_HEADER_SIZE = 12
    }
}

private const val DISCOVERY_MAGIC = "EYETRACING_DISCOVERY_V1"
private const val DISCOVERY_VERSION = 1
private const val DISCOVERY_PORT = 5006
