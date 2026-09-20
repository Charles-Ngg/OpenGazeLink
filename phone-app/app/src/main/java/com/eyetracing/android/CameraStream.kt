package com.eyetracing.android

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraConstrainedHighSpeedCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.ImageReader
import android.net.wifi.WifiManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.SystemClock
import android.util.Range
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.ArrayDeque
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

internal enum class StreamFailure { CONNECTION, CAMERA, COMBINATION, INTRINSICS }
internal data class StreamRates(
    val captureFps: Double, val sentFps: Double, val mbps: Double,
    val encodingAgeMs: Double?, val exposureMs: Double?,
)

/** One capture session owns all resources, queues and callbacks across restarts. */
internal class CameraStream(
    context: Context,
    private val option: CaptureOption,
    private val host: String,
    private val port: Int,
    private val rotation: Int,
    private val cropPercent: CropPercent,
    private val jpegQuality: Int,
    private val onReady: () -> Unit,
    private val onRates: (StreamRates) -> Unit,
    private val onFailure: (StreamFailure, String) -> Unit,
) : AutoCloseable {
    private val softwareCrop = (if (option.mode == SessionMode.CAMERA) cropPercent else CropPercent())
        .pixels(option.width, option.height, rotation)
    init { require(jpegQuality in 1..100) }
    private val app = context.applicationContext
    private val manager = app.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    private val main = Handler(Looper.getMainLooper())
    private val cameraThread = HandlerThread("phone-camera").apply { start() }
    private val cameraHandler = Handler(cameraThread.looper)
    private val transportThread = HandlerThread("phone-jpeg-send").apply { start() }
    private val transport = Handler(transportThread.looper)
    private val closed = AtomicBoolean(false)
    private val resourceLock = Any()
    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var encoder: AvcEncoder? = null
    private var udp: UdpYuvSender? = null
    private var avc: AvcTcpSender? = null
    private var wifiLock: WifiManager.WifiLock? = null
    private var intrinsics: JSONObject? = null
    private var intrinsicsSentAt = 0L
    private val captured = AtomicLong()
    private val sent = AtomicLong()
    private val sentBytes = AtomicLong()
    private var lastCaptured = 0L
    private var lastSent = 0L
    private var lastBytes = 0L
    private val timingLock = Any()
    private var encodedAgeTotal = 0.0
    private var encodedAgeCount = 0
    private var exposureTotal = 0.0
    private var exposureCount = 0
    private var realtimeSensorClock = false
    private var lastStatsTime = SystemClock.elapsedRealtime()
    private var lastSensorNs = 0L
    private val converter = FullFrameYuv()
    private data class Frame(val bytes: ByteArray, val sensorNs: Long)
    private val queueLock = Any()
    private val pool = ArrayDeque<ByteArray>()
    private var pending: Frame? = null
    private var sending = false
    private class JpegBuffer : ByteArrayOutputStream(512 * 1024) { val bytes: ByteArray get() = buf }

    fun start() {
        acquireWifiLock()
        main.post(statsTick)
        cameraHandler.post {
            if (closed.get()) return@post
            try {
                val sender = UdpYuvSender(host, port, 1400)
                synchronized(resourceLock) { if (closed.get()) sender.close() else udp = sender }
                if (option.mode == SessionMode.HIGH_SPEED && !closed.get()) {
                    val senderAvc = AvcTcpSender(host, port, option.width, option.height,
                        onSent = { bytes -> if (!closed.get()) { sent.incrementAndGet(); sentBytes.addAndGet(bytes.toLong()) } },
                        onError = { fail(StreamFailure.CONNECTION, it) })
                    synchronized(resourceLock) { if (closed.get()) senderAvc.close() else avc = senderAvc }
                }
            } catch (error: Exception) { fail(StreamFailure.CONNECTION, error); return@post }
            if (closed.get()) return@post
            try { openCamera() } catch (error: Exception) { fail(StreamFailure.CAMERA, error) }
        }
    }

    @SuppressLint("MissingPermission")
    private fun openCamera() {
        val chars = manager.getCameraCharacteristics(option.cameraId)
        realtimeSensorClock = chars.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE) ==
            CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
        try {
            intrinsics = CameraIntrinsics.build(option.cameraId, chars, option.width, option.height,
                chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE), rotation, softwareCrop)
            sendIntrinsics()
        } catch (error: Exception) { fail(StreamFailure.INTRINSICS, error); return }
        try {
            if (option.mode == SessionMode.HIGH_SPEED) {
                val output = AvcEncoder(option, cameraHandler,
                    onFrame = { sensor -> recordCapture(sensor) },
                    onPacket = { data, sensor, now, flags ->
                        if (!closed.get()) {
                            if (flags and android.media.MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0) recordEncodedAge(sensor, now)
                            avc?.offer(data, sensor, now, flags)
                        }
                    },
                    onError = { fail(StreamFailure.COMBINATION, it) })
                synchronized(resourceLock) { if (closed.get()) output.close() else encoder = output }
            } else {
                val output = ImageReader.newInstance(option.width, option.height, ImageFormat.YUV_420_888, 3)
                output.setOnImageAvailableListener({ source -> captureImage(source) }, cameraHandler)
                synchronized(resourceLock) { if (closed.get()) output.close() else reader = output }
            }
        } catch (error: Exception) { fail(StreamFailure.COMBINATION, error); return }
        if (closed.get()) return
        // Keep device callbacks on the main looper so closing an in-flight open
        // cannot lose onOpened when the worker looper is shut down.
        manager.openCamera(option.cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                synchronized(resourceLock) {
                    if (closed.get()) { camera.close(); return }
                    device = camera
                }
                cameraHandler.post {
                    if (closed.get()) return@post
                    try { configureSession(camera, chars) } catch (error: Exception) { fail(StreamFailure.COMBINATION, error) }
                }
            }
            override fun onDisconnected(camera: CameraDevice) {
                camera.close()
                fail(StreamFailure.CAMERA, IllegalStateException("Camera disconnected"))
            }
            override fun onError(camera: CameraDevice, error: Int) {
                camera.close()
                fail(StreamFailure.CAMERA, IllegalStateException("Camera error " + error))
            }
        }, main)
    }

    @Suppress("DEPRECATION")
    private fun configureSession(camera: CameraDevice, chars: CameraCharacteristics) {
        val target = encoder?.surface ?: reader?.surface ?: return
        val highSpeed = option.mode == SessionMode.HIGH_SPEED
        val builder = camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD).apply {
            addTarget(target)
            set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
            set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(option.fpsLower, option.fps))
            if (!highSpeed) CameraTuning.apply(this, chars)
        }
        val callback = object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(capture: CameraCaptureSession) {
                synchronized(resourceLock) {
                    if (closed.get()) { capture.close(); return }
                    session = capture
                }
                cameraHandler.post {
                    if (closed.get()) return@post
                    try {
                        val request = builder.build()
                        val captureCallback = object : CameraCaptureSession.CaptureCallback() {
                            override fun onCaptureCompleted(s: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                                if (closed.get()) return
                                result.get(CaptureResult.SENSOR_EXPOSURE_TIME)?.takeIf { it > 0 }?.let {
                                    synchronized(timingLock) { exposureTotal += it / 1_000_000.0; exposureCount++ }
                                }
                                val now = SystemClock.elapsedRealtime()
                                if (now - intrinsicsSentAt > 2000) {
                                    try {
                                        intrinsics = CameraIntrinsics.build(option.cameraId, chars, option.width, option.height,
                                            result.get(CaptureResult.SCALER_CROP_REGION), rotation, softwareCrop)
                                        sendIntrinsics()
                                    } catch (error: Exception) { fail(StreamFailure.INTRINSICS, error) }
                                }
                            }
                        }
                        if (highSpeed) {
                            val constrained = capture as? CameraConstrainedHighSpeedCaptureSession
                                ?: error("Camera did not create a high-speed session")
                            constrained.setRepeatingBurst(constrained.createHighSpeedRequestList(request), captureCallback, cameraHandler)
                        } else capture.setRepeatingRequest(request, captureCallback, cameraHandler)
                        main.post { if (!closed.get()) onReady() }
                    } catch (error: Exception) { fail(StreamFailure.COMBINATION, error) }
                }
            }
            override fun onConfigureFailed(capture: CameraCaptureSession) {
                capture.close()
                fail(StreamFailure.COMBINATION, IllegalStateException("Camera rejected this size/rate combination"))
            }
        }
        if (Build.VERSION.SDK_INT >= 28) {
            val output = OutputConfiguration(target)
            if (Build.VERSION.SDK_INT >= 33) output.setTimestampBase(OutputConfiguration.TIMESTAMP_BASE_SENSOR)
            camera.createCaptureSession(SessionConfiguration(
                if (highSpeed) SessionConfiguration.SESSION_HIGH_SPEED else SessionConfiguration.SESSION_REGULAR,
                listOf(output), java.util.concurrent.Executor { main.post(it) }, callback))
        } else if (highSpeed) {
            camera.createConstrainedHighSpeedCaptureSession(listOf(target), callback, main)
        } else camera.createCaptureSession(listOf(target), callback, main)
    }

    private fun sendIntrinsics() {
        if (closed.get()) return
        intrinsics?.let { udp?.sendIntrinsics(it) }
        intrinsicsSentAt = SystemClock.elapsedRealtime()
    }

    private fun recordCapture(sensorNs: Long) {
        if (closed.get() || sensorNs <= lastSensorNs) return
        lastSensorNs = sensorNs
        captured.incrementAndGet()
    }

    private fun recordEncodedAge(sensorNs: Long, readyNs: Long) {
        if (!realtimeSensorClock || sensorNs <= 0 || readyNs < sensorNs) return
        // Older encoder surfaces cannot explicitly select the sensor timebase.
        if (option.mode == SessionMode.HIGH_SPEED && Build.VERSION.SDK_INT < 33) return
        synchronized(timingLock) {
            encodedAgeTotal += (readyNs - sensorNs) / 1_000_000.0
            encodedAgeCount++
        }
    }

    private fun captureImage(source: ImageReader) {
        val image = runCatching { source.acquireLatestImage() }.getOrNull() ?: return
        var bytes: ByteArray? = null
        try {
            if (closed.get()) return
            check(image.width == option.width && image.height == option.height) { "Camera output size changed" }
            recordCapture(image.timestamp)
            val buffer = synchronized(queueLock) { pool.pollFirst() } ?: ByteArray(softwareCrop.width * softwareCrop.height * 3 / 2)
            bytes = buffer
            converter.copy(image, buffer, softwareCrop)
            synchronized(queueLock) {
                if (closed.get()) { pool.addLast(buffer); bytes = null; return }
                pending?.let { pool.addLast(it.bytes) }
                pending = Frame(buffer, image.timestamp)
                bytes = null
                if (!sending) {
                    sending = true
                    transport.post { sendLatestFrames() }
                }
            }
        } catch (error: Exception) { fail(StreamFailure.CAMERA, error) }
        finally {
            bytes?.let { synchronized(queueLock) { pool.addLast(it) } }
            image.close()
        }
    }

    private fun sendLatestFrames() {
        val output = JpegBuffer()
        while (!closed.get()) {
            val frame = synchronized(queueLock) {
                pending.also { pending = null; if (it == null) sending = false }
            } ?: return
            try {
                output.reset()
                check(YuvImage(frame.bytes, ImageFormat.NV21, softwareCrop.width, softwareCrop.height, null)
                    .compressToJpeg(Rect(0, 0, softwareCrop.width, softwareCrop.height), jpegQuality, output)) { "JPEG encoding failed" }
                if (!closed.get()) {
                    recordEncodedAge(frame.sensorNs, SystemClock.elapsedRealtimeNanos())
                    udp?.sendFrame(output.bytes, output.size(), softwareCrop.width, softwareCrop.height, frame.sensorNs, UdpYuvSender.FORMAT_JPEG)?.let {
                        sent.incrementAndGet(); sentBytes.addAndGet(it.bytes.toLong() + it.chunks * 42L)
                    }
                }
            } catch (error: Exception) { fail(StreamFailure.CONNECTION, error); return }
            finally { synchronized(queueLock) { if (pool.size < 4) pool.addLast(frame.bytes) } }
        }
    }

    private val statsTick = object : Runnable {
        override fun run() {
            if (closed.get()) return
            val now = SystemClock.elapsedRealtime()
            val elapsed = (now - lastStatsTime).coerceAtLeast(1) / 1000.0
            val captureCount = captured.get()
            val sentCount = sent.get()
            val bytes = sentBytes.get()
            val timing = synchronized(timingLock) {
                val result = (if (encodedAgeCount > 0) encodedAgeTotal / encodedAgeCount else null) to
                    (if (exposureCount > 0) exposureTotal / exposureCount else null)
                encodedAgeTotal = 0.0; encodedAgeCount = 0; exposureTotal = 0.0; exposureCount = 0
                result
            }
            onRates(StreamRates((captureCount - lastCaptured) / elapsed, (sentCount - lastSent) / elapsed,
                (bytes - lastBytes) * 8 / elapsed / 1_000_000, timing.first, timing.second))
            lastStatsTime = now; lastCaptured = captureCount; lastSent = sentCount; lastBytes = bytes
            main.postDelayed(this, 1000)
        }
    }

    @Suppress("DEPRECATION")
    private fun acquireWifiLock() {
        runCatching {
            val wifi = app.getSystemService(Context.WIFI_SERVICE) as WifiManager
            val mode = if (Build.VERSION.SDK_INT >= 29) WifiManager.WIFI_MODE_FULL_LOW_LATENCY else WifiManager.WIFI_MODE_FULL_HIGH_PERF
            wifi.createWifiLock(mode, "OpenGazeLink:camera").also {
                it.setReferenceCounted(false); it.acquire(); wifiLock = it
            }
        } // A driver may ignore this request; it is not shown as a measured capability.
    }

    private fun fail(kind: StreamFailure, error: Throwable) {
        if (closed.get()) return
        close()
        main.post { onFailure(kind, error.message ?: error.javaClass.simpleName) }
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        main.removeCallbacks(statsTick)
        wifiLock?.let { runCatching { if (it.isHeld) it.release() } }
        wifiLock = null
        synchronized(resourceLock) {
            udp?.close(); avc?.close()
            runCatching { session?.stopRepeating() }
            runCatching { session?.close() }
            runCatching { device?.close() }
        }
        synchronized(queueLock) { pending = null; pool.clear() }
        cameraHandler.post {
            runCatching { reader?.close() }
            runCatching { encoder?.close() }
            reader = null; encoder = null
        }
        transportThread.quitSafely()
        cameraThread.quitSafely()
    }
}
