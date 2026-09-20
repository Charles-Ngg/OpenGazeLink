package com.eyetracing.android

import android.graphics.ImageFormat
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import kotlin.math.floor

internal data class CameraChoice(val id: String, val facing: Int?, val options: List<CaptureOption>)

internal object CameraCatalog {
    fun read(manager: CameraManager): List<CameraChoice> = manager.cameraIdList.mapNotNull { id ->
        // One broken auxiliary camera must not hide other usable cameras.
        runCatching {
            val chars = manager.getCameraCharacteristics(id)
            val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return@runCatching null
            val options = mutableListOf<CaptureOption>()
            val highSpeed = chars.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
                ?.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_CONSTRAINED_HIGH_SPEED_VIDEO) == true
            if (highSpeed) {
                for (size in runCatching { map.highSpeedVideoSizes }.getOrNull().orEmpty()) {
                    for (range in runCatching { map.getHighSpeedVideoFpsRangesFor(size) }.getOrNull().orEmpty()) {
                        if (range.lower != range.upper || range.upper < 120) continue
                        val encoder = runCatching { AvcEncoder.findEncoder(size.width, size.height, range.upper) }.getOrNull() ?: continue
                        options += CaptureOption(id, SessionMode.HIGH_SPEED, size.width, size.height,
                            range.upper, range.lower, encoder)
                    }
                }
            }
            val ranges = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES).orEmpty()
            for (size in runCatching { map.getOutputSizes(ImageFormat.YUV_420_888) }.getOrNull().orEmpty()) {
                val minimumNs = runCatching { map.getOutputMinFrameDuration(ImageFormat.YUV_420_888, size) }.getOrNull() ?: continue
                val maximumFps = if (minimumNs > 0) floor(1_000_000_000.0 / minimumNs + .01).toInt() else 60
                for (range in ranges) {
                    if (range.upper > 60 || range.upper > maximumFps) continue
                    options += CaptureOption(id, SessionMode.CAMERA, size.width, size.height, range.upper, range.lower)
                }
            }
            CameraChoice(id, chars.get(CameraCharacteristics.LENS_FACING), CaptureOptions.ordered(options))
        }.getOrNull()
    }.sortedWith(compareBy<CameraChoice>({ if (it.facing == CameraCharacteristics.LENS_FACING_FRONT) 0 else 1 }, { it.id }))
}
