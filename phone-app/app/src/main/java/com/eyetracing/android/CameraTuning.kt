package com.eyetracing.android

import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CaptureRequest
import android.os.Build

/** One automatic low-latency configuration, no experimental UI toggles. */
internal object CameraTuning {
    fun apply(
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

}
