package com.eyetracing.android

import android.graphics.Rect
import android.hardware.camera2.CameraCharacteristics
import android.os.Build
import android.os.SystemClock
import android.util.Size
import android.util.SizeF
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.abs

internal object CameraIntrinsics {
    fun build(
        cameraId: String,
        chars: CameraCharacteristics,
        streamWidth: Int,
        streamHeight: Int,
        captureCrop: Rect?,
        frameRotation: Int,
        softwareCrop: PixelCrop = PixelCrop(0, 0, streamWidth, streamHeight),
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
        // Camera2's sensor-to-stream aspect fit is physical camera geometry.
        // Invalid factory principal points fall back to the full frame center,
        // BEFORE subtracting software crop offsets. Focal lengths do not change.
        softwareCrop.validateIn(streamWidth, streamHeight)
        val estimatedPrincipal = !factoryPrincipalPlausible
        val (outputCx, outputCy) = softwareCrop.principalPoint(streamWidth, streamHeight,
            (sensor[2] - fittedCrop[0]) * scaleX, (sensor[3] - fittedCrop[1]) * scaleY, estimatedPrincipal)
        val distortion = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P)
            chars.get(CameraCharacteristics.LENS_DISTORTION) else null
        val hardwareLevel = chars.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
        val sensorOrientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0

        return JSONObject().apply {
            put("schema", "eyetracing-camera2-intrinsics-v1")
            put("cameraId", cameraId)
            put("lensFacing", lensFacingName(chars.get(CameraCharacteristics.LENS_FACING)))
            put("frameRotation", frameRotation)
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
                put("coordinateSystem", "unrotated_stream")
                put("left", softwareCrop.x); put("top", softwareCrop.y)
                put("right", streamWidth - softwareCrop.x - softwareCrop.width)
                put("bottom", streamHeight - softwareCrop.y - softwareCrop.height)
                put("width", softwareCrop.width); put("height", softwareCrop.height)
                put("sourceWidth", streamWidth); put("sourceHeight", streamHeight)
            })
            put("factoryIntrinsic", floatArrayJson(factory))
            put("derivedIntrinsic", floatArrayJson(derived))
            put("selectedSensorIntrinsic", floatArrayJson(sensor))
            put("principalPointSource", if (estimatedPrincipal) "full_stream_center_before_software_crop" else "camera2_factory_calibration")
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

    private fun lensFacingName(facing: Int?): String = when (facing) {
        CameraCharacteristics.LENS_FACING_FRONT -> "front"
        CameraCharacteristics.LENS_FACING_BACK -> "back"
        CameraCharacteristics.LENS_FACING_EXTERNAL -> "external"
        else -> "unknown"
    }
}
