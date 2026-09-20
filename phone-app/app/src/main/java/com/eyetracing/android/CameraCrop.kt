package com.eyetracing.android

import kotlin.math.roundToInt

/** Insets in the displayed (clockwise-rotated, not mirrored) frame. */
internal data class CropPercent(
    val left: Double = 0.0, val right: Double = 0.0,
    val top: Double = 0.0, val bottom: Double = 0.0,
) {
    init { require(listOf(left, right, top, bottom).all { it.isFinite() && it in 0.0..45.0 }) }

    fun pixels(width: Int, height: Int, rotation: Int): PixelCrop {
        require(width >= 4 && height >= 4 && width % 2 == 0 && height % 2 == 0)
        // Inverse-map display edges instead of rotating/copying a full YUV image.
        val raw = when (rotation) {
            0 -> this
            90 -> CropPercent(top, bottom, right, left)
            180 -> CropPercent(right, left, bottom, top)
            270 -> CropPercent(bottom, top, left, right)
            else -> error("Rotation must be 0, 90, 180 or 270")
        }
        fun inset(size: Int, percent: Double) = ((size * percent / 100).roundToInt() and -2).coerceIn(0, size - 4)
        val x = inset(width, raw.left)
        val y = inset(height, raw.top)
        val r = inset(width, raw.right).coerceAtMost(width - x - 4)
        val b = inset(height, raw.bottom).coerceAtMost(height - y - 4)
        return PixelCrop(x, y, width - x - r, height - y - b)
    }
}

internal data class PixelCrop(val x: Int, val y: Int, val width: Int, val height: Int) {
    fun validateIn(frameWidth: Int, frameHeight: Int) {
        require(x >= 0 && y >= 0 && width >= 4 && height >= 4)
        require(listOf(x, y, width, height).all { it % 2 == 0 })
        require(x + width <= frameWidth && y + height <= frameHeight)
    }

    /** Even the fallback principal point belongs to the ORIGINAL stream. Never re-center after cropping. */
    fun principalPoint(streamWidth: Int, streamHeight: Int, cx: Float, cy: Float, estimated: Boolean): Pair<Float, Float> =
        (if (estimated) (streamWidth - 1) * 0.5f else cx) - x to
            (if (estimated) (streamHeight - 1) * 0.5f else cy) - y
}

internal object JpegQuality {
    val levels = listOf(50, 65, 80, 90, 95, 100)
    const val DEFAULT = 80
}
