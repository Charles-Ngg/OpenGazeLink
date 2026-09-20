package com.eyetracing.android

import org.junit.Assert.*
import org.junit.Test

class CameraCropTest {
    @Test fun asymmetricDisplayInsetsMapToAllFourRotations() {
        val crop = CropPercent(10.0, 20.0, 30.0, 40.0)
        assertEquals(PixelCrop(20, 30, 140, 30), crop.pixels(200, 100, 0))
        assertEquals(PixelCrop(60, 20, 60, 70), crop.pixels(200, 100, 90))
        assertEquals(PixelCrop(40, 40, 140, 30), crop.pixels(200, 100, 180))
        assertEquals(PixelCrop(80, 10, 60, 70), crop.pixels(200, 100, 270))
    }

    @Test fun everyRetainedPixelMatchesRotateThenCrop() {
        val w = 200; val h = 100
        for (rotation in listOf(0, 90, 180, 270)) {
            val rotatedW = if (rotation % 180 == 0) w else h
            val rotatedH = if (rotation % 180 == 0) h else w
            val percent = CropPercent(10.0, 20.0, 30.0, 40.0)
            val display = percent.pixels(rotatedW, rotatedH, 0)
            val raw = percent.pixels(w, h, rotation)
            for (y in 0 until h) for (x in 0 until w) {
                val (rx, ry) = when (rotation) {
                    90 -> h - 1 - y to x
                    180 -> w - 1 - x to h - 1 - y
                    270 -> y to w - 1 - x
                    else -> x to y
                }
                assertEquals(x in raw.x until raw.x + raw.width && y in raw.y until raw.y + raw.height,
                    rx in display.x until display.x + display.width && ry in display.y until display.y + display.height)
            }
        }
    }

    @Test fun cropIsEvenBoundedAndNonemptyAtLimits() {
        for (w in listOf(4, 6, 640, 1920)) for (h in listOf(4, 10, 480, 1080))
            for (r in listOf(0, 90, 180, 270)) {
                CropPercent(45.0, 45.0, 45.0, 45.0).pixels(w, h, r).validateIn(w, h)
                assertEquals(PixelCrop(0, 0, w, h), CropPercent().pixels(w, h, r))
            }
    }

    @Test fun principalPointIsTranslatedNotRecentered() {
        val crop = PixelCrop(80, 10, 60, 70)
        assertEquals(43f to 32f, crop.principalPoint(200, 100, 123f, 42f, false))
        assertEquals(19.5f to 39.5f, crop.principalPoint(200, 100, 0f, 0f, true))
        // A valid optical axis can even lie outside a retained off-axis ROI.
        assertEquals(-70f to -5f, crop.principalPoint(200, 100, 10f, 5f, false))
    }

    @Test fun invalidPercentIsRejected() {
        for (v in listOf(-1.0, 46.0, Double.NaN, Double.POSITIVE_INFINITY)) {
            assertThrows(IllegalArgumentException::class.java) { CropPercent(left = v) }
        }
    }
}
