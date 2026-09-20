package com.eyetracing.android

import org.junit.Assert.*
import org.junit.Test

class CaptureOptionTest {
    private fun high(width: Int, height: Int, fps: Int, low: Int = fps, encoder: String? = "hardware") =
        CaptureOption("front", SessionMode.HIGH_SPEED, width, height, fps, low, encoder)

    @Test fun fixedRangeAndEncoderAreRequired() {
        val accepted = high(1920, 1080, 120)
        val options = CaptureOptions.ordered(listOf(accepted, high(1280, 720, 120, 30),
            high(640, 480, 60), high(1280, 720, 120, encoder = null), high(641, 480, 120)))
        assertEquals(listOf(accepted), options)
    }
    @Test fun recommend120WithoutHardCodedResolution() {
        val only = high(1920, 1080, 120)
        assertEquals(only, CaptureOptions.selected(CaptureOptions.ordered(listOf(only)), null))
        assertEquals(high(1280, 720, 120), CaptureOptions.ordered(listOf(
            high(640, 480, 240), only, high(1280, 720, 120))).first())
    }
    @Test fun doNotInventSizeRateCrossProducts() {
        val declared = listOf(high(1920, 1080, 120), high(1280, 720, 240))
        assertEquals(declared.toSet(), CaptureOptions.ordered(declared).toSet())
        assertFalse(CaptureOptions.ordered(declared).any { it.width == 1920 && it.fps == 240 })
    }
    @Test fun preferFixedAeRangeAndRemoveDuplicates() {
        val variable = CaptureOption("0", SessionMode.CAMERA, 1280, 720, 30, 15)
        val fixed = variable.copy(fpsLower = 30)
        assertEquals(listOf(fixed), CaptureOptions.ordered(listOf(variable, fixed)))
    }
    @Test fun validateSavedSelectionAgainstCurrentCapabilities() {
        val options = CaptureOptions.ordered(listOf(high(1280, 720, 120), high(1920, 1080, 120)))
        assertEquals(options[1], CaptureOptions.selected(options, options[1].key))
        assertEquals(options.first(), CaptureOptions.selected(options, "stale:cropped:mode"))
        assertNull(CaptureOptions.selected(emptyList(), options[0].key))
    }
}
