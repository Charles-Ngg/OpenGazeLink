package com.eyetracing.android

import org.junit.Assert.*
import org.junit.Test

class CaptureSelectionTest {
    private fun high(width: Int, height: Int, fps: Int) =
        CaptureOption("front", SessionMode.HIGH_SPEED, width, height, fps, encoderName = "hardware")

    @Test fun pickerValuesAreUniqueAndSortedNotRepeatedForEveryPair() {
        val options = listOf(high(1280, 720, 120), high(1280, 720, 240),
            high(1920, 1080, 120), high(640, 480, 240), high(1280, 720, 120))
        assertEquals(listOf(CaptureResolution(1920, 1080), CaptureResolution(1280, 720),
            CaptureResolution(640, 480)), CaptureOptions.resolutions(options))
        assertEquals(listOf(240, 120), CaptureOptions.frameRates(options))
    }

    @Test fun changingResolutionPreservesCompatibleFrameRate() {
        val current = high(1280, 720, 240)
        val target = high(1920, 1080, 240)
        val options = CaptureOptions.ordered(listOf(current, target, high(1920, 1080, 120)))
        assertEquals(target, CaptureOptions.selectResolution(options, target.resolution, current))
    }

    @Test fun changingFrameRatePreservesCompatibleResolution() {
        val current = high(1920, 1080, 120)
        val target = high(1920, 1080, 240)
        val options = CaptureOptions.ordered(listOf(current, target, high(1280, 720, 240)))
        assertEquals(target, CaptureOptions.selectFrameRate(options, 240, current))
    }

    @Test fun resolutionFallbackChoosesNearestFrameRateWithLowerRateOnTie() {
        val current = high(1280, 720, 180)
        val target = high(1920, 1080, 120)
        val options = listOf(high(1920, 1080, 360), high(1920, 1080, 240), current, target)
        assertEquals(target, CaptureOptions.selectResolution(options, target.resolution, current))
    }

    @Test fun frameRateFallbackChoosesNearestResolutionWithSmallerSizeOnTie() {
        val current = high(1280, 720, 120)
        val target = high(1024, 720, 240)
        val options = listOf(high(1920, 1080, 240), high(1536, 720, 240), current, target)
        assertEquals(target, CaptureOptions.selectFrameRate(options, 240, current))
    }

    @Test fun samePixelCountDoesNotConfusePortraitAndLandscape() {
        val current = high(1080, 1920, 120)
        val target = high(1080, 1920, 240)
        assertEquals(target, CaptureOptions.selectFrameRate(
            listOf(high(1920, 1080, 240), target, current), 240, current))
    }

    @Test fun eitherPickerCanReachDisconnectedCombinations() {
        val largeSlow = high(1920, 1080, 120)
        val smallFast = high(1280, 720, 240)
        val options = listOf(largeSlow, smallFast)
        assertEquals(smallFast, CaptureOptions.selectFrameRate(options, 240, largeSlow))
        assertEquals(smallFast, CaptureOptions.selectResolution(options, smallFast.resolution, largeSlow))
        assertEquals(largeSlow, CaptureOptions.selectFrameRate(options, 120, smallFast))
        assertEquals(largeSlow, CaptureOptions.selectResolution(options, largeSlow.resolution, smallFast))
        assertEquals(2, CaptureOptions.resolutions(options).size)
        assertEquals(2, CaptureOptions.frameRates(options).size)
    }

    @Test fun eitherFieldCanBeSelectedWithoutAnExistingPreference() {
        val options = listOf(high(1920, 1080, 120), high(1280, 720, 240))
        assertEquals(options[1], CaptureOptions.selectFrameRate(options, 240, null))
        assertEquals(options[1], CaptureOptions.selectResolution(options, options[1].resolution, null))
    }

    @Test fun absentAndExcludedValuesNeverCreateUnsupportedCombinations() {
        val options = listOf(high(1920, 1080, 120))
        val excluded = high(1280, 720, 240)
        assertNull(CaptureOptions.selectFrameRate(options, excluded.fps, options[0]))
        assertNull(CaptureOptions.selectResolution(options, excluded.resolution, options[0]))
        assertEquals(emptyList<Int>(), CaptureOptions.frameRates(emptyList()))
        assertEquals(emptyList<CaptureResolution>(), CaptureOptions.resolutions(emptyList()))
        assertNull(CaptureOptions.selectFrameRate(emptyList(), 120, options[0]))
        assertNull(CaptureOptions.selectResolution(emptyList(), options[0].resolution, options[0]))
    }

    @Test fun hundredsOfPairsRemainReachableInBothOrdersWithoutInventingAny() {
        val options = CaptureOptions.ordered((1..80).flatMap { size ->
            listOf(15, 24, 30, 48, 60).mapIndexedNotNull { index, fps ->
                if ((size + index) % 4 == 0) null
                else CaptureOption("rear", SessionMode.CAMERA, 640 + size * 16, 480 + size * 12, fps)
            }
        })
        assertTrue(options.size >= 300)
        assertEquals(80, CaptureOptions.resolutions(options).size)
        assertEquals(5, CaptureOptions.frameRates(options).size)
        for (initial in listOf(options.first(), options[options.size / 2], options.last())) {
            for (target in options) {
                val sizeFirst = CaptureOptions.selectResolution(options, target.resolution, initial)!!
                val rateFirst = CaptureOptions.selectFrameRate(options, target.fps, initial)!!
                assertTrue(sizeFirst in options)
                assertTrue(rateFirst in options)
                assertEquals(target.resolution, sizeFirst.resolution)
                assertEquals(target.fps, rateFirst.fps)
                assertEquals(target, CaptureOptions.selectFrameRate(options, target.fps, sizeFirst))
                assertEquals(target, CaptureOptions.selectResolution(options, target.resolution, rateFirst))
            }
        }
    }
}
