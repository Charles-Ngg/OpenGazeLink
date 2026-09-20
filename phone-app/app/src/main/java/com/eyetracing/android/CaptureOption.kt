package com.eyetracing.android

import kotlin.math.abs

internal enum class SessionMode { HIGH_SPEED, CAMERA }

internal data class CaptureResolution(val width: Int, val height: Int) {
    val pixels: Long get() = width.toLong() * height
    val label: String get() = "" + width + " × " + height
}

/** A single advertised size/rate pair, never a cross product of unrelated lists. */
internal data class CaptureOption(
    val cameraId: String,
    val mode: SessionMode,
    val width: Int,
    val height: Int,
    val fps: Int,
    val fpsLower: Int = fps,
    val encoderName: String? = null,
) {
    val key: String get() = listOf(cameraId, mode.name, width, height, fps).joinToString(":")
    val resolution: CaptureResolution get() = CaptureResolution(width, height)
    val label: String get() = resolution.label + " · " + fps + " FPS"
}

internal object CaptureOptions {
    fun ordered(options: List<CaptureOption>): List<CaptureOption> =
        options.filter {
            it.width > 0 && it.height > 0 && it.width % 2 == 0 && it.height % 2 == 0 &&
                it.fps > 0 && it.fpsLower in 1..it.fps &&
                (it.mode != SessionMode.HIGH_SPEED ||
                    (it.fps >= 120 && it.fpsLower == it.fps && !it.encoderName.isNullOrBlank()))
        }.sortedWith(compareBy<CaptureOption>(
            { if (it.fps == (if (it.mode == SessionMode.HIGH_SPEED) 120 else 30)) 0 else 1 },
            { abs(it.width.toLong() * it.height - 1280L * 720L) },
            { abs(it.fps - (if (it.mode == SessionMode.HIGH_SPEED) 120 else 30)) },
            { it.fps - it.fpsLower },
            { it.width },
            { it.height },
        )).distinctBy { it.key }

    fun selected(options: List<CaptureOption>, savedKey: String?): CaptureOption? =
        options.firstOrNull { it.key == savedKey } ?: options.firstOrNull()

    // Both pickers include the distinct values advertised for this camera/mode.
    // Filtering each by the other would trap disconnected pairs, e.g. 1080p/30
    // and 720p/120. The last edited dimension wins; resolve the other to an
    // existing CaptureOption, never synthesize a size/rate cross product.
    fun resolutions(options: List<CaptureOption>): List<CaptureResolution> =
        options.map { it.resolution }.distinct().sortedWith(
            compareByDescending<CaptureResolution> { it.pixels }
                .thenByDescending { it.width }.thenByDescending { it.height })

    fun frameRates(options: List<CaptureOption>): List<Int> =
        options.map { it.fps }.distinct().sortedDescending()

    fun selectResolution(options: List<CaptureOption>, resolution: CaptureResolution,
                         preferred: CaptureOption?): CaptureOption? {
        val matching = options.filter { it.resolution == resolution }
        if (preferred == null) return matching.firstOrNull()
        // Keep FPS when possible; otherwise choose the nearest supported rate,
        // preferring the lower rate on a tie to avoid an unnecessary load jump.
        return matching.minWithOrNull(compareBy<CaptureOption>(
            { abs(it.fps.toLong() - preferred.fps) }, { it.fps }))
    }

    fun selectFrameRate(options: List<CaptureOption>, fps: Int,
                        preferred: CaptureOption?): CaptureOption? {
        val matching = options.filter { it.fps == fps }
        if (preferred == null) return matching.firstOrNull()
        return matching.minWithOrNull(compareBy<CaptureOption>(
            { if (it.resolution == preferred.resolution) 0 else 1 },
            { abs(it.resolution.pixels - preferred.resolution.pixels) },
            { abs(it.width.toLong() - preferred.width) + abs(it.height.toLong() - preferred.height) },
            { it.resolution.pixels }, { it.width }, { it.height }))
    }
}
