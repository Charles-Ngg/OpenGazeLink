package com.eyetracing.android

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.media.MediaFormat
import android.os.Build
import android.os.Handler
import android.os.SystemClock
import android.view.Surface

/** The same format check is used by discovery and by the actual encoder. */
internal class AvcEncoder(
    option: CaptureOption, handler: Handler,
    onFrame: (Long) -> Unit,
    onPacket: (ByteArray, Long, Long, Int) -> Unit,
    onError: (Throwable) -> Unit,
) : AutoCloseable {
    private var codec: MediaCodec? = null
    private var output: Surface? = null
    @Volatile private var closed = false
    private var started = false
    val surface: Surface get() = checkNotNull(output)

    init {
        try {
            val name = option.encoderName ?: findEncoder(option.width, option.height, option.fps)
                ?: error("No compatible hardware H.264 encoder")
            val encoder = MediaCodec.createByCodecName(name)
            codec = encoder
            encoder.setCallback(object : MediaCodec.Callback() {
                override fun onInputBufferAvailable(codec: MediaCodec, index: Int) = Unit
                override fun onOutputBufferAvailable(codec: MediaCodec, index: Int, info: MediaCodec.BufferInfo) {
                    if (closed) return
                    try {
                        val now = SystemClock.elapsedRealtimeNanos()
                        if (info.size > 0) {
                            check(info.flags and MediaCodec.BUFFER_FLAG_PARTIAL_FRAME == 0) { "Partial H.264 output is unsupported" }
                            val data = checkNotNull(codec.getOutputBuffer(index)).duplicate()
                            data.position(info.offset); data.limit(info.offset + info.size)
                            val packet = ByteArray(info.size)
                            data.get(packet)
                            onPacket(packet, info.presentationTimeUs * 1000L, now, info.flags)
                            if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0) onFrame(info.presentationTimeUs * 1000L)
                        }
                        codec.releaseOutputBuffer(index, false)
                    } catch (error: Exception) { if (!closed) onError(error) }
                }
                override fun onOutputFormatChanged(codec: MediaCodec, format: MediaFormat) = Unit
                override fun onError(codec: MediaCodec, error: MediaCodec.CodecException) {
                    if (!closed) onError(error)
                }
            }, handler)
            encoder.configure(format(option.width, option.height, option.fps), null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            output = encoder.createInputSurface()
            encoder.start()
            started = true
        } catch (error: Exception) { close(); throw error }
    }

    override fun close() {
        if (closed) return
        closed = true
        if (started) runCatching { codec?.stop() }
        runCatching { codec?.release() }
        output?.release()
        codec = null; output = null
    }

    companion object {
        private fun format(width: Int, height: Int, fps: Int): MediaFormat =
            MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
                setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
                setInteger(MediaFormat.KEY_BIT_RATE, (width.toLong() * height * fps / 8).coerceIn(2_000_000, 80_000_000).toInt())
                setInteger(MediaFormat.KEY_FRAME_RATE, fps)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
                setInteger(MediaFormat.KEY_PROFILE, MediaCodecInfo.CodecProfileLevel.AVCProfileBaseline)
                setInteger(MediaFormat.KEY_LATENCY, 0)
                if (Build.VERSION.SDK_INT >= 29) setInteger(MediaFormat.KEY_MAX_B_FRAMES, 0)
            }

        fun findEncoder(width: Int, height: Int, fps: Int): String? {
            val requested = format(width, height, fps)
            return MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos.firstOrNull { info ->
                info.isEncoder && info.supportedTypes.any { it.equals(MediaFormat.MIMETYPE_VIDEO_AVC, true) } &&
                    (if (Build.VERSION.SDK_INT >= 29) info.isHardwareAccelerated
                    else !info.name.startsWith("OMX.google.") && !info.name.startsWith("c2.android.")) &&
                    runCatching { info.getCapabilitiesForType(MediaFormat.MIMETYPE_VIDEO_AVC).isFormatSupported(requested) }.getOrDefault(false)
            }?.name
        }
    }
}
