package com.eyetracing.android

import android.os.SystemClock
import org.json.JSONObject
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.math.min

internal data class FrameSendStats(val bytes: Int, val chunks: Int)

internal class UdpYuvSender(host: String, port: Int, private val chunkPayloadSize: Int) : AutoCloseable {
    private val socket = DatagramSocket().apply {
        // Keep back-pressure close to the camera. A multi-megabyte UDP queue
        // makes send() look fast while displaying frames hundreds of ms late.
        sendBufferSize = 128 * 1024
        connect(InetAddress.getByName(host), port)
    }
    private var frameSeq = 0
    init {
        // Reply independently of JPEG encoding and frame sending. Each reply
        // includes phone receive/send times so PC can subtract reply processing.
        Thread({
            val bytes = ByteArray(64)
            val incoming = DatagramPacket(bytes, bytes.size)
            while (!socket.isClosed) {
                try {
                    incoming.length = bytes.size
                    socket.receive(incoming)
                    val receivedNs = SystemClock.elapsedRealtimeNanos()
                    if (incoming.length != 32) continue
                    val input = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                    if (input.int != 0x54435945 || input.short.toInt() != 1 || input.short.toInt() != 1) continue
                    val pcSendNs = input.long
                    val reply = ByteBuffer.allocate(32).order(ByteOrder.LITTLE_ENDIAN)
                    reply.putInt(0x54435945)
                    reply.putShort(1.toShort())
                    reply.putShort(2.toShort())
                    reply.putLong(pcSendNs)
                    reply.putLong(receivedNs)
                    reply.putLong(SystemClock.elapsedRealtimeNanos())
                    socket.send(DatagramPacket(reply.array(), 32))
                } catch (_: Exception) {
                    if (!socket.isClosed) Thread.sleep(10)
                }
            }
        }, "udp-clock-reply").apply { isDaemon = true; start() }
    }
    private val packetBytes = ByteArray(HEADER_SIZE + chunkPayloadSize)
    private val buffer = ByteBuffer.wrap(packetBytes).order(ByteOrder.LITTLE_ENDIAN)
    private val packet = DatagramPacket(packetBytes, packetBytes.size)

    fun sendFrame(
        payload: ByteArray,
        payloadLength: Int,
        width: Int,
        height: Int,
        sensorTimeNs: Long,
        format: Byte,
    ): FrameSendStats {
        require(payloadLength in 0..payload.size)
        val seq = frameSeq++
        val chunkCount = ceil(payloadLength / chunkPayloadSize.toDouble()).toInt()
        val frameSendTimeNs = SystemClock.elapsedRealtimeNanos()
        var offset = 0
        var sent = 0
        for (chunkIndex in 0 until chunkCount) {
            val payloadSize = min(chunkPayloadSize, payloadLength - offset)
            buffer.clear()
            buffer.putInt(MAGIC)
            buffer.putShort(VERSION.toShort())
            buffer.putShort(HEADER_SIZE.toShort())
            buffer.putInt(seq)
            buffer.putShort(chunkIndex.toShort())
            buffer.putShort(chunkCount.toShort())
            buffer.putShort(width.toShort())
            buffer.putShort(height.toShort())
            buffer.put(format)
            buffer.put(0)
            buffer.putLong(sensorTimeNs)
            buffer.putLong(frameSendTimeNs)
            buffer.putInt(payloadSize)
            System.arraycopy(payload, offset, packetBytes, HEADER_SIZE, payloadSize)
            packet.length = HEADER_SIZE + payloadSize
            socket.send(packet)
            offset += payloadSize
            sent += payloadSize
        }
        return FrameSendStats(sent, chunkCount)
    }

    fun sendIntrinsics(message: JSONObject) {
        val payload = message.toString().toByteArray(Charsets.UTF_8)
        val packetBytes = ByteArray(INTRINSICS_HEADER_SIZE + payload.size)
        val buffer = ByteBuffer.wrap(packetBytes).order(ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(INTRINSICS_MAGIC)
        buffer.putShort(INTRINSICS_VERSION.toShort())
        buffer.putShort(INTRINSICS_HEADER_SIZE.toShort())
        buffer.putInt(payload.size)
        buffer.put(payload)
        socket.send(DatagramPacket(packetBytes, packetBytes.size))
    }

    override fun close() {
        socket.close()
    }

    companion object {
        private const val MAGIC = 0x56555945
        private const val VERSION = 1
        private const val HEADER_SIZE = 42
        const val FORMAT_NV21: Byte = 1
        const val FORMAT_JPEG: Byte = 2
        private const val INTRINSICS_MAGIC = 0x49435945
        private const val INTRINSICS_VERSION = 1
        private const val INTRINSICS_HEADER_SIZE = 12
    }
}
