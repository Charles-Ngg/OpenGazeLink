package com.eyetracing.android

import android.os.SystemClock
import java.net.InetSocketAddress
import java.net.Socket
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit

/** Keep the TCP stream alive under transient PC decode/model load. */
internal class AvcTcpSender(host: String, port: Int, private val width: Int, private val height: Int,
    private val onSent: (Int) -> Unit, private val onError: (Throwable) -> Unit,
) : AutoCloseable {
    private data class Packet(val data: ByteArray, val sensor: Long, val encoded: Long, val flags: Int)
    private val socket = Socket()
    private val queue = ArrayBlockingQueue<Packet>(16)
    @Volatile private var closed = false
    init {
        try {
            socket.tcpNoDelay = true
            socket.sendBufferSize = 32 * 1024
            socket.connect(InetSocketAddress(host, port), 3000)
        } catch (error: Exception) { socket.close(); throw error }
        Thread({
            try {
                val output = socket.getOutputStream()
                var sequence = 0
                while (!closed) {
                    val packet = queue.poll(100, TimeUnit.MILLISECONDS) ?: continue
                    val now = SystemClock.elapsedRealtimeNanos()
                    val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
                    header.putInt(0x31435641) // AVC1
                    header.putInt(sequence++).putInt(packet.flags)
                    header.putShort(width.toShort()).putShort(height.toShort())
                    header.putLong(packet.sensor).putLong(packet.encoded).putLong(now)
                    header.putInt(packet.data.size)
                    output.write(header.array()); output.write(packet.data)
                    if (packet.flags and 2 == 0) onSent(packet.data.size + 44)
                }
            } catch (error: Exception) { if (!closed) { close(); onError(error) } }
        }, "avc-tcp-send").apply { isDaemon = true; start() }
    }
    fun offer(data: ByteArray, sensor: Long, encoded: Long, flags: Int) {
        if (closed) return
        val packet = Packet(data, sensor, encoded, flags)
        if (!queue.offer(packet)) {
            queue.poll()
            if (!queue.offer(packet)) return
        }
    }
    override fun close() {
        closed = true
        runCatching { socket.close() }
        queue.clear()
    }
}
