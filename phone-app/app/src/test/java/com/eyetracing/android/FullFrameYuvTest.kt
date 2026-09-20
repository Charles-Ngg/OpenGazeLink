package com.eyetracing.android

import java.nio.ByteBuffer
import org.junit.Assert.*
import org.junit.Test

class FullFrameYuvTest {
    private fun plane(bytes: ByteArray, row: Int, pixel: Int = 1) = FullFrameYuv.Plane(ByteBuffer.wrap(bytes), row, pixel)
    private val y = ByteArray(16) { it.toByte() }
    private val expected = y + byteArrayOf(80, 50, 81, 51, 82, 52, 83, 53)

    @Test fun includesAllFourCorners() {
        val result = ByteArray(24)
        FullFrameYuv().copy(4, 4, arrayOf(plane(y, 4), plane(byteArrayOf(50, 51, 52, 53), 2),
            plane(byteArrayOf(80, 81, 82, 83), 2)), result)
        assertArrayEquals(expected, result)
    }
    @Test fun rowPaddingIsNotCopiedOrCropped() {
        val padded = ByteArray(24) { -1 }
        for (row in 0..3) y.copyInto(padded, row * 6, row * 4, row * 4 + 4)
        val result = ByteArray(24)
        FullFrameYuv().copy(4, 4, arrayOf(plane(padded, 6), plane(byteArrayOf(50, 51, -1, 52, 53), 3),
            plane(byteArrayOf(80, 81, -1, 82, 83), 3)), result)
        assertArrayEquals(expected, result)
    }
    @Test fun interleavedVuHandlesTruncatedTail() {
        val vu = byteArrayOf(80, 50, 81, 51, 82, 52, 83, 53)
        val v = ByteBuffer.wrap(vu).apply { limit(7) }
        val u = ByteBuffer.wrap(vu).apply { position(1) }
        val result = ByteArray(24)
        FullFrameYuv().copy(4, 4, arrayOf(plane(y, 4), FullFrameYuv.Plane(u, 4, 2),
            FullFrameYuv.Plane(v, 4, 2)), result)
        assertArrayEquals(expected, result)
    }
    @Test fun interleavedUvUsesCorrectChannelOrder() {
        val uv = byteArrayOf(50, 80, 51, 81, 52, 82, 53, 83)
        val result = ByteArray(24)
        FullFrameYuv().copy(4, 4, arrayOf(plane(y, 4), plane(uv, 4, 2),
            FullFrameYuv.Plane(ByteBuffer.wrap(uv).apply { position(1) }, 4, 2)), result)
        assertArrayEquals(expected, result)
    }
    @Test(expected = IllegalArgumentException::class)
    fun oddSizesAreRejectedNotCropped() {
        FullFrameYuv().copy(3, 4, arrayOf(plane(y, 4)), ByteArray(18))
    }

    @Test fun croppedCopyMatchesReferenceAcrossPlaneLayoutsAndPositions() {
        val w = 12; val h = 8
        // Planar, VU (including a truncated last U), UV and generic strided planes.
        for (layout in 0..3) for (crop in listOf(PixelCrop(2, 2, 6, 4), PixelCrop(8, 4, 4, 4))) {
            val yStride = if (layout == 3) 2 else 1
            val yRow = w * yStride + 5
            val yBuffer = ByteBuffer.allocate(3 + yRow * h).apply { position(3) }
            for (y in 0 until h) for (x in 0 until w) yBuffer.put(3 + y * yRow + x * yStride, (y * w + x).toByte())
            val pixel = when (layout) { 0 -> 1; 3 -> 3; else -> 2 }
            val row = w / 2 * pixel + 4
            val uBase = if (layout == 1) 4 else 3
            val vBase = if (layout == 2) 4 else 3
            val u = ByteBuffer.allocate(4 + row * h / 2).apply { position(uBase) }
            val v = if (layout in 1..2) u.duplicate().apply { position(vBase) }
                else ByteBuffer.allocate(4 + row * h / 2).apply { position(vBase) }
            for (y in 0 until h / 2) for (x in 0 until w / 2) {
                u.put(uBase + y * row + x * pixel, (50 + y * w / 2 + x).toByte())
                v.put(vBase + y * row + x * pixel, (100 + y * w / 2 + x).toByte())
            }
            if (layout == 1) v.limit(vBase + (h / 2 - 1) * row + (w / 2 - 1) * pixel + 1)
            val out = ByteArray(crop.width * crop.height * 3 / 2)
            val converter = FullFrameYuv()
            repeat(2) { // Exercise the cached VU fast-path decision, too.
                converter.copy(w, h, arrayOf(FullFrameYuv.Plane(yBuffer, yRow, yStride),
                    FullFrameYuv.Plane(u, row, pixel), FullFrameYuv.Plane(v, row, pixel)), out, crop)
                val expected = ArrayList<Byte>()
                for (y in crop.y until crop.y + crop.height) for (x in crop.x until crop.x + crop.width)
                    expected.add((y * w + x).toByte())
                for (y in crop.y / 2 until (crop.y + crop.height) / 2) for (x in crop.x / 2 until (crop.x + crop.width) / 2) {
                    expected.add((100 + y * w / 2 + x).toByte()); expected.add((50 + y * w / 2 + x).toByte())
                }
                assertArrayEquals("layout=$layout crop=$crop", expected.toByteArray(), out)
                assertEquals(3, yBuffer.position()); assertEquals(uBase, u.position()); assertEquals(vBase, v.position())
            }
        }
    }
}
