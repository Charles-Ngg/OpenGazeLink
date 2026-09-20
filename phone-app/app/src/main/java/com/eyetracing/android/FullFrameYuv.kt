package com.eyetracing.android

import android.media.Image
import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.min

/** Copy only the requested YUV region to NV21, preserving row-copy fast paths. */
internal class FullFrameYuv {
    internal data class Plane(val buffer: ByteBuffer, val rowStride: Int, val pixelStride: Int)
    private var planarURow = ByteArray(0)
    private var planarVRow = ByteArray(0)
    private var useInterleavedVuFastPath: Boolean? = null

    fun reset() { useInterleavedVuFastPath = null }

    fun copy(image: Image, out: ByteArray, crop: PixelCrop = PixelCrop(0, 0, image.width, image.height)) {
        copy(image.width, image.height,
            image.planes.map { Plane(it.buffer, it.rowStride, it.pixelStride) }.toTypedArray(), out, crop)
    }

    fun copy(frameWidth: Int, frameHeight: Int, planes: Array<Plane>, out: ByteArray,
             crop: PixelCrop = PixelCrop(0, 0, frameWidth, frameHeight)) {
        val sourceWidth = frameWidth
        val sourceHeight = frameHeight
        val width = crop.width
        val height = crop.height
        require(sourceWidth % 2 == 0 && sourceHeight % 2 == 0) {
            "YUV dimensions must be even, got ${sourceWidth}x$sourceHeight"
        }
        crop.validateIn(frameWidth, frameHeight)
        val outputSize = width * height * 3 / 2
        require(out.size == outputSize) {
            "NV21 output buffer is ${out.size} bytes, expected $outputSize"
        }
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]

        val yBuffer = yPlane.buffer.duplicate()
        val yBase = yBuffer.position()
        if (yPlane.pixelStride == 1) {
            for (row in 0 until height) {
                val source = yBase + (crop.y + row) * yPlane.rowStride + crop.x
                require(source + width <= yBuffer.limit()) { "Y plane row $row is truncated" }
                yBuffer.position(source)
                yBuffer.get(out, row * width, width)
            }
        } else {
            for (row in 0 until height) {
                val rowStart = yBase + (crop.y + row) * yPlane.rowStride + crop.x * yPlane.pixelStride
                for (col in 0 until width) {
                    out[row * width + col] = yBuffer.get(rowStart + col * yPlane.pixelStride)
                }
            }
        }

        val chromaHeight = height / 2
        val chromaWidth = width / 2
        val sourceChromaHeight = sourceHeight / 2
        val sourceChromaWidth = sourceWidth / 2
        val chromaX = crop.x / 2
        val chromaY = crop.y / 2
        val fastPath = useInterleavedVuFastPath ?: isInterleavedVu(
            uPlane, vPlane, sourceChromaWidth, sourceChromaHeight
        ).also { useInterleavedVuFastPath = it }
        if (fastPath && copyInterleavedVu(
                vPlane, uPlane, out, width * height,
                chromaX, chromaY, chromaWidth, chromaHeight,
            )
        ) {
            return
        }

        val uBuffer = uPlane.buffer.duplicate()
        val vBuffer = vPlane.buffer.duplicate()
        val uBase = uBuffer.position()
        val vBase = vBuffer.position()
        var outPos = width * height
        if (uPlane.pixelStride == 1 && vPlane.pixelStride == 1) {
            if (planarURow.size != chromaWidth) planarURow = ByteArray(chromaWidth)
            if (planarVRow.size != chromaWidth) planarVRow = ByteArray(chromaWidth)
            for (row in 0 until chromaHeight) {
                val uRowStart = uBase + (chromaY + row) * uPlane.rowStride + chromaX
                val vRowStart = vBase + (chromaY + row) * vPlane.rowStride + chromaX
                require(uRowStart + chromaWidth <= uBuffer.limit()) { "U plane row $row is truncated" }
                require(vRowStart + chromaWidth <= vBuffer.limit()) { "V plane row $row is truncated" }
                uBuffer.position(uRowStart)
                vBuffer.position(vRowStart)
                uBuffer.get(planarURow, 0, chromaWidth)
                vBuffer.get(planarVRow, 0, chromaWidth)
                for (col in 0 until chromaWidth) {
                    out[outPos++] = planarVRow[col]
                    out[outPos++] = planarURow[col]
                }
            }
            return
        }

        for (row in 0 until chromaHeight) {
            val uRowStart = uBase + (chromaY + row) * uPlane.rowStride + chromaX * uPlane.pixelStride
            val vRowStart = vBase + (chromaY + row) * vPlane.rowStride + chromaX * vPlane.pixelStride
            for (col in 0 until chromaWidth) {
                out[outPos++] = vBuffer.get(vRowStart + col * vPlane.pixelStride)
                out[outPos++] = uBuffer.get(uRowStart + col * uPlane.pixelStride)
            }
        }
        return
    }

    private fun isInterleavedVu(
        uPlane: Plane,
        vPlane: Plane,
        chromaWidth: Int,
        chromaHeight: Int,
    ): Boolean {
        val rowBytes = chromaWidth * 2
        if (
            uPlane.pixelStride != 2 || vPlane.pixelStride != 2 ||
            uPlane.rowStride < rowBytes - 1 || vPlane.rowStride < rowBytes ||
            chromaWidth < 2 || chromaHeight < 1
        ) {
            return false
        }
        val u = uPlane.buffer.duplicate()
        val v = vPlane.buffer.duplicate()
        val uBase = u.position()
        val vBase = v.position()
        val rows = intArrayOf(0, chromaHeight / 2, chromaHeight - 1).distinct()
        val columnStep = max(1, (chromaWidth - 1) / 16)
        var compared = 0
        for (row in rows) {
            var col = 0
            while (col < chromaWidth - 1) {
                val uIndex = uBase + row * uPlane.rowStride + col * uPlane.pixelStride
                val vGapIndex = vBase + row * vPlane.rowStride + col * vPlane.pixelStride + 1
                if (uIndex >= u.limit() || vGapIndex >= v.limit() || u.get(uIndex) != v.get(vGapIndex)) {
                    return false
                }
                compared += 1
                col += columnStep
            }
        }
        return compared >= min(16, chromaWidth - 1)
    }

    private fun copyInterleavedVu(
        vPlane: Plane,
        uPlane: Plane,
        out: ByteArray,
        outputOffset: Int,
        chromaX: Int,
        chromaY: Int,
        chromaWidth: Int,
        chromaHeight: Int,
    ): Boolean {
        val v = vPlane.buffer.duplicate()
        val u = uPlane.buffer.duplicate()
        val vBase = v.position()
        val uBase = u.position()
        val rowBytes = chromaWidth * 2
        for (row in 0 until chromaHeight) {
            val sourceRow = chromaY + row
            val vRowStart = vBase + sourceRow * vPlane.rowStride + chromaX * vPlane.pixelStride
            val available = v.limit() - vRowStart
            if (available < rowBytes - 1) return false
            val copyLength = min(rowBytes, available)
            v.position(vRowStart)
            v.get(out, outputOffset + row * rowBytes, copyLength)
            if (copyLength < rowBytes) {
                val finalU = uBase + sourceRow * uPlane.rowStride +
                    (chromaX + chromaWidth - 1) * uPlane.pixelStride
                if (finalU >= u.limit()) return false
                out[outputOffset + (row + 1) * rowBytes - 1] = u.get(finalU)
            }
        }
        return true
    }
}
