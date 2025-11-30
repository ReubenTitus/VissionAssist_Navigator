package com.example.vissionav

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

class YoloDetector(private val context: Context, private val focalLengthPixels: Float) {
    private var interpreter: Interpreter? = null
    private val labels: List<String>
    private val inputSize = 320
    private val confidenceThreshold = 0.5f
    private val iouThreshold = 0.45f

    private val objectSizes = mapOf(
        "person" to 0.5f, "bicycle" to 0.6f, "car" to 1.8f, "motorcycle" to 0.8f, "airplane" to 10.0f,
        "bus" to 2.5f, "train" to 3.0f, "truck" to 2.5f, "boat" to 2.0f, "traffic light" to 0.3f,
        "fire hydrant" to 0.4f, "stop sign" to 0.75f, "parking meter" to 0.3f, "bench" to 1.5f,
        "bird" to 0.2f, "cat" to 0.3f, "dog" to 0.4f, "horse" to 0.8f, "sheep" to 0.6f, "cow" to 0.9f,
        "elephant" to 2.5f, "bear" to 1.0f, "zebra" to 0.8f, "giraffe" to 1.0f, "backpack" to 0.4f,
        "umbrella" to 1.0f, "handbag" to 0.3f, "tie" to 0.1f, "suitcase" to 0.5f, "frisbee" to 0.25f,
        "skis" to 0.1f, "snowboard" to 0.3f, "sports ball" to 0.22f, "kite" to 1.0f, "baseball bat" to 0.07f,
        "baseball glove" to 0.25f, "skateboard" to 0.2f, "surfboard" to 0.5f, "tennis racket" to 0.27f,
        "bottle" to 0.07f, "wine glass" to 0.08f, "cup" to 0.08f, "fork" to 0.03f, "knife" to 0.02f,
        "spoon" to 0.04f, "bowl" to 0.15f, "banana" to 0.03f, "apple" to 0.08f, "sandwich" to 0.12f,
        "orange" to 0.08f, "broccoli" to 0.15f, "carrot" to 0.03f, "hot dog" to 0.03f, "pizza" to 0.3f,
        "donut" to 0.1f, "cake" to 0.25f, "chair" to 0.5f, "couch" to 1.8f, "potted plant" to 0.3f,
        "bed" to 1.5f, "dining table" to 1.0f, "toilet" to 0.5f, "TV" to 0.8f, "laptop" to 0.35f,
        "mouse" to 0.06f, "remote" to 0.05f, "keyboard" to 0.45f, "cell phone" to 0.07f, "microwave" to 0.5f,
        "oven" to 0.6f, "toaster" to 0.25f, "sink" to 0.6f, "refrigerator" to 0.8f, "book" to 0.15f,
        "clock" to 0.3f, "vase" to 0.15f, "scissors" to 0.08f, "teddy bear" to 0.3f, "hair drier" to 0.08f,
        "toothbrush" to 0.02f
    )

    companion object {
        private const val TAG = "YoloDetector"
    }

    init {
        labels = try {
            context.assets.open("labels.txt").bufferedReader().useLines { lines ->
                lines.toList().map { it.trim() }.filter { it.isNotEmpty() }
            }.also {
                Log.d(TAG, "Loaded ${it.size} labels: ${it.joinToString(", ")}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load labels.txt: $e")
            emptyList()
        }
    }

    fun loadModel() {
        try {
            val modelData = context.assets.open("yolov8n_float32.tflite").readBytes()
            val modelBuffer = ByteBuffer.allocateDirect(modelData.size).apply {
                order(ByteOrder.nativeOrder())
                put(modelData)
                rewind()
            }
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }
            interpreter = Interpreter(modelBuffer, options)
            Log.d(TAG, "Model loaded on CPU. Input shape: ${interpreter?.getInputTensor(0)?.shape()?.joinToString()}")
            Log.d(TAG, "Output shape: ${interpreter?.getOutputTensor(0)?.shape()?.joinToString()}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model: $e")
            throw e
        }
    }

    fun detectObjects(image: ImageProxy): List<Detection> {
        interpreter ?: run {
            Log.w(TAG, "Interpreter not initialized")
            return emptyList()
        }

        val bitmap = image.toBitmap()
        Log.d(TAG, "ImageProxy size: ${image.width}x${image.height}, Rotation: ${image.imageInfo.rotationDegrees}")
        if (!isImageValid(bitmap)) {
            Log.d(TAG, "Skipping detection: Image is blank or low variance")
            return emptyList()
        }

        val inputTensor = preprocessImage(bitmap)
        val outputTensor = Array(1) { Array(84) { FloatArray(2100) } }

        interpreter?.run(inputTensor, outputTensor)
        return postProcessOutput(outputTensor[0], image.width, image.height)
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val inputTensor = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4).apply {
            order(ByteOrder.nativeOrder())
        }

        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in pixels) {
            inputTensor.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // R
            inputTensor.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // G
            inputTensor.putFloat((pixel and 0xFF) / 255.0f)          // B
        }
        inputTensor.rewind()
        return inputTensor
    }

    private fun isImageValid(bitmap: Bitmap): Boolean {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var mean = 0.0
        var variance = 0.0

        for (pixel in pixels) {
            val gray = ((pixel shr 16) and 0xFF) * 0.299 + ((pixel shr 8) and 0xFF) * 0.587 + (pixel and 0xFF) * 0.114
            mean += gray
        }
        mean /= pixels.size

        for (pixel in pixels) {
            val gray = ((pixel shr 16) and 0xFF) * 0.299 + ((pixel shr 8) and 0xFF) * 0.587 + (pixel and 0xFF) * 0.114
            variance += (gray - mean).pow(2)
        }
        variance /= pixels.size

        Log.d(TAG, "Image variance: $variance")
        return variance > 50
    }

    fun postProcessOutput(output: Array<FloatArray>, origWidth: Int, origHeight: Int): List<Detection> {
        val results = mutableListOf<Detection>()
        val numClasses = 80
        val modelSize = 320f

        if (labels.size != numClasses) {
            Log.e(TAG, "Mismatch: labels.txt has ${labels.size} labels, but model expects $numClasses")
            return emptyList()
        }

        for (i in 0 until 2100) {
            val boxData = FloatArray(84) { output[it][i] }
            val classScores = boxData.sliceArray(4 until 4 + numClasses)
            val maxScore = classScores.maxOrNull() ?: 0f

            if (maxScore >= confidenceThreshold) {
                val classIndex = classScores.indexOfFirst { it == maxScore }
                val label = labels[classIndex]

                val xCenter = boxData[0]
                val yCenter = boxData[1]
                val width = boxData[2]
                val height = boxData[3]

                val x = (xCenter - width / 2) * modelSize
                val y = (yCenter - height / 2) * modelSize
                val w = width * modelSize
                val h = height * modelSize

                val realWidth = objectSizes[label]
                val distance = if (realWidth != null && w > 0) {
                    focalLengthPixels * realWidth / w
                } else {
                    null
                }

                Log.d(TAG, "Raw detection: $label at ($x, $y, $w, $h) with confidence $maxScore, distance: $distance")
                results.add(Detection(label, maxScore, x, y, w, h, distance))
            }
        }
        return applyNMS(results)
    }

    private fun applyNMS(boxes: List<Detection>): List<Detection> {
        val sortedBoxes = boxes.sortedByDescending { it.confidence }.toMutableList()
        val selectedBoxes = mutableListOf<Detection>()

        while (sortedBoxes.isNotEmpty()) {
            val box = sortedBoxes.removeAt(0)
            selectedBoxes.add(box)
            sortedBoxes.removeAll { calculateIoU(box, it) > iouThreshold }
        }
        return selectedBoxes
    }

    private fun calculateIoU(box1: Detection, box2: Detection): Float {
        val xLeft = max(box1.x, box2.x)
        val yTop = max(box1.y, box2.y)
        val xRight = min(box1.x + box1.width, box2.x + box2.width)
        val yBottom = min(box1.y + box1.height, box2.y + box2.height)

        if (xRight <= xLeft || yBottom <= yTop) return 0f

        val intersection = (xRight - xLeft) * (yBottom - yTop)
        val union = box1.width * box1.height + box2.width * box2.height - intersection
        return intersection / union
    }

    fun close() {
        interpreter?.close()
        interpreter = null
        Log.d(TAG, "Model closed")
    }
}

data class Detection(
    val label: String,
    val confidence: Float,
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float,
    val distance: Float? = null
)

fun ImageProxy.toBitmap(): Bitmap {
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
    val out = java.io.ByteArrayOutputStream()
    yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 100, out)
    val imageBytes = out.toByteArray()
    return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}