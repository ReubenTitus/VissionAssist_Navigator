package com.example.vissionav

import android.Manifest
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material.Button
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import kotlin.math.roundToInt

class MainActivity : ComponentActivity(), TextToSpeech.OnInitListener {
    private lateinit var yoloDetector: YoloDetector
    private lateinit var tts: TextToSpeech
    private val REQUEST_CODE_PERMISSIONS = 10
    private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    private val lastSpokenTime = ConcurrentHashMap<String, Long>()
    private val executor = Executors.newSingleThreadExecutor()
    private val focalLengthPixels = 400f

    companion object {
        private const val TAG = "MainActivity"
        private const val SPEAK_COOLDOWN_MS = 5000L
        private const val DETECTION_LIFETIME_MS = 1000L
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        yoloDetector = YoloDetector(this, focalLengthPixels)
        tts = TextToSpeech(this, this)

        if (allPermissionsGranted()) {
            yoloDetector.loadModel()
            setContent {
                MainScreen()
            }
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.US
            tts.speak("Vision Assist Navigator opening", TextToSpeech.QUEUE_FLUSH, null, null)
            Log.d(TAG, "TextToSpeech initialized successfully")
        } else {
            Log.e(TAG, "TextToSpeech initialization failed")
        }
    }

    fun onCameraBound() {
        Log.d(TAG, "Camera bound to lifecycle successfully")
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS && allPermissionsGranted()) {
            yoloDetector.loadModel()
            setContent {
                MainScreen()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        yoloDetector.close()
        tts.shutdown()
        executor.shutdown()
    }

    data class PersistentDetection(
        val detection: Detection,
        val timestamp: Long,
        val imageWidth: Int,
        val imageHeight: Int,
        val rotationDegrees: Int
    )

    @Composable
    fun MainScreen() {
        val isCameraStarted = remember { mutableStateOf(false) }

        if (!isCameraStarted.value) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Button(onClick = { isCameraStarted.value = true }) {
                    Text("Start")
                }
            }
        } else {
            ObjectDetectionScreen()
        }
    }

    @Composable
    fun ObjectDetectionScreen() {
        val context = LocalContext.current
        val lifecycleOwner = LocalLifecycleOwner.current
        val configuration = LocalConfiguration.current
        val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
        val persistentDetections = remember { mutableStateListOf<PersistentDetection>() }
        val fps = remember { mutableStateOf(0f) }
        val lastFrameTime = remember { mutableStateOf(System.currentTimeMillis()) }
        val frameCount = remember { mutableStateOf(0) }
        val textMeasurer = rememberTextMeasurer()
        val isBlackScreen = remember { mutableStateOf(false) }

        val isPortrait = configuration.orientation == Configuration.ORIENTATION_PORTRAIT
        val modelSize = 320f

        Box(modifier = Modifier.fillMaxSize()) {
            AndroidView(
                factory = { ctx ->
                    PreviewView(ctx).apply {
                        implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                        cameraProviderFuture.addListener({
                            val cameraProvider = cameraProviderFuture.get()
                            val preview = Preview.Builder()
                                .setTargetRotation(android.view.Surface.ROTATION_0)
                                .build()
                                .also {
                                    it.setSurfaceProvider(this.surfaceProvider)
                                }
                            val imageAnalysis = ImageAnalysis.Builder()
                                .setTargetResolution(android.util.Size(800, 1280))
                                .setTargetRotation(android.view.Surface.ROTATION_0)
                                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                                .build().also {
                                    it.setAnalyzer(executor) { image ->
                                        val currentTime = System.currentTimeMillis()
                                        frameCount.value++
                                        if (currentTime - lastFrameTime.value >= 1000) {
                                            fps.value = (frameCount.value * 1000f / (currentTime - lastFrameTime.value)).roundToInt().toFloat()
                                            frameCount.value = 0
                                            lastFrameTime.value = currentTime
                                        }

                                        if (!isPortrait) {
                                            Log.d(TAG, "Skipping detection: Device not in portrait mode")
                                            image.close()
                                            return@setAnalyzer
                                        }

                                        val isImageBlack = checkIfImageIsBlack(image)
                                        isBlackScreen.value = isImageBlack

                                        if (isImageBlack) {
                                            tts.stop()
                                            synchronized(persistentDetections) {
                                                persistentDetections.clear()
                                            }
                                            lastSpokenTime.clear()
                                            Log.d(TAG, "Black screen detected, TTS stopped and queue cleared")
                                            image.close()
                                            return@setAnalyzer
                                        }

                                        val rotationDegrees = image.imageInfo.rotationDegrees
                                        val imageWidth = image.width
                                        val imageHeight = image.height
                                        Log.d(TAG, "Image size: ${imageWidth}x${imageHeight}, Rotation: $rotationDegrees")

                                        val detected = yoloDetector.detectObjects(image)
                                        if (detected.isEmpty()) {
                                            Log.d(TAG, "No objects detected in this frame")
                                        }
                                        val filteredDetections = detected.groupBy { it.label }
                                            .mapValues { it.value.maxByOrNull { d -> d.confidence }!! }
                                            .values.map { detection ->
                                                PersistentDetection(
                                                    detection = detection,
                                                    timestamp = currentTime,
                                                    imageWidth = imageWidth,
                                                    imageHeight = imageHeight,
                                                    rotationDegrees = rotationDegrees
                                                )
                                            }

                                        synchronized(persistentDetections) {
                                            persistentDetections.removeAll { currentTime - it.timestamp > DETECTION_LIFETIME_MS }
                                            persistentDetections.clear()
                                            persistentDetections.addAll(filteredDetections)
                                        }

                                        filteredDetections.forEach { persistent ->
                                            val detection = persistent.detection
                                            val currentTimeMs = System.currentTimeMillis()
                                            val lastTime = lastSpokenTime[detection.label] ?: 0L
                                            if (currentTimeMs - lastTime >= SPEAK_COOLDOWN_MS) {
                                                val direction = calculateDirection(detection.x, detection.width, imageWidth.toFloat(), modelSize)
                                                val speakText = buildString {
                                                    append("${detection.label} detected")
                                                    if (direction.isNotEmpty()) append(" $direction")
                                                    if (detection.distance != null) {
                                                        append(" at ${String.format("%.1f", detection.distance)} meters")
                                                    }
                                                }
                                                Log.d(TAG, "Detected: ${detection.label} with confidence ${detection.confidence} at (${detection.x}, ${detection.y}, ${detection.width}, ${detection.height}), direction: $direction, distance: ${detection.distance}")
                                                tts.speak(speakText, TextToSpeech.QUEUE_ADD, null, null)
                                                lastSpokenTime[detection.label] = currentTimeMs
                                            }
                                        }
                                        image.close()
                                    }
                                }
                            cameraProvider.unbindAll()
                            cameraProvider.bindToLifecycle(
                                lifecycleOwner,
                                CameraSelector.DEFAULT_BACK_CAMERA,
                                preview,
                                imageAnalysis
                            )
                            onCameraBound()
                        }, ContextCompat.getMainExecutor(ctx))
                    }
                },
                modifier = Modifier.fillMaxSize()
            )

            Canvas(modifier = Modifier.fillMaxSize()) {
                val screenWidth = size.width
                val screenHeight = size.height

                drawRect(
                    color = Color.Green,
                    topLeft = Offset(50f, 50f),
                    size = Size(100f, 100f),
                    style = Stroke(width = 4f)
                )

                val fpsText = textMeasurer.measure(
                    text = "FPS: ${fps.value}",
                    style = TextStyle(color = Color.White, fontSize = 20.sp)
                )
                drawText(
                    textLayoutResult = fpsText,
                    topLeft = Offset(10f, screenHeight - fpsText.size.height - 10f)
                )

                synchronized(persistentDetections) {
                    persistentDetections.forEach { persistent ->
                        val detection = persistent.detection
                        val imageWidth = persistent.imageWidth.toFloat()
                        val imageHeight = persistent.imageHeight.toFloat()
                        val rotationDegrees = persistent.rotationDegrees

                        val scaleXModelToImage = imageWidth / modelSize
                        val scaleYModelToImage = imageHeight / modelSize
                        var imageX = detection.x * scaleXModelToImage
                        var imageY = detection.y * scaleYModelToImage
                        var imageWidthScaled = detection.width * scaleXModelToImage
                        var imageHeightScaled = detection.height * scaleYModelToImage

                        var rotatedX = imageX
                        var rotatedY = imageY
                        var rotatedWidth = imageWidthScaled
                        var rotatedHeight = imageHeightScaled

                        when (rotationDegrees) {
                            90 -> {
                                rotatedX = imageHeight - imageY - imageHeightScaled
                                rotatedY = imageX
                                rotatedWidth = imageHeightScaled
                                rotatedHeight = imageWidthScaled
                            }
                            270 -> {
                                rotatedX = imageY
                                rotatedY = imageWidth - imageX - imageWidthScaled
                                rotatedWidth = imageHeightScaled
                                rotatedHeight = imageWidthScaled
                            }
                            0 -> {}
                            180 -> {
                                rotatedX = imageWidth - imageX - imageWidthScaled
                                rotatedY = imageHeight - imageY - imageHeightScaled
                            }
                        }

                        val scaleXScreen = screenWidth / imageHeight
                        val scaleYScreen = screenHeight / imageWidth
                        val screenX = rotatedX * scaleXScreen
                        val screenY = rotatedY * scaleYScreen
                        val screenWidthScaled = rotatedWidth * scaleXScreen
                        val screenHeightScaled = rotatedHeight * scaleYScreen

                        val maxWidth = screenWidth - screenX
                        val maxHeight = screenHeight - screenY
                        val boundedX = screenX.coerceIn(0f, screenWidth)
                        val boundedY = screenY.coerceIn(0f, screenHeight)
                        val boundedWidth = screenWidthScaled.coerceIn(0f, maxWidth.coerceAtLeast(0f))
                        val boundedHeight = screenHeightScaled.coerceIn(0f, maxHeight.coerceAtLeast(0f))

                        Log.d(TAG, "Drawing box for ${detection.label}: ($boundedX, $boundedY, $boundedWidth, $boundedHeight)")

                        drawRect(
                            color = Color.Red,
                            topLeft = Offset(boundedX, boundedY),
                            size = Size(boundedWidth, boundedHeight),
                            style = Stroke(width = 4f)
                        )

                        val direction = calculateDirection(detection.x, detection.width, imageWidth, modelSize)
                        val labelText = buildString {
                            append(detection.label)
                            if (direction.isNotEmpty()) append(" ($direction)")
                            if (detection.distance != null) {
                                append(" ${String.format("%.1f", detection.distance)}m")
                            }
                        }
                        val labelTextLayout = textMeasurer.measure(
                            text = labelText,
                            style = TextStyle(color = Color.Yellow, fontSize = 16.sp)
                        )
                        val labelX = boundedX
                        val labelY = (boundedY - labelTextLayout.size.height).coerceAtLeast(0f)
                        drawText(
                            textLayoutResult = labelTextLayout,
                            topLeft = Offset(labelX, labelY)
                        )
                    }
                }
            }
        }
    }

    private fun checkIfImageIsBlack(image: ImageProxy): Boolean {
        val buffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)

        val totalLuminance = bytes.map { it.toInt() and 0xFF }.average()
        val threshold = 20.0

        return totalLuminance < threshold
    }
}