package com.example.vissionav

import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.sqrt

class FaceRecognizer(private val context: Context) {
    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .build()
    )
    private val storedLandmarks = ConcurrentHashMap<String, Map<String, Float>>()

    companion object {
        private const val TAG = "FaceRecognizer"
        private const val LANDMARK_THRESHOLD = 5.0f // Lowered for stricter matching
    }

    fun loadFaceLandmarks(folderImages: Map<String, MutableList<String>>) {
        storedLandmarks.clear()
        Log.d(TAG, "Loading face landmarks from folders: ${folderImages.keys}")
        folderImages.forEach { (name, paths) ->
            if (paths.isEmpty()) {
                Log.w(TAG, "No images found for: $name")
                return@forEach
            }
            paths.take(1).forEach { path ->
                val file = File(path)
                if (!file.exists()) {
                    Log.e(TAG, "Image file does not exist: $path")
                    return@forEach
                }
                extractFacialLandmarks(file) { landmarks ->
                    if (landmarks.size >= 8) {
                        storedLandmarks[name] = landmarks
                        Log.d(TAG, "Loaded landmarks for $name from $path: $landmarks")
                    } else {
                        Log.w(TAG, "Insufficient landmarks extracted for $name from $path: $landmarks")
                    }
                }
            }
        }
        Log.d(TAG, "Stored landmarks: $storedLandmarks")
    }

    private fun extractFacialLandmarks(imageFile: File, callback: (Map<String, Float>) -> Unit) {
        try {
            val image = InputImage.fromFilePath(context, Uri.fromFile(imageFile))
            detector.process(image)
                .addOnSuccessListener { faces ->
                    if (faces.isEmpty()) {
                        Log.e(TAG, "No faces detected in $imageFile")
                        callback(emptyMap())
                        return@addOnSuccessListener
                    }
                    val face = faces[0]
                    val landmarks = extractNormalizedLandmarks(face, image.width, image.height)
                    if (landmarks.size >= 8) {
                        Log.d(TAG, "Extracted normalized landmarks from $imageFile: $landmarks")
                    } else {
                        Log.w(TAG, "Insufficient landmarks extracted from $imageFile: ${landmarks.size}")
                    }
                    callback(landmarks)
                }
                .addOnFailureListener { e ->
                    Log.e(TAG, "Face detection failed for $imageFile: ${e.message}")
                    callback(emptyMap())
                }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading image $imageFile: ${e.message}")
            callback(emptyMap())
        }
    }

    fun detectFaces(image: ImageProxy, callback: (List<Face>) -> Unit) {
        try {
            val inputImage = InputImage.fromBitmap(image.toBitmap(), image.imageInfo.rotationDegrees)
            detector.process(inputImage)
                .addOnSuccessListener { faces ->
                    Log.d(TAG, "Detected ${faces.size} faces in frame")
                    callback(faces)
                }
                .addOnFailureListener { e ->
                    Log.e(TAG, "Face detection failed: ${e.message}")
                    callback(emptyList())
                }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing ImageProxy: ${e.message}")
            callback(emptyList())
        }
    }

    fun recognizeFace(face: Face, imageWidth: Int, imageHeight: Int): String? {
        val landmarks = extractNormalizedLandmarks(face, imageWidth, imageHeight)
        if (landmarks.size < 8) {
            Log.w(TAG, "Insufficient landmarks extracted from detected face: ${landmarks.size}")
            return null
        }

        Log.d(TAG, "Comparing face landmarks: $landmarks")
        for ((name, stored) in storedLandmarks) {
            if (compareFacialLandmarks(landmarks, stored, name)) {
                Log.d(TAG, "Face recognized as $name")
                return name
            }
        }
        Log.d(TAG, "No match found for face")
        return null
    }

    private fun extractNormalizedLandmarks(face: Face, imageWidth: Int, imageHeight: Int): Map<String, Float> {
        val landmarks = mutableMapOf<String, Float>()
        val bbox = face.boundingBox
        val faceWidth = bbox.width().toFloat()
        val faceHeight = bbox.height().toFloat()
        if (faceWidth <= 0 || faceHeight <= 0) {
            Log.w(TAG, "Invalid face bounding box: $bbox")
            return emptyMap()
        }

        fun normalizeX(x: Float) = (x - bbox.left) / faceWidth
        fun normalizeY(y: Float) = (y - bbox.top) / faceHeight

        face.getLandmark(FaceLandmark.LEFT_EYE)?.position?.let {
            landmarks["leftEyeX"] = normalizeX(it.x)
            landmarks["leftEyeY"] = normalizeY(it.y)
        }
        face.getLandmark(FaceLandmark.RIGHT_EYE)?.position?.let {
            landmarks["rightEyeX"] = normalizeX(it.x)
            landmarks["rightEyeY"] = normalizeY(it.y)
        }
        face.getLandmark(FaceLandmark.NOSE_BASE)?.position?.let {
            landmarks["noseBaseX"] = normalizeX(it.x)
            landmarks["noseBaseY"] = normalizeY(it.y)
        }
        face.getLandmark(FaceLandmark.MOUTH_LEFT)?.position?.let {
            landmarks["leftMouthX"] = normalizeX(it.x)
            landmarks["leftMouthY"] = normalizeY(it.y)
        }
        face.getLandmark(FaceLandmark.MOUTH_RIGHT)?.position?.let {
            landmarks["rightMouthX"] = normalizeX(it.x)
            landmarks["rightMouthY"] = normalizeY(it.y)
        }

        return landmarks
    }

    private fun compareFacialLandmarks(landmarks1: Map<String, Float>, landmarks2: Map<String, Float>, name: String): Boolean {
        if (landmarks1.size < 8 || landmarks2.size < 8) {
            Log.w(TAG, "Insufficient landmarks for $name: l1=${landmarks1.size}, l2=${landmarks2.size}")
            return false
        }

        val keys = listOf(
            "leftEyeX", "leftEyeY", "rightEyeX", "rightEyeY",
            "noseBaseX", "noseBaseY", "leftMouthX", "leftMouthY",
            "rightMouthX", "rightMouthY"
        )
        var totalDistance = 0.0f
        var count = 0

        for (key in keys) {
            if (landmarks1.containsKey(key) && landmarks2.containsKey(key)) {
                val value1 = landmarks1[key]!!
                val value2 = landmarks2[key]!!
                val distance = sqrt((value1 - value2) * (value1 - value2))
                totalDistance += distance
                count++
                Log.d(TAG, "Distance for $name, $key: $distance")
            }
        }

        if (count < 8) {
            Log.w(TAG, "Insufficient landmarks for $name: count=$count")
            return false
        }

        val averageDistance = totalDistance / count
        Log.d(TAG, "Average normalized landmark distance for $name: $averageDistance")
        return averageDistance < LANDMARK_THRESHOLD
    }

    fun close() {
        detector.close()
        storedLandmarks.clear()
    }
}