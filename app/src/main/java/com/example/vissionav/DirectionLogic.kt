package com.example.vissionav

fun calculateDirection(x: Float, width: Float, imageWidth: Float, modelSize: Float): String {
    val scaledX = x * (imageWidth / modelSize)
    val scaledWidth = width * (imageWidth / modelSize)
    val leftThreshold = imageWidth * 0.2f
    val rightThreshold = imageWidth * 0.8f
    val objectLeft = scaledX
    val objectRight = scaledX + scaledWidth

    return when {

        objectRight <= leftThreshold -> "on the left"

        objectLeft >= rightThreshold -> "on the right"

        else -> ""
    }
}