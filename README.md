 VisionAssist Navigator

VisionAssist Navigator is an Android application designed to assist visually impaired users by detecting objects in real time, estimating their distance, and identifying the direction of the object.  
The system uses a TensorFlow Lite (TFLite) model converted from YOLOv8, running fully on-device for fast and efficient processing.
 
 🔍 Real-Time Object Detection
- On-device detection using a YOLOv8 → TFLite model.
- Optimized for mobile performance.
- Detects common objects in road and indoor environments.
📏 Monocular Distance Estimation
Estimates approximate distance of detected objects using focal length, bounding box size, and real object width.

Formula Used: Distance (cm) = (Known Width × Focal Length) / Perceived Width

Where:
- *Known Width* = actual width of the object (predefined)
- *Perceived Width* = bounding box width in pixels
- *Focal Length* = calibrated value from sample images

 Direction Estimation
Determines whether an object is:
- Left
- Center
- Right

 Voice Assistance
Speaks out:
- Object detected
- Its direction
- Estimated distance


 Tech Stack

- Kotlin – Android development  
- TensorFlow Lite – On-device inference  
- YOLOv8 – Base object detection model  
- Custom Distance Estimation Algorithm  
- Text-To-Speech (TTS)  
- CameraX for real-time frames  



