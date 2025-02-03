---
title: Chaplin - Live Visual Speech Recognition
emoji: 🎬
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# Chaplin - Live Visual Speech Recognition

This Gradio app provides real-time visual speech recognition using your webcam. Simply allow camera access and start speaking - the model will attempt to read your lips and convert the movements to text.

## Features
- Real-time webcam processing
- Lip movement detection and tracking
- Text prediction from visual speech

## Usage
1. Allow camera access when prompted
2. Position yourself so your face is clearly visible
3. Speak naturally while facing the camera
4. View the predicted text in real-time

## Technical Details
- Uses MediaPipe for face detection
- Processes frames at 16 FPS
- Includes frame compression for optimal performance
