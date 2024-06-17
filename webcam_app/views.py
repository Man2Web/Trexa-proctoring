import cv2
import numpy as np
import face_recognition
import os
import datetime
import json
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import base64
import io
from PIL import Image

# Load pre-trained YOLO model for object detection
net = cv2.dnn.readNet(r"C:\Users\Ak\Desktop\NEW_TREXA\yolov3.weights", r"C:\Users\Ak\Desktop\NEW_TREXA\yolov3.cfg")  
classes = []
with open(r"C:\Users\Ak\Desktop\NEW_TREXA\coco.names", "r") as f:  
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()

# Load reference image and encode
reference_image = cv2.imread(r"C:\Users\Ak\Downloads\WhatsApp Image 2024-05-19 at 2.15.50 AM.jpeg")
reference_image_small = cv2.resize(reference_image, (0, 0), fx=0.25, fy=0.25)  # Reduce size for faster processing
reference_encoding = face_recognition.face_encodings(reference_image_small)[0]

# Function to detect objects (mobile phones) in an image
def detect_objects(frame):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    mobile_detected = False

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 67:  # Detect only mobile phones (class_id=67)
                mobile_detected = True

    return mobile_detected

# Function to perform face recognition
def recognize_face(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame_small = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
    face_locations = face_recognition.face_locations(rgb_frame_small)
    face_encodings = face_recognition.face_encodings(rgb_frame_small, face_locations)

    num_faces = len(face_locations)
    if num_faces >= 2:
        return "multiple_faces"

    if not face_locations:
        return "away from screen"
   
    

    face_matched = False
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces([reference_encoding], face_encoding)
        if matches[0]:
            face_matched = True

    return "matched_face" if face_matched else "unmatched_faces"

# Function to check for low light conditions
def check_low_light(frame, threshold=50):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness < threshold

def index(request):
    return render(request, 'webcam_app/index.html')

@csrf_exempt
def process_frame(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        data_url = data.get('image')
        if data_url:
            # Convert data URL to OpenCV image format
            image_data = base64.b64decode(data_url.split(',')[1])
            image = np.array(Image.open(io.BytesIO(image_data)))

            # Check for low light conditions
            low_light = check_low_light(image)

            # Detect objects (mobile phones)
            mobile_detected = detect_objects(image)

            # Perform face recognition
            face_category = recognize_face(image)

            if low_light:
                return JsonResponse({
                    "category": "low_light"
                })
            elif mobile_detected:
                return JsonResponse({
                    "category": "mobile_detected"
                })
            else:
                return JsonResponse({
                    "category": face_category
                })

        return JsonResponse({
            "error": "No image provided"
        }, status=400)

    return JsonResponse({
        "error": "Invalid request method"
    }, status=400)
