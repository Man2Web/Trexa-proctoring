import cv2
import numpy as np
import face_recognition
import os
import datetime

# Create a directory for saving images
save_folder = "captured_images1"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Define sub-category names
sub_categories = ["unmatched_faces", "multiple_faces", "mobile_detected", "low_light"]

# Create subfolders for each category
for sub_category in sub_categories:
    sub_folder = os.path.join(save_folder, sub_category)
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

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

# Function to save images with timestamps
def save_image(image, category):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_folder, category, f"{timestamp}.jpg")
    cv2.imwrite(filename, image)

# Function to detect objects (mobile phones) in an image
def detect_objects(image):
    blob = cv2.dnn.blobFromImage(image, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    mobile_detected = False

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 67:  # Detect only mobile phones (class_id=67)
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                w = int(detection[2] * image.shape[1])
                h = int(detection[3] * image.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                mobile_detected = True
                print("Mobile detected")
                save_image(frame, "mobile_detected")

    return mobile_detected

# Function to perform face recognition
def recognize_face(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame_small = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
    face_locations = face_recognition.face_locations(rgb_frame_small)
    face_encodings = face_recognition.face_encodings(rgb_frame_small, face_locations)

    num_faces = len(face_locations)
    if num_faces >= 2:
        print("Multiple faces detected")
        save_image(frame, "multiple_faces")

    if not face_locations:
        print("Away from screen")
        save_image(frame, "unmatched_faces")
        return

    face_matched = False
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces([reference_encoding], face_encoding)
        if matches[0]:
            top, right, bottom, left = [coord * 4 for coord in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "Reference Person", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            face_matched = True
            print("Face matching")
        else:
            print("Face not matching")
            save_image(frame, "unmatched_faces")

    return face_matched

# Function to check for low light conditions
def check_low_light(frame, threshold=50):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < threshold:
        cv2.putText(frame, "Low Light Warning", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print("Low light detected")
        save_image(frame, "low_light")
        return True
    return False

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Check for low light conditions
    low_light = check_low_light(frame)

    # Detect objects (mobile phones)
    mobile_detected = detect_objects(frame)

    # Perform face recognition
    face_matched = recognize_face(frame)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
