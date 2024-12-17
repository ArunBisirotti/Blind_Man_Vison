import cv2
import torch
import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

def speak(text):
    """Thread-safe Text-to-Speech function."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    engine.say(text)
    engine.runAndWait()

# Load YOLOv5 model (optimized for accuracy)
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# Select the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Use half precision for GPU (if available)
if device.type == 'cuda':
    model = model.half()

# Camera Parameters
focal_length = 800
real_object_width = 0.5

# Access the laptop camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise Exception("Could not open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame for YOLOv5 input
    img_resized = cv2.resize(frame, (1280, 1280))

    # Perform inference with higher confidence threshold and NMS
    results = model(img_resized)
    detections = results.pandas().xyxy[0]
    detections = detections[detections['confidence'] > 0.6]  # Apply confidence threshold

    for _, row in detections.iterrows():
        # Extract bounding box and label information
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        label = row['name']

        # Calculate object width in pixels
        perceived_width = x2 - x1

        # Calculate distance using the focal length formula
        if perceived_width > 0:
            distance = (real_object_width * focal_length) / perceived_width
            distance_text = f"{label}: {distance:.2f} meters"
            cv2.putText(frame, distance_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            speak(f"{label} detected at {distance:.2f} meters")

        # Draw bounding boxes and labels
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Object Detection', frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
