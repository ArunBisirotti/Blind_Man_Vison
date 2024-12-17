from flask import Flask, render_template, Response
import cv2
import torch
import pyttsx3
import threading

app = Flask(__name__)

# Initialize YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Lock for TTS engine to ensure thread safety
tts_lock = threading.Lock()

def speak_safe(text):
    """Thread-safe TTS function with per-instance initialization."""
    def tts_thread():
        with tts_lock:
            try:
                # Reinitialize the engine for each call to avoid blocking issues
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")

    # Create a new thread for each speech output
    threading.Thread(target=tts_thread, daemon=True).start()


# Video stream generator
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for YOLOv5 input
        frame_resized = cv2.resize(frame, (640, 640))

        # Perform object detection
        results = model(frame_resized)
        detections = results.pandas().xyxy[0].to_dict(orient="records")

        # Process detections
        for detection in detections:
            label = detection['name']
            confidence = detection['confidence']
            xmin, ymin, xmax, ymax = map(int, [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Announce detection (thread-safe)
            speak_safe(f"Detected {label} with confidence {confidence:.2f}")

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
