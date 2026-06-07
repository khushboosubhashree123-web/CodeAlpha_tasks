from flask import Flask, Response, render_template, request
import cv2
from ultralytics import YOLO
import time
import traceback

app = Flask(__name__)

model = None

def load_model():
    global model
    try:
        print("🔄 Downloading/Loading YOLOv8 nano model...")
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8 model loaded successfully!")
    except Exception as e:
        print("❌ Failed to load model:", e)
        print(traceback.format_exc())

# Load model
load_model()

def generate_frames(tracking=True):
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️ Webcam not accessible. Trying index 1...")
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                print("❌ No webcam found!")
                error_frame = cv2.putText(cv2.imread('black.jpg') if cv2.imread('black.jpg') is not None else 
                                        np.zeros((480, 640, 3), dtype=np.uint8),
                                        "Webcam Not Found", (100, 240), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_frame)
                while True:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    time.sleep(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame")
                break

            try:
                if model is not None:
                    if tracking:
                        results = model.track(frame, persist=True, conf=0.4, iou=0.5, verbose=False)
                    else:
                        results = model(frame, conf=0.4, verbose=False)
                    annotated_frame = results[0].plot()
                else:
                    annotated_frame = frame.copy()
                    cv2.putText(annotated_frame, "Model not loaded", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            except Exception as e:
                print("Detection error:", e)
                annotated_frame = frame.copy()

            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)

    except Exception as e:
        print("Error in generate_frames:", e)
        print(traceback.format_exc())
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Server Error", (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        ret, buffer = cv2.imencode('.jpg', error_img)
        while True:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    tracking = request.args.get('tracking', 'true').lower() == 'true'
    return Response(generate_frames(tracking=tracking),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🚀 Starting Object Detection Server...")
    print("🌐 Please open: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)   # debug=True helps see errors