from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load pre-trained YOLOv5s model (modern ultralytics package)
model = YOLO('yolov5s.pt')  # downloads automatically if not present

# Initialize webcam (0 = default camera)
video_capture = cv2.VideoCapture(0)


def detect_objects(frame):
    """
    Detect objects in a frame using ultralytics YOLO.
    Draw bounding boxes and return object count.
    """

    results = model(frame)  # returns a list
    result = results[0]     # get the first Results object
    boxes = result.boxes    # now this works

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, len(boxes)


def gen_frames():
    """
    Generator function to yield frames for Flask streaming.
    """
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            frame, count = detect_objects(frame)
            cv2.putText(frame, f"Total Objects: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
