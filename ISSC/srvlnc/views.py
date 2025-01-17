from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse
from threading import Thread
from queue import Queue
import cv2
import numpy as np  # For creating the "no signal" frame
import supervision as sv
from inference import get_model

from dotenv import load_dotenv
import os
from pathlib import Path



env_path = Path(__file__).resolve().parent.parent.parent / ".env"

load_dotenv(dotenv_path=env_path)
api_key = os.getenv("API_KEY")
model_id = os.getenv("MODEL_ID")

# Initialize the YOLO model
model = get_model(model_id=model_id, api_key=api_key)

# Initialize cameras
# cameras = {
#     0: cv2.VideoCapture(0),  # Camera 0
#     1: cv2.VideoCapture(1),  # Camera 1
#     2: cv2.VideoCapture(1),  # Camera 1
#     3: cv2.VideoCapture(1),  # Camera 1
#     4: cv2.VideoCapture(1),  # Camera 1
#     5: cv2.VideoCapture(1),  # Camera 1
# }

cameras = {i : cv2.VideoCapture(i) for i in range(4)}
# Create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Frame queues for each camera
frame_queues = {camera_id: Queue(maxsize=10) for camera_id in cameras}

# Global variable to signal thread shutdown
running = True

def generate_no_signal_frame():
    """Generate a visually appealing 'No Signal' frame."""
    # Create a gradient background (480x640)
    width, height = 640, 480
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Gradient colors (blue to black)
    for y in range(height):
        color = (255 - int(255 * y / height), 0, 128)  # Purple gradient
        frame[y, :] = color

    # Add the "No Signal" text
    cv2.putText(frame, "NO SIGNAL", (130, 240), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)

    # Add a warning/camera icon (triangle with exclamation or circle with a line)
    center = (320, 320)  # Icon position
    cv2.circle(frame, center, 50, (0, 0, 255), -1)  # Red circle
    cv2.putText(frame, "!", (305, 345), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)

    return frame


def background_inference(camera_id):
    """Background thread for running YOLO inference on a specific camera."""
    global running
    camera = cameras[camera_id]
    frame_queue = frame_queues[camera_id]

    while running:
        if not camera.isOpened():
            # If the camera is not accessible, send a 'No Signal' frame
            no_signal_frame = generate_no_signal_frame()
            if frame_queue.full():
                frame_queue.get()
            frame_queue.put(no_signal_frame)
            continue

        ret, frame = camera.read()
        if not ret:
            # If frame cannot be read, send a 'No Signal' frame
            no_signal_frame = generate_no_signal_frame()
            if frame_queue.full():
                frame_queue.get()
            frame_queue.put(no_signal_frame)
            continue

        # Run inference on the frame
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)

        # Annotate the frame with bounding boxes and labels
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # Add the annotated frame to the queue
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(annotated_frame)

        # INSERT CODE FOR EMAIL
        if len(detections) > 0:
            print(f"Camera {camera_id}: {len(detections)} objects detected.")

def generate_frames(camera_id):
    """Generate frames for the video stream of a specific camera."""
    frame_queue = frame_queues[camera_id]
    while running:
        if not frame_queue.empty():
            frame = frame_queue.get()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def start_inference_threads():
    """Start the background inference threads for all cameras."""
    for camera_id in cameras:
        thread = Thread(target=background_inference, args=(camera_id,), daemon=True)
        thread.start()

# Start the background inference threads
start_inference_threads()

def home(request):
    """Render the home page."""
    return render(request, 'index.html')

def check_cams(request):
    """Check the number of available cameras."""
    return HttpResponse(len(cameras))

def video_feed(request, camera_id):
    """Stream the video feed with annotations for a specific camera."""
    camera_id = int(camera_id)  # Convert camera_id from string to integer
    if camera_id not in cameras:
        return StreamingHttpResponse(
            f"Camera {camera_id} not found", status=404
        )

    return StreamingHttpResponse(
        generate_frames(camera_id),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

def shutdown(request):
    """Stop the background inference."""
    global running
    running = False
    for camera in cameras.values():
        camera.release()
    cv2.destroyAllWindows()
    return HttpResponse("Inference stopped. Cameras released.")
