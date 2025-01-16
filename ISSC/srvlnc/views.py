from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse
from threading import Thread
from queue import Queue
import cv2
import supervision as sv
from inference import get_model

from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path('G:/Freelance/ISSC-Python-Backend') / '.env'
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("API_KEY")
model_id = os.getenv("MODEL_ID")

# Initialize the YOLO model
model = get_model(model_id=model_id, api_key=api_key)

# Initialize cameras
cameras = {
    0: cv2.VideoCapture(0),  # Camera 0
    1: cv2.VideoCapture(1),  # Camera 1
}

# Create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Frame queues for each camera
frame_queues = {camera_id: Queue(maxsize=10) for camera_id in cameras}

# Global variable to signal thread shutdown
running = True

def background_inference(camera_id):
    """Background thread for running YOLO inference on a specific camera."""
    global running
    camera = cameras[camera_id]
    frame_queue = frame_queues[camera_id]

    while running:
        ret, frame = camera.read()
        if not ret:
            continue

        # Run inference on the frame
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)

        # Annotate the frame with bounding boxes and labels
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # Add the annotated frame to the queue
        if frame_queue.full():
            frame_queue.get()  # Remove the oldest frame if the queue is full
        frame_queue.put(annotated_frame)

        ###### INSERT CODE FOR EMAIL NOTIFICATION
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
