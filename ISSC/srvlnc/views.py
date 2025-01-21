from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse
from threading import Thread
from queue import Queue
import cv2
import numpy as np
import supervision as sv
from inference import get_model
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("API_KEY")
model_id_person = os.getenv("HUMAN_MODEL")
model_id_plate = os.getenv("PLATE_MODEL")

# Initialize the YOLO models
model_person = get_model(model_id=model_id_person, api_key=api_key)
model_plate = get_model(model_id_plate, api_key=api_key)

# Initialize cameras
cameras = {i: cv2.VideoCapture(i) for i in range(4)}

# Create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Frame queues for each camera
frame_queues = {camera_id: Queue(maxsize=10) for camera_id in cameras}

# Global variable to signal thread shutdown
running = True

def generate_no_signal_frame():
    """Generate a visually appealing 'No Signal' frame."""
    width, height = 640, 480
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        color = (255 - int(255 * y / height), 0, 128)  # Purple gradient
        frame[y, :] = color
    cv2.putText(frame, "NO SIGNAL", (130, 240), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)
    center = (320, 320)
    cv2.circle(frame, center, 50, (0, 0, 255), -1)
    cv2.putText(frame, "!", (305, 345), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)
    return frame

def background_inference(camera_id):
    """Background thread for running YOLO inference on a specific camera."""
    global running
    camera = cameras[camera_id]
    frame_queue = frame_queues[camera_id]

    while running:
        if not camera.isOpened():
            no_signal_frame = generate_no_signal_frame()
            if frame_queue.full():
                frame_queue.get()
            frame_queue.put(no_signal_frame)
            continue

        ret, frame = camera.read()
        if not ret:
            no_signal_frame = generate_no_signal_frame()
            if frame_queue.full():
                frame_queue.get()
            frame_queue.put(no_signal_frame)
            continue

        # Run inference on the frame using both models
        results_person = model_person.infer(frame)[0]
        results_plate = model_plate.infer(frame)[0]

        # Convert results to detections
        detections_person = sv.Detections.from_inference(results_person)
        detections_plate = sv.Detections.from_inference(results_plate)

        # Annotate the frame for person detection
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections_person)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections_person)

        # Annotate the frame for plate detection
        annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections_plate)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections_plate)

        # Add the annotated frame to the queue
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(annotated_frame)

        # Print detection info for debugging
        if len(detections_person) > 0:
            print(f"Camera {camera_id}: {len(detections_person)} person(s) detected.")
        if len(detections_plate) > 0:
            print(f"Camera {camera_id}: {len(detections_plate)} plate(s) detected.")

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
    camera_id = int(camera_id)
    if camera_id not in cameras:
        return StreamingHttpResponse(f"Camera {camera_id} not found", status=404)

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
