import threading
import cv2
import numpy as np
import os
from django.http import StreamingHttpResponse,HttpResponse
from io import BytesIO
from django.shortcuts import render
from inference_sdk import InferenceHTTPClient
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import time
import easyocr

# Set up the Roboflow Inference client
# CLIENT = InferenceHTTPClient(
#     api_url="https://infer.roboflow.com",
#     api_key="sDa7pej7MbiBdSRowjPm"
# )

# Dictionary to store pipelines and frames for multiple cameras
camera_pipelines = {}
camera_frames = {}

reader = easyocr.Reader(['en'], gpu=False)

def num_of_cams():
    return 1
def check_cams(request):
    return HttpResponse(num_of_cams())
def sink(predictions: dict, video_frame, camera_id):
    global camera_frames
    try:
        frame = video_frame.image
        if frame is None:
            print(f"Error: `video_frame.image` is None for camera {camera_id}.")
            return

        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        if 'model_predictions' in predictions:
            human_detections = predictions['model_predictions']
            for bbox, confidence, class_name in zip(
                    human_detections.xyxy,
                    human_detections.confidence,
                    human_detections.data['class_name']
            ):
                custom_class_name = "Person"
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                label = f"{custom_class_name} ({confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if 'model_1_predictions' in predictions:
            plate_detections = predictions['model_1_predictions']

            for bbox, confidence, class_name in zip(
                plate_detections.xyxy,
                plate_detections.confidence,
                plate_detections.data['class_name']
            ):
                x1, y1, x2, y2 = map(int, bbox)


                height, width, _ = frame.shape
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)

                if x1 == x2 or y1 == y2:
                    continue

                cropped_image = frame[y1:y2, x1:x2]

                gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                denoised = cv2.fastNlMeansDenoising(thresholded, None, 30, 7, 21)

                result = reader.readtext(denoised)


                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                if result:
                    for detection in result:
                        _, text, confidence = detection

                        ocr_label = f"Plate : {text} ({confidence:.2f})"

                        text_size = cv2.getTextSize(ocr_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                        text_w, text_h = text_size
                        cv2.rectangle(frame, (x1, y2 + 10), (x1 + text_w, y2 + 10 + text_h), (0, 0, 255), -1)


                        cv2.putText(frame, ocr_label, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Update the latest frame for this camera_id
        camera_frames[camera_id] = frame

    except Exception as e:
        print(f"Error in sink function for camera {camera_id}: {e}")

# Start the inference pipeline for a specific camera_id
def start_inference_pipeline(camera_id, video_reference):
    global camera_pipelines
    if camera_id not in camera_pipelines:
        pipeline = InferencePipeline.init_with_workflow(
            api_key="sDa7pej7MbiBdSRowjPm",
            workspace_name="ccs4avillaflor",
            workflow_id="custom-workflow",
            video_reference=video_reference,  # Path to video or camera ID (int)
            max_fps=30,
            on_prediction=lambda predictions, video_frame: sink(predictions, video_frame, camera_id)
        )
        pipeline.start()
        camera_pipelines[camera_id] = pipeline

# Start the inference pipeline on a background thread
def start_inference_on_startup(camera_id, video_reference):
    threading.Thread(target=start_inference_pipeline, args=(camera_id, video_reference), daemon=True).start()

# Stream the video in response to a request
def stream_video(request, camera_id):
    def gen():
        while camera_id not in camera_frames or camera_frames[camera_id] is None:
            time.sleep(1)  # Wait until a frame is ready

        while True:
            if camera_id in camera_frames and camera_frames[camera_id] is not None:
                frame = camera_frames[camera_id]
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    byte_io = BytesIO()
                    byte_io.write(jpeg.tobytes())
                    byte_io.seek(0)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + byte_io.read() + b'\r\n\r\n')

    return StreamingHttpResponse(gen(), content_type='multipart/x-mixed-replace; boundary=frame')
