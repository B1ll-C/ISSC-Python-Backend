from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
import cv2
# Create your views here.

def home(request):
    return render(request, 'index.html')


cameras = {
    0: cv2.VideoCapture(0),  # Camera 0
    1: cv2.VideoCapture(1),  # Camera 1
}

def check_cams(request):
    return HttpResponse(len(cameras))

# Function to generate frames for a specific camera
def generate_frames(camera_id):
    camera = cameras[camera_id]
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in byte format for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# View function for video feed of a specific camera
def video_feed(request, camera_id):
    camera_id = int(camera_id)  # Convert camera_id from string to integer
    if camera_id not in cameras:
        return StreamingHttpResponse(
            f"Camera {camera_id} not found", status=404
        )

    return StreamingHttpResponse(
        generate_frames(camera_id),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )