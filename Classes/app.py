# from flask import Flask, Response
# import cv2

# app = Flask(__name__)

# # Initialize the camera (0 for the first camera, or provide the camera index)
# camera = cv2.VideoCapture(0)

# def generate_frames():
#     while True:
#         # Read the camera frame
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             # Encode the frame in JPEG format
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
            
#             # Yield the frame in byte format for streaming
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     # Stream the frames as a response
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000)
import cv2

# Initialize the video capture (0 is typically the default webcam)
cap = cv2.VideoCapture(1)

# Check if the video capture is opened successfully
if not cap.isOpened():
    print("Error: Unable to access the camera")
    exit()

# Set properties (optional, adjust as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height

print("Press 'q' to quit the video stream")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Display the frame
    cv2.imshow('Video Capture', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
