from flask import Flask, render_template, request, redirect, url_for, send_from_directory

import cv2


app = Flask(__name__)


@app.route('/')
def home():

    available_cam = check_available_cameras(10)

    return f'Available Cameras {len(available_cam)}'



def check_available_cameras(max_cameras=10):
    """
    Check available video input devices (cameras).
    
    Parameters:
    - max_cameras (int): The maximum number of camera indices to check.
    
    Returns:
    - available_cameras (list): List of indices of available cameras.
    """
    available_cameras = []
    for camera_id in range(max_cameras):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(camera_id)
            cap.release()
    return available_cameras




if __name__ == '__main__':
    app.run(debug=True)