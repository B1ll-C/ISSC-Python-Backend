import cv2

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

if __name__ == "__main__":
    max_cameras_to_check = 10  # Adjust this based on the expected number of cameras
    available_cameras = check_available_cameras(max_cameras=max_cameras_to_check)
    
    if available_cameras:
        print(f"Available cameras: {available_cameras}")
    else:
        print("No cameras found.")
