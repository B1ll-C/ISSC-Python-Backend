# **ISSC Python Backend Setup**

## Prerequisites

Before running the server, ensure you have the following:

1. **Install Dependencies**: Make sure to install the necessary dependencies listed in `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

2. **API Keys Configuration**: Create an `.env` file in the root directory of your project to store your API keys for the models.

    Example `.env` file:

    ```makefile
    API_KEY=your_roboflow_api_here
    HUMAN_MODEL=issc-human-detection/1
    PLATE_MODEL=issc-plate-recognition/1
    ```

    Replace `your_roboflow_api_here` with your actual Roboflow API key.

## Running the Server

To start the server, use the following command:

```bash
python manage.py runserver localhost:5000
```

### File Structure
```sql
- srvlnc/
    - migrations/
    - models.py
    - views.py           <-- Contains the background inference and streaming logic
    - apps.py            <-- Application configuration for starting threads on app load
    - urls.py            <-- URLs for video feeds, checking camera availability, etc.
    - settings.py        <-- Django settings
    - templates/
        - index.html     <-- Main HTML template for displaying video feeds
    - static/

```
# Background Inference and Video Streaming
## 1. Background Inference (background_inference function)

#### Purpose:
This function runs in the background and processes video frames captured from a camera. It performs inference using YOLO models to detect people and license plates, and then annotates the frames with bounding boxes and labels.

#### Function Flow:
- **Camera Initialization**: The function accesses a specific camera based on `camera_id`. The camera object (`cv2.VideoCapture`) is retrieved from the cameras dictionary.
- **Frame Queue**: A queue is used to store frames for each camera. This ensures frames are processed in a controlled manner and that the latest frames are available for streaming.
- **Camera Check**: If the camera is not opened or there is an issue reading the frame, a special "No Signal" frame is generated (using `generate_no_signal_frame()`), and it’s placed in the queue for that camera. The frame is displayed if there’s no feed from the camera.
- **Frame Capture**: Once a valid frame is captured, the inference is performed on it using two YOLO models:
  - `model_person.infer(frame)`: Detects people in the frame.
  - `model_plate.infer(frame)`: Detects license plates in the frame.
- **Detection Results**: The results from the YOLO models are converted into detections using `sv.Detections.from_inference()`.
- **Frame Annotation**: The `bounding_box_annotator` is used to draw bounding boxes around detected objects (people and plates). Additionally, the `label_annotator` is used to label the detected objects on the frame.
- **OCR (Optical Character Recognition) for Plates**: If a license plate is detected, a region of interest (ROI) is cropped around the plate, and OCR is performed on the cropped plate image to extract the plate number.
  - **Denoising**: The cropped license plate image is converted to grayscale and denoised using `cv2.fastNlMeansDenoising()` to improve OCR accuracy.
  - **OCR**: `easyocr.Reader.readtext()` is used to detect and extract text from the denoised plate image.
  - **OCR Confidence**: OCR results are printed if the confidence level is above a certain threshold (in this case, 0.1).
- **Frame Queue**: Once the frame is annotated, it’s added to the queue. If the queue is full, the oldest frame is removed to make room for the new one.
- **Debugging**: The number of detected persons and plates is printed to the console for debugging purposes.

---

## 2. Frame Generation (generate_frames function)

#### Purpose:
This function generates frames for streaming to the client in real-time. It continuously retrieves annotated frames from the frame queue and sends them as JPEG images over a multipart HTTP response.

#### Function Flow:
- **Frame Retrieval**: The function checks if there are any frames available in the camera's frame queue (`frame_queues[camera_id]`). If there are, it retrieves the latest frame.
- **JPEG Encoding**: The frame is encoded to the JPEG format using OpenCV’s `cv2.imencode('.jpg', frame)` function. This converts the frame into a byte stream.
- **Streaming Response**: The encoded frame is returned as part of a multipart HTTP response, allowing the frame to be displayed as a continuous video stream in the browser.
- **MIME Boundary**: The multipart format (`multipart/x-mixed-replace`) is used to send continuous video frames, where each frame is separated by a boundary (`--frame`).

---

## 3. No Signal Frame (generate_no_signal_frame function)

#### Purpose:
Generates a special "No Signal" frame when the camera feed is not available or if the camera is not working.

#### Function Flow:
- The frame is created as a purple gradient and displays the text "NO SIGNAL" with a red circle and an exclamation mark to indicate that no video feed is available.



## 4. Start Inference Threads (start_inference_threads function)

#### Purpose:
Starts background inference threads for each connected camera, allowing multiple cameras to be processed concurrently without blocking the main program.

#### Function Flow:
- The function iterates over the list of cameras and creates a separate thread for each camera.
- Each thread runs the `background_inference` function, which processes video frames in the background.
- Threads are started as **daemon threads**, meaning they automatically terminate when the main program exits.

---



# How These Functions Work Together

- **Start Inference Threads**: This function initializes background inference tasks for all cameras by spawning separate threads. It ensures that each camera can process its own video feed independently and simultaneously.
- **Background Inference**: Each thread runs this function, which performs real-time processing of frames, including object detection and OCR for license plates.
- **Concurrency**: With the inference running in the background, the main application can continue performing other tasks, such as streaming annotated video frames or handling client requests, without being slowed down by frame processing.
- **Streaming Video**: The annotated frames are then made available for streaming to a client (e.g., a web browser) through the `generate_frames` function, which continuously sends frames to the client.
- **Real-Time Updates**: This setup allows the application to display real-time video feeds with annotations and OCR results while processing the frames in the background.


---

# Key Technologies Used:

- **OpenCV**: For handling video capture and image processing (frame capture, annotation, and OCR).
- **YOLO Models**: For detecting people and license plates.
- **Supervision (sv)**: For bounding box and label annotations on detected objects.
- **EasyOCR**: For optical character recognition (OCR) on detected license plates.
- **Multithreading**: To run background inference for multiple cameras simultaneously without blocking the main thread.
- **HTTP Streaming**: For delivering live video streams to the client with annotated frames.

---

# Summary
This setup enables real-time monitoring and detection of persons and license plates with optical character recognition, making it useful for applications such as surveillance and vehicle monitoring.
