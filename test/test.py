# Import the InferencePipeline object
from inference import InferencePipeline

# import VideoFrame for type hinting
from inference.core.interfaces.camera.entities import VideoFrame


import cv2
import easyocr
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)

def sink(predictions: dict, video_frame: VideoFrame):
    try:
        frame = video_frame.image

        if frame is None:
            print("Error: `video_frame.image` is None.")
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


                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
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

        # Display annotated frame
        cv2.imshow("Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

    except Exception as e:
        print(f"Error in sink function: {e}")



# initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key="sDa7pej7MbiBdSRowjPm",
    workspace_name="ccs4avillaflor",
    workflow_id="custom-workflow",
    video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    max_fps=30,
    on_prediction=sink
)
pipeline.start() #start the pipeline
pipeline.join() #wait for the pipeline thread to finish
