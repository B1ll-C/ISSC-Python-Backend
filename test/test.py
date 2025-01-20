# Import the InferencePipeline object
from inference import InferencePipeline
# import VideoFrame for type hinting
from inference.core.interfaces.camera.entities import VideoFrame

import cv2
import numpy as np
import os
from inference_sdk import InferenceHTTPClient

# Set up the Roboflow Inference client
CLIENT = InferenceHTTPClient(
    api_url="https://infer.roboflow.com",
    api_key= "sDa7pej7MbiBdSRowjPm"
)

def sink(predictions: dict, video_frame):
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

                custom_plate_class_name = "License Plate"
                x1, y1, x2, y2 = map(int, bbox)


                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


                label = f"{custom_plate_class_name} ({confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                cropped_img = frame[y1:y2, x1:x2]

                temp_cropped_image_path = "temp_cropped_image.jpg"
                cv2.imwrite(temp_cropped_image_path, cropped_img)

                result = CLIENT.ocr_image(inference_input=temp_cropped_image_path)


                if result and 'predictions' in result:
                    ocr_text = result['predictions']
                    if ocr_text:
                        ocr_label = f"Text: {ocr_text[0]['text']}"
                        print(f"OCR Text: {ocr_text[0]['text']}")
                        cv2.putText(frame, ocr_label, (x1, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


                os.remove(temp_cropped_image_path)


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
