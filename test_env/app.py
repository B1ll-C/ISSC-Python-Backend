# Import the InferencePipeline object
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import cv2

from dotenv import load_dotenv
import os
from pathlib import Path



env_path = Path(__file__).resolve().parent.parent / ".env"

load_dotenv(dotenv_path=env_path)
gemini_api_key = os.getenv("GEMINI_API")
robo_api_key = os.getenv("ROBO_API")



def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    print(predictions)
   
    

# initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key=robo_api_key,
    workspace_name="issc-tvfrz",
    workflow_id="custom-workflow",
    video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    max_fps=30,
    on_prediction=my_custom_sink,
    workflows_parameters={
        "gemini_ocr__api_key": gemini_api_key
    }
)
pipeline.start() #start the pipeline
pipeline.join() #wait for the pipeline thread to finish
