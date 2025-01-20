from django.apps import AppConfig
import threading

class SrvlncConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'srvlnc'
    
    def ready(self):
        from .views import start_inference_on_startup,num_of_cams
        # Example to start inference for multiple cameras
        for i in range(num_of_cams()):
            start_inference_on_startup(camera_id=i, video_reference=i)  # Camera 1
        # start_inference_on_startup(camera_id=2, video_reference=1)  # Camera 2

