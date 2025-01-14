from django.urls import path
from . import views


urlpatterns = [
    path("", views.home, name='home'),
   path('video_feed/<int:camera_id>/', views.video_feed, name='video_feed'),
   path('video_feed/check_cams/', views.check_cams, name='check_cams'),
]