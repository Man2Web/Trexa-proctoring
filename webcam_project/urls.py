from django.contrib import admin
from django.urls import path
from webcam_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('process_frame/', views.process_frame, name='process_frame'),
]
