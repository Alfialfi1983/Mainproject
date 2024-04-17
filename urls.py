from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.log, name='log'),
    path('signup/', views.signup, name='signup'),
    path('signin/', views.signin, name='signin'),
    path('index/', views.index, name='index'),
    path('dehaze-video/', views.dehaze_video_view, name='dehaze_video'),
    path('dehaze/', views.dehaze_image, name='dehaze_image'),
    path('display/', views.display_videos, name='display_videos'),
    path('index/image/', views.image, name='image'),
    path('index/video/', views.video, name='video'),
    path('logout/', views.logout_view, name='logout'),


]
