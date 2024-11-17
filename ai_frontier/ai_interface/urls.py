


from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('prompt/', views.prompt_interface, name='prompt_interface'),
]