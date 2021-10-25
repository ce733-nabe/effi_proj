from django.urls import path
from . import views

app_name = 'effi_app'
urlpatterns = [
    path('index/', views.EffiappView.as_view(), name='index'),
]


