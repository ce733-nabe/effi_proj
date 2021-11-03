from django.urls import path
from . import views

app_name = 'effi_app'
urlpatterns = [
    path('index/', views.IndexView.as_view(), name='index'),
    path('single/', views.SingleView.as_view(), name='single'),
    path('multi/', views.MultiView.as_view(), name='multi'),
]


