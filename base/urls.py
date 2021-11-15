from django.urls import path
from . import views

print('----base.urls----')
app_name = 'base'
urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),
]


