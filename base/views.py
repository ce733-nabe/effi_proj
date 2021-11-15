from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views.generic import TemplateView
from django.conf import settings
from django.contrib.auth.mixins import LoginRequiredMixin

class IndexView(TemplateView):
    template_name = 'base/index.html'
    #login_url = '/account/login/'
