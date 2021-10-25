from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views.generic import TemplateView
from .forms import PhotoImageForm
from effi_app.main import pred
from .models import PhotoImage
from django.conf import settings

class EffiappView(TemplateView):

    def __init__(self):
        self.params={'pred': 'none',
                    'form': PhotoImageForm(),
                    'pred_image':'none'
                    }

    def get(self, req):
        return render(req, 'effi_app/index.html', self.params)

    def post(self, req):
        form = PhotoImageForm(req.POST, req.FILES)
        if form.is_valid():
            
            photoimage = PhotoImage()
            photoimage.photo_image = form.cleaned_data['photo_image']
            photoimage.save()

            print('----------------------')
            print('photoimage.photo_image:{}'.format(settings.MEDIA_ROOT + '/' + str(photoimage.photo_image)))

            self.params['pred'] = pred(settings.MEDIA_ROOT + '/' + str(photoimage.photo_image))
            self.params['pred_image'] = PhotoImage.objects.all().order_by('-id')[:1]
            
    
        return render(req, 'effi_app/index.html', self.params)

    
    