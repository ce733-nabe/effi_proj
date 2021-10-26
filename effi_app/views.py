from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views.generic import TemplateView
from .forms import SingleForm ,MultiForm
from effi_app.main import pred
from .models import EfficientData
from django.conf import settings

class SingleView(TemplateView):

    def __init__(self):
        self.params={'effidata': 'none',
                    'form': SingleForm(),
                    }

    def get(self, req):
        return render(req, 'effi_app/single.html', self.params)

    def post(self, req):
        if req.method == 'POST':
            form = SingleForm(req.POST, req.FILES)

            if form.is_valid():
                effidata = EfficientData()

                effidata.photo_image = form.cleaned_data['photo_image']
                effidata.save()

                effidata.pred_result =pred(settings.MEDIA_ROOT + '/' + str(effidata.photo_image))
                effidata.save()
                
                self.params['effidata'] = EfficientData.objects.all().order_by('-id')[:1]
        else:
            form = SingleForm()

        return render(req, 'effi_app/single.html', self.params)


class MultiView(TemplateView):

    def __init__(self):
        self.params={'effidata': 'none',
                    'form': MultiForm(),
                    }

    def get(self, req):
        return render(req, 'effi_app/multi.html', self.params)

    def post(self, req):
        if req.method == 'POST':
            form = MultiForm(req.POST, req.FILES)
            if form.is_valid():
                cnt = 0
                for ff in req.FILES.getlist('photo_image'):
                    effidata = EfficientData()

                    effidata.photo_image = ff
                    effidata.save()

                    effidata.pred_result = pred(settings.MEDIA_ROOT + '/' + str(effidata.photo_image))
                    effidata.save()
                    
                    cnt = cnt + 1

                self.params['effidata'] = EfficientData.objects.all().order_by('-id')[:cnt]
                             
        else:
            form = MultiForm()

        return render(req, 'effi_app/multi.html', self.params)


    
    