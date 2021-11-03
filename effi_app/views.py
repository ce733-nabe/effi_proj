from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views.generic import TemplateView
from .forms import SingleForm ,MultiForm
from effi_app.main import pred, Pred, Preds
from .models import EfficientData
from django.conf import settings
import time

class IndexView(TemplateView):
    template_name = 'effi_app/index.html'

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
            pred = Pred()

            if form.is_valid():
                effidata = EfficientData()

                effidata.photo_image = form.cleaned_data['photo_image']
                effidata.save()

                #effidata.pred_result = pred(settings.MEDIA_ROOT + '/' + str(effidata.photo_image))
                effidata.pred_result =pred.pred(settings.MEDIA_ROOT + '/' + str(effidata.photo_image))
                effidata.save()
                
                self.params['effidata'] = EfficientData.objects.all().order_by('-id')[:1]
        else:
            form = SingleForm()

        self.params[form] = form
        return render(req, 'effi_app/single.html', self.params)


class MultiView(TemplateView):

    def __init__(self):
        print('------------init-----------')
        self.params={'effidata': 'none',
                    'form': MultiForm(),
                    }

    def get(self, req):
        print('------------get-----------')
        return render(req, 'effi_app/multi.html', self.params)

    def post(self, req):
        print('------------post-----------')
        start = time.time()

        if req.method == 'POST':
            form = MultiForm(req.POST, req.FILES)
            #pred = Pred()

            if form.is_valid():
                cnt = 0
                pimg_box = []
                pimg_box2 = []
    
                for ff in req.FILES.getlist('photo_image'):
                    pimg_box.append(EfficientData(photo_image=ff, pred_result=''))
                    cnt = cnt + 1
                EfficientData.objects.bulk_create(pimg_box)
                

                pimg_box2 = []
                iiis = EfficientData.objects.all().order_by('-id')[:cnt]
                for iii in iiis:
                    pimg_box2.append(settings.MEDIA_ROOT + '/' + str(iii.photo_image))
                preds = Preds(filenames=pimg_box2, batch_size=20).preds()


                cnt2 = 0
                iiis = EfficientData.objects.all().order_by('-id')[:cnt]
                for iii in iiis:
                    iii.pred_result = str(preds[cnt2])
                    cnt2 = cnt2 + 1
                EfficientData.objects.bulk_update(iiis, fields=["pred_result"])

                self.params['effidata'] = EfficientData.objects.all().order_by('-id')[:cnt]
                             
        else:
            form = MultiForm()

        self.params[form] = form

        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

        return render(req, 'effi_app/multi.html', self.params)


    
    