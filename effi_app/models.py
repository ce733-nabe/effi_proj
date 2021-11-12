from django.db import models
from django.utils import timezone

# Create your models here.
class EfficientData(models.Model):
    pub_date = models.DateTimeField(verbose_name='日付',default=timezone.now)
    photo_image = models.ImageField(verbose_name='画像',upload_to='images/%Y%m%d/')
    #photo_image = models.ImageField(verbose_name='画像',upload_to='images/%Y%m%d_%H%M%S/')
    pred_result = models.CharField(verbose_name='推論結果',max_length=200)
    
    def __str__(self):
        return str(self.pub_date)

    #def __str__(self):
    #    return str(self.photo_image)

    #def __str__(self):
    #    return str(self.pub_date)