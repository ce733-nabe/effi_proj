from django.db import models

# Create your models here.
class PhotoImage(models.Model):

    photo_image = models.ImageField(verbose_name='画像',upload_to='images/')
    
    def __str__(self):
        return self.photo_image