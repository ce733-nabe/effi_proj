from django import forms
from .models import PhotoImage
#from PIL import Image


class PhotoImageForm(forms.ModelForm):
    class Meta:
        model = PhotoImage
        fields = ('photo_image',)

        

        
        
        
        
    
    