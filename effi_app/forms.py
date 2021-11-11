from django import forms
from .models import EfficientData
#from PIL import Image


class SingleForm(forms.ModelForm):
    print('----forms.SingleForm----')
    class Meta:
        model = EfficientData
        fields = ('photo_image',)

    
class MultiForm(forms.Form):
    print('----forms.MultiForm----')
    photo_image = forms.ImageField(widget=forms.ClearableFileInput(attrs={'multiple': True}))

    
    

        

        
        
        
        
    
    