from django import forms
from .models import EfficientData
#from PIL import Image


class SingleForm(forms.ModelForm):
    class Meta:
        model = EfficientData
        fields = ('photo_image',)

    
class MultiForm(forms.Form):
    photo_image = forms.ImageField(widget=forms.ClearableFileInput(attrs={'multiple': True}))

    
    

        

        
        
        
        
    
    