from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from accounts.models import CustomUser
from .models import EfficientData
 
 
class CustomUserAdmin(UserAdmin):
    model = CustomUser
    #fieldsets = UserAdmin.fieldsets 
    list_display = ('username', 'email',)
 
admin.site.register(CustomUser, CustomUserAdmin)

class EfficientDataAdmin(admin.ModelAdmin):
    model = EfficientData
    #fieldsets = UserAdmin.fieldsets 
    list_display = ('pub_date', 'photo_image','pred_result')
 
admin.site.register(EfficientData, EfficientDataAdmin)
#admin.site.register(EfficientData)
