from django.contrib import admin
from django.urls import path, include

#from effi_proj import settings
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

print('----effi_proj.urls----')
urlpatterns = [
    path('admin/', admin.site.urls),
    path('effi_app/', include('effi_app.urls')),
    path('account/', include('allauth.urls')),
]

if settings.DEBUG:
    #urlpatterns += staticfiles_urlpatterns()
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
