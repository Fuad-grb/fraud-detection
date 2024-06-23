from django.contrib import admin
from django.urls import path, include
from django.http import HttpResponse

def home(request):
    return HttpResponse("Welcome to the Fraud Detection API")

urlpatterns = [
    path('admin/', admin.site.urls),
    path('predict/', include('predict.urls')),
    path('', home),  
]
