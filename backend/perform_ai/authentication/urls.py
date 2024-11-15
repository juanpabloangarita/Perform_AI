from django.contrib.auth.views import LoginView, LogoutView
from django.urls import path
from rest_framework.routers import DefaultRouter

app_name = 'authentication'

router = DefaultRouter()

urlpatterns = [
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
]