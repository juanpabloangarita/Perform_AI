from django.contrib.auth.views import LogoutView
from django.urls import path
from rest_framework.routers import DefaultRouter
from . import views

app_name = 'authentication'

router = DefaultRouter()

urlpatterns = [
    path('login/', views.LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
]