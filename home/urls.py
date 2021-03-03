"""server URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from . import views

urlpatterns = [
    url(r'^edit$', views.index),
    url(r'^edit/stroke$', views.generate_image_given_stroke),
    url(r'^edit/new$', views.generate_new_image),
    url(r'^train$', views.train),
    url(r'^train/new$', views.train_get_new_image),
    url(r'^train/ann$', views.add_annotation),
    url(r'^train/clear$', views.clear_annotation),
    url(r'^train/val$', views.get_validation),
    url(r'^train/ctrl$', views.ctrl_training),
]
