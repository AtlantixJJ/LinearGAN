# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.template import loader, Context
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import home.api as api
import traceback
from io import BytesIO
from PIL import Image
from base64 import b64encode, b64decode
from datetime import datetime


EDIT_INDEX = "index_edit.html"
TRAIN_INDEX = "index_train.html"
model_manager = api.ModelAPI("home/static/config.json")
editor = api.ImageGenerationAPI(model_manager)
trainer = api.TrainAPI(model_manager)
model_name = list(model_manager.models_config.keys())[0]
imsize = model_manager.models_config[model_name]["output_size"]
base_dic = {
  "imsize" : imsize,
  "canvas_box" : imsize * 2 + 50}

# np.array
def image2bytes(image):
  buffered = BytesIO()
  Image.fromarray(image).save(buffered, format="PNG")
  return b64encode(buffered.getvalue()).decode('utf-8')


def response_image_label(image, label):
  imageString = image2bytes(image)
  segString = image2bytes(label)
  json = '{"ok":"true","img":"data:image/png;base64,%s","label":"data:image/png;base64,%s"}' % (imageString, segString)
  return HttpResponse(json)


def response_image(image):
  imageString = image2bytes(image)
  json = '{"ok":"true","img":"data:image/png;base64,%s"}' % imageString
  return HttpResponse(json)


def save_to_session(session, z_s):
  session["z_s"] = z_s

def restore_from_session(session):
  latent = session["z_s"]
  return latent


def index(request):
  res = render(request, EDIT_INDEX, base_dic)
  res.set_cookie('last_visit', datetime.now())
  return res

def train(request):
  res = render(request, TRAIN_INDEX, base_dic)
  res.set_cookie('last_visit', datetime.now())
  return res

@csrf_exempt
def generate_image_given_stroke(request):
  form_data = request.POST
  sess = request.session
  if request.method == 'POST' and 'image_stroke' in form_data:
    try:
      model = form_data['model']
      if not editor.has_model(model):
        print(f"!> Model not exist {model}")
        return HttpResponse('{}')

      imageStrokeData = b64decode(form_data['image_stroke'].split(',')[1])
      labelStrokeData = b64decode(form_data['label_stroke'].split(',')[1])
      latent = restore_from_session(sess)

      imageStroke = Image.open(BytesIO(imageStrokeData))
      labelStroke = Image.open(BytesIO(labelStrokeData))
      # TODO: hard coded for stylegan
      imageStroke, imageMask = api.stroke2array(imageStroke)
      labelStroke, labelMask = api.stroke2array(labelStroke)

      image, label, latent = editor.generate_image_given_stroke(
        model, latent,
        imageStroke, imageMask,
        labelStroke, labelMask)
      save_to_session(sess, latent)
      return response(image, label)
    except Exception:
      print("!> Exception:")
      traceback.print_exc()
      return HttpResponse('{}')
  print(f"!> Invalid request: {str(form_data.keys())}")
  return HttpResponse('{}')

@csrf_exempt
def generate_new_image(request):
  form_data = request.POST
  sess = request.session
  if request.method == 'POST' and 'model' in form_data:
    try:
      model = form_data['model']
      if not editor.has_model(model):
        print("=> No model name %s" % model)
        return HttpResponse('{}')

      image, z = editor.generate_new_image(model)
      save_to_session(sess, z)
      return response(image, label)
    except Exception:
      print("!> Exception:")
      traceback.print_exc()
      return HttpResponse('{}')
  return HttpResponse('{}')

@csrf_exempt
def train_get_new_image(request):
  form_data = request.POST
  sess = request.session
  sess.flush()
  if request.method == 'POST' and 'model' in form_data:
    try:
      model = form_data['model']
      if not editor.has_model(model):
        print("=> No model name %s" % model)
        return HttpResponse('{}')

      image, z_s = trainer.generate_new_image(model_name)
      save_to_session(sess, z_s)
      return response_image(image)
    except Exception:
      print("!> Exception:")
      traceback.print_exc()
      return HttpResponse('{}')
  return HttpResponse('{}')
