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
editor = api.EditAPI(model_manager)
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


def save_to_session(session, zs):
  session["zs"] = zs


def restore_from_session(session):
  zs = session["zs"]
  return zs


def index(request):
  res = render(request, EDIT_INDEX, base_dic)
  res.set_cookie('last_visit', datetime.now())
  return res

def train(request):
  res = render(request, TRAIN_INDEX, base_dic)
  res.set_cookie('last_visit', datetime.now())
  return res

@csrf_exempt
def add_annotation(request):
  print("Add annotation")
  form_data = request.POST
  sess = request.session
  if request.method == 'POST' and 'ann' in form_data:
    try:
      model = form_data['model']
      if not editor.has_model(model):
        print(f"!> Model not exist {model}")
        return HttpResponse('{}')

      ann = b64decode(form_data['ann'].split(',')[1])
      zs = restore_from_session(sess)

      ann = Image.open(BytesIO(ann))
      ann, ann_mask = api.stroke2array(ann)

      trainer.add_annotation(
        model, zs,
        ann, ann_mask)
      return HttpResponse('{}') # no need to return any information
    except Exception:
      print("!> Exception:")
      traceback.print_exc()
      return HttpResponse('{}')
  print(f"!> Invalid request: {str(form_data.keys())}")
  return HttpResponse('{}')

@csrf_exempt
def ctrl_training(request):
  form_data = request.POST
  sess = request.session
  if request.method == 'POST' and 'model' in form_data:
    try:
      model = form_data['model']
      if not editor.has_model(model):
        print(f"!> Model not exist {model}")
        return HttpResponse('{}')
      cmd = form_data['action']
      flag = trainer.ctrl_training(model, cmd)
      flag = '"true"' if flag else '"false"'
      json = '{"action": "%s", "status" : %s}'
      return HttpResponse(json % (cmd, flag))
    except Exception:
      print("!> Exception:")
      traceback.print_exc()
      return HttpResponse('{}')
  print(f"!> Invalid request: {str(form_data.keys())}")
  return HttpResponse('{}')

@csrf_exempt
def get_validation(request):
  form_data = request.POST
  sess = request.session
  if request.method == 'POST' and 'model' in form_data:
    try:
      model = form_data['model']
      if not editor.has_model(model):
        print(f"!> Model not exist {model}")
        return HttpResponse('{}')

      image, segviz = trainer.get_validation(model)
      img_format = '"data:image/png;base64,{img}"'
      image_str = ",".join([img_format.format(img=image2bytes(img))
        for img in image])
      label_str = ",".join([img_format.format(img=image2bytes(img))
        for img in segviz])
      json = '{"ok":"true", "images": [%s], "labels": [%s]}'
      return HttpResponse(json % (image_str, label_str))
    except Exception:
      print("!> Exception:")
      traceback.print_exc()
      return HttpResponse('{}')
  print(f"!> Invalid request: {str(form_data.keys())}")
  return HttpResponse('{}')


@csrf_exempt
def clear_annotation(request):
  pass

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

      image_stroke = b64decode(form_data['image_stroke'].split(',')[1])
      label_stroke = b64decode(form_data['label_stroke'].split(',')[1])
      zs = restore_from_session(sess)

      image_stroke = Image.open(BytesIO(image_stroke))
      label_stroke = Image.open(BytesIO(label_stroke))
      image_stroke, image_mask = api.stroke2array(image_stroke)
      label_stroke, label_mask = api.stroke2array(label_stroke)

      image, label, zs = editor.generate_image_given_stroke(
        model, zs,
        image_stroke, image_mask,
        label_stroke, label_mask)
      save_to_session(sess, zs)
      return response_image_label(image, label)
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

      image, label, zs = editor.generate_new_image(model)
      save_to_session(sess, zs)
      return response_image_label(image, label)
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
