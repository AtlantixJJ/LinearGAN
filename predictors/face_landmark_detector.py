# python 3.7
"""Contains the face landmark detector."""

import os.path
import bz2
import requests
import numpy as np
import scipy.ndimage
from PIL import Image, ImageDraw
import dlib  # pip install dlib

from .predictor_settings import MODEL_DIR

__all__ = ['FaceLandmarkDetector']

LANDMARK_MODEL_NAME = 'shape_predictor_68_face_landmarks.dat'
LANDMARK_MODEL_PATH = os.path.join(MODEL_DIR, LANDMARK_MODEL_NAME)
LANDMARK_MODEL_URL = f'http://dlib.net/files/{LANDMARK_MODEL_NAME}.bz2'


class FaceLandmarkDetector(object):
  """Class of face landmark detector."""

  def __init__(self, align_size=1024, enable_padding=True):
    """Initializes face detector and landmark detector.

    Args:
      align_size: Size of the aligned face if performing face alignment.
        (default: 1024)
      enable_padding: Whether to enable padding for face alignment (default:
        True)
    """
    # Download models if needed.
    if not os.path.exists(LANDMARK_MODEL_PATH):
      data = requests.get(LANDMARK_MODEL_URL)
      data_decompressed = bz2.decompress(data.content)
      with open(LANDMARK_MODEL_PATH, 'wb') as f:
        f.write(data_decompressed)

    self.face_detector = dlib.get_frontal_face_detector()
    self.landmark_detector = dlib.shape_predictor(LANDMARK_MODEL_PATH)
    self.align_size = align_size
    self.enable_padding = enable_padding

  def detect(self, image_path):
    """Detects landmarks from the given image.

    This function will first perform face detection on the input image. All
    detected results will be grouped into a list. If no face is detected, an
    empty list will be returned.

    For each element in the list, it is a dictionary consisting of `image_path`,
    `bbox` and `landmarks`. `image_path` is the path to the input image. `bbox`
    is the 4-element bounding box with order (left, top, right, bottom), and
    `landmarks` is a list of 68 (x, y) points.

    Args:
      image_path: Path to the image to detect landmarks from.

    Returns:
      A list of dictionaries, each of which is the detection results of a
        particular face.
    """
    results = []
    image = dlib.load_rgb_image(image_path)
    # Face detection (1 means to upsample the image for 1 time.)
    bboxes = self.face_detector(image, 1)
    # Landmark detection
    for bbox in bboxes:
      landmarks = []
      for point in self.landmark_detector(image, bbox).parts():
        landmarks.append((point.x, point.y))
      results.append({
          'image_path': image_path,
          'bbox': (bbox.left(), bbox.top(), bbox.right(), bbox.bottom()),
          'landmarks': landmarks,
      })
    return results

  def align(self, face_info, save_path='', viz_path=''):
    """Aligns face based on landmark detection.

    The face alignment process is borrowed from
    https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py,
    which only supports aligning faces to square size.

    Args:
      face_info: Face information, which is the element of the list returned by
        `self.detect()`.
      save_path: Path to save the aligned result. If not specified, the aligned
        result will not be save to disk.
      viz_path: Path to save the visualization result, which will draw bounding
        box and the landmarks on the raw image. If not specified, visualization
        will be skipped.

    Returns:
      A `np.ndarray`, containing the aligned result. It is with `RGB` channel
        order.
    """
    img = Image.open(face_info['image_path'])
    if viz_path:
      viz_img = img.copy()
      draw = ImageDraw.Draw(viz_img)
      draw.rectangle(face_info['bbox'], outline=(255, 0, 0))
      for point in face_info['landmarks']:
        x = point[0]
        y = point[1]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=(0, 0, 255))
      viz_img.save(viz_path)

    landmarks = np.array(face_info['landmarks'])
    eye_left = np.mean(landmarks[36 : 42], axis=0)
    eye_right = np.mean(landmarks[42 : 48], axis=0)
    eye_middle = (eye_left + eye_right) / 2
    eye_to_eye = eye_right - eye_left
    mouth_middle = (landmarks[48] + landmarks[54]) / 2
    eye_to_mouth = mouth_middle - eye_middle

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_middle + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / self.align_size * 0.5))
    if shrink > 1:
      rsize = (int(np.rint(float(img.size[0]) / shrink)),
               int(np.rint(float(img.size[1]) / shrink)))
      img = img.resize(rsize, Image.ANTIALIAS)
      quad /= shrink
      qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0),
            max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
      img = img.crop(crop)
      quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0),
           max(-pad[1] + border, 0),
           max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if self.enable_padding and max(pad) > border - 4:
      pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
      img = np.pad(np.float32(img),
                   ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
                   'reflect')
      h, w, _ = img.shape
      y, x, _ = np.ogrid[:h, :w, :1]
      mask = np.maximum(
          1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]),
          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
      blur = qsize * 0.02
      blurred_image = scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img
      img += blurred_image * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
      img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
      img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
      quad += pad[:2]

    # Transform.
    img = img.transform((self.align_size * 4, self.align_size * 4), Image.QUAD,
                        (quad + 0.5).flatten(), Image.BILINEAR)
    img = img.resize((self.align_size, self.align_size), Image.ANTIALIAS)

    # Save results
    if save_path:
      img.save(save_path)

    return np.asarray(img)
