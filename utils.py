import cv2
import torch
from PIL import Image
from mtcnn import MTCNN  # Или замените на retinaface

# Альтернатива Haar Cascade - MTCNN или RetinaFace (гораздо лучше)
#def detect_and_align_face(image):
#  """Обнаруживает и выравнивает лицо на изображении (Haar Cascade)."""
#  face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#  faces = face_detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#  if len(faces) == 0:
#    return None
#
#  (x, y, w, h) = faces[0]
#  face = image[y:y+h, x:x+w]
#  return face


def detect_and_align_face(image):
  """Обнаруживает и выравнивает лицо на изображении (MTCNN)."""
  detector = MTCNN()
  faces = detector.detect_faces(image)
  if not faces:
      return None

  # Берем первое обнаруженное лицо
  face = faces[0]
  x, y, width, height = face['box']

  # Обрезаем лицо
  face_image = image[y:y+height, x:x+width]
  return face_image
