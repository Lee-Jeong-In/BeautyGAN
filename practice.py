import dlib                             # 이미지 처리 라이브러리 (이미지 landmark 처리)
import matplotlib.pyplot as plt         # 그래프 패키지
import matplotlib.image as img
import matplotlib.patches as patches
import tensorflow as tf                 # 머신러닝 라이브러리
import numpy as np                      # 행렬 연산 라이브러리

image_path = './imgs/01.jpg'
image = img.imread(image_path)
plt.imshow(image)
plt.axis('off')
plt.show()


'''
# Load Models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')

# Load Images
img = dlib.load_rgb_image('imgs/01.jpg')

plt.figure(figsize=(16, 10))

plt.imshow(img)
'''
