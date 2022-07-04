# pyQt5
import dlib #이미지처리 라이브러리 , 페이스 디텍션, 랜드마크 디텍션, 페이스 얼라이브
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os
import cv2
import sys


#Load Models
detector = dlib.get_frontal_face_detector()  #얼굴 영역 인식 모델 로드
sp = dlib.shape_predictor("./models/shape_predictor_5_face_landmarks.dat")

#Load Images
img = dlib.load_rgb_image("./imgs/01.jpg")

#Face Detection
img_result = img.copy()

dets = detector(img,1) #이미지에서 얼굴 영역을 찾는다.

if len(dets) == 0: #얼굴영역의 갯수가 0일 경우
  print('cannot find faces!')

fig, ax = plt.subplots(1, figsize=(16,10))

for det in dets: #det이라는 직사각형 정보에 대한 for문
  x, y, w, h = det.left(), det.top(), det.width(), det.height()

  rect = patches.Rectangle((x,y),w, h, linewidth=2, edgecolor='r', facecolor='none')
  ax.add_patch(rect)

ax.imshow(img)

#Landmark Detection
fig, ax = plt.subplots(1, figsize=(16,10))

objs =dlib.full_object_detections() #얼굴 수평맞춰줄때 사용

for detection in dets:
  s = sp(img, detection) #sp() : 얼굴의 랜드마크를 찾는다.
  objs.append(s)

  for point in s.parts(): #5개의 점에 대한 for문
    circle = patches.Circle((point.x,point.y), radius=3, edgecolor='r',facecolor='r')
    ax.add_patch(circle)

ax.imshow(img_result)

#Align Faces
faces = dlib.get_face_chips(img, objs, size=256, padding=0.3) #dilb.get_face_chips() : 얼굴을 수평으로 회전하여 얼굴 부분만 자른 이미지 반환

flg,axes = plt.subplots(1, len(faces)+1, figsize=(20,16))

axes[0].imshow(img) #원본이미지

for i,face in enumerate(faces): #디텍션한 얼굴이 여러개면 여러개 찍어준다.
  axes[i+1].imshow(face)


#To be Function
def align_faces(img):  # 원본이미지를 넣으면 align 완료된 얼굴이미지 반환하는 함수
    dets = detector(img, 1)

    objs = dlib.full_object_detections()

    for detection in dets:
        s = sp(img, detection)
        objs.append(s)

    faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)

    return faces

# test
test_img = dlib.load_rgb_image("./imgs/test.jpg")

test_faces = align_faces(test_img)

fig, axes = plt.subplots(1, len(test_faces) + 1, figsize=(20, 16))
axes[0].imshow(test_img)

for i, face in enumerate(test_faces):
    axes[i + 1].imshow(face)

plt.figure(figsize=(16,10))
plt.imshow(img)
plt.show()
