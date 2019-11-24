import cv2
import numpy as np
from PIL import Image
import glob
import os

# I used ffmpeg to extract every frame from the videos,
# then I moved the frames to a folder called A (or B).
# Once I'd done that, I ran this script and got the folder
# with the faces ready to train.
#
# Note: this method sometimes confuses faces with random
# things. Therefore the datase won't be perfect.

directory = '../dataset/[A or B]/'

extracted = directory+'extracted'

import os
if not os.path.exists(extracted):
    os.makedirs(extracted)

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

num = 0
for image in glob.glob(directory+'*.jpg'):
    img = cv2.imread(image)
    print(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.imwrite(f"{extracted}/{num}.jpg", img[y:y+h, x:x+w])

    num +=1
