# Black Bg --> Colorful Image (Tourism Places of Bangkok)

# Import 

import cv2
import os
import mediapipe as mp
import numpy as np 

# Store Background Images in a List

imgPath = 'Images'
images = os.listdir(imgPath)

img = 0

# Path of bg
bg = cv2.imread(imgPath + '/' + images[img])

#---------------------Selfie Segmentation-------------------------------
MP_SelfieSegmentation = mp.solutions.selfie_segmentation

Selfie_Segmentation = MP_SelfieSegmentation.SelfieSegmentation(model_selection=1)

Capture = cv2.VideoCapture(0)

while Capture.isOpened():
    r , frame = Capture.read()

    frame = np.flip(frame , axis = 1)

    height , width , channel = frame.shape

    rgb = cv2.cvtColor( frame , cv2.COLOR_BGR2RGB)

    result = Selfie_Segmentation.process(rgb)

    # # extract segmented mask
    # mask = results.segmentation_mask
    # # show outputs
    # cv2.imshow("mask", mask)
    # cv2.imshow("Frame", frame)

    # Masking

    mask = result.segmentation_mask

    key = cv2.waitKey(1)

    # After pressing 'q' , break it

    if key == ord('q'):
        break

    condition = np.stack(
    (result.segmentation_mask,) * 3, axis=-1) > 0.5

    bg = cv2.resize( bg , (width , height))

    output = np.where(condition, frame, bg)
    cv2.imshow("Output", output)

    key = cv2.waitKey(1)

    if key == ord('q'):
                break
    
  # if 'd' key is pressed then change the background image
    elif key == ord('d'):
        if img != len(images)-1:
              img += 1
        else:
              img = 0
        bg = cv2.imread(imgPath+'/'+images[img])

