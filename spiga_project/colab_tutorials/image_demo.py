import cv2
import json
import numpy as np

# Load image and bbox
image = cv2.imread("../assets/colab/image_sportsfan.jpg")
with open('../assets/colab/bbox_sportsfan.json') as jsonfile:
    bbox = json.load(jsonfile)['bbox']

#
# image = image[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
# bbox = [0, 0, image.shape[1], image.shape[0]]

from spiga_project.spiga.inference.config import ModelConfig
from spiga_project.spiga.inference.framework import SPIGAFramework

# Process image
dataset = 'merlrav'
processor = SPIGAFramework(ModelConfig(dataset))
features = processor.inference(image, [bbox])

import copy
from spiga_project.spiga.demo.visualize.plotter import Plotter

# Prepare variables
# x0,y0,w,h = bbox
canvas = copy.deepcopy(image)
landmarks = np.array(features['landmarks'][0])
print(len(landmarks))
# mask = np.zeros(len(landmarks), dtype=bool)
# mask[1:16] = True
# mask[29] = True
# landmarks = landmarks[mask]
# headpose = np.array(features['headpose'][0])


# cv2.rectangle(canvas, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (255, 0, 0), thickness=4, lineType=cv2.LINE_AA)

plotter = Plotter()
canvas = plotter.landmarks.draw_landmarks(canvas, landmarks, thick=5)

# cv2.fillPoly(canvas, pts=[landmarks.astype(int)], color=(0, 0, 0))
# Show image results
(h, w) = canvas.shape[:2]
canvas = cv2.resize(canvas, (512, int(h*512/w)))
cv2.imwrite('a2.png', canvas)
