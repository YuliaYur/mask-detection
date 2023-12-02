from glob import glob
from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np
from numpy.random import randint

from spiga_project.spiga.inference.config import ModelConfig
from spiga_project.spiga.inference.framework import SPIGAFramework


np.random.seed(0)


dataset = 'merlrav'  # 'wflw'
processor = SPIGAFramework(ModelConfig(dataset))

mask = np.zeros(68, dtype=bool)  # 98
# mask[3:30] = True
# mask[53] = True
mask[1:16] = True
mask[29] = True

for i, img_path in tqdm(enumerate(sorted(glob('../dataset/VGG-Face2/data/for_vgg/no_mask/*.jpg')))):
    # Load image and bbox
    image = cv2.imread(img_path)
    bbox = [0, 0, image.shape[1], image.shape[0]]

    # Process image
    features = processor.inference(image, [bbox])

    # Prepare variables
    # x0, y0, w, h = bbox
    landmarks = np.array(features['landmarks'][0])
    landmarks = landmarks[mask]

    landmarks = landmarks.astype(int)

    random_color = [randint(0, 256), randint(0, 256), randint(0, 256)]
    colors = np.array([[245, 245, 245], [0, 0, 0], random_color])
    color = colors[np.random.choice(np.arange(3), p=[0.5, 0.2, 0.3])]

    cv2.fillPoly(image, pts=[landmarks], color=(int(color[0]), int(color[1]), int(color[2])))

    img_path = img_path.replace('no_mask', 'mask')
    # Path(img_path[:-12]).mkdir(parents=False, exist_ok=True)

    cv2.imwrite(img_path, image)


# images = []
# for img_path in tqdm(sorted(glob('../dataset/VGG-Face2/data/crops_copy/**/*.jpg'))[:8]):
#     # Load image and bbox
#     image = cv2.imread(img_path)
#     images.append(image)
#
# # Process image
# features = processor.inference_cropped_images(images, [np.array([0, 0, x.shape[1], x.shape[0]]) for x in images])
#
# # Prepare variables
# # x0, y0, w, h = bbox
# landmarks = np.array(features['landmarks'])
# # l = landmarks[0, mask]
# #
# # landmarks = landmarks.astype(int)
# # cv2.fillPoly(image, pts=[l], color=(randint(0, 256), randint(0, 256), randint(0, 256)))
# # cv2.imwrite(img_path, image)
