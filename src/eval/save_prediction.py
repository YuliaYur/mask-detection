import json

import numpy as np
import cv2
import torch
from tqdm import tqdm
import keras
from torchvision.transforms.functional import normalize

from yolo7_face.models.experimental import attempt_load
from yolo7_face.utils.general import check_img_size, non_max_suppression, scale_coords
from yolo7_face.utils.torch_utils import select_device
from yolo7_face.utils.datasets import letterbox

from code_former.basicsr.utils.registry import ARCH_REGISTRY
from code_former.basicsr.utils import img2tensor, tensor2img


device = select_device('cpu')
weights = '../../models/yolov7-lite-s.pt'
face_detector = attempt_load(weights, map_location=device)
img_size = 640
stride = int(face_detector.stride.max())
img_size = check_img_size(img_size, s=stride)
conf_threshold: float = 0.25
iou_threshold: float = 0.45


mask_classificator = keras.models.load_model('../../models/efficientnet_v2_b3/original_data/efficientnetv2_b3_original_data_epoch_6.h5')


face_sr = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512,
                                          codebook_size=1024,
                                          n_head=8,
                                          n_layers=9,
                                          connect_list=['32', '64', '128', '256']).to(device)

ckpt_path = '../../code_former/weights/CodeFormer/codeformer.pth'
checkpoint = torch.load(ckpt_path)['params_ema']
face_sr.load_state_dict(checkpoint)
face_sr.eval()
w = 0.5


dataset_path = '../../dataset/School/raw/'
with open('annotation/school/annotation.json', 'r') as f:
    gt_annotations = json.load(f)


pred_annotations = []
pred_id = 0

i = 0
for img_ann in tqdm(gt_annotations['images']):

    file_name = img_ann['file_name']
    image_id = img_ann['id']

    img0 = cv2.imread(dataset_path + file_name)
    img = letterbox(img0, img_size, stride=stride, auto=False)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = face_detector(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_threshold, iou_threshold, kpt_label=5)

    detection = pred[0]

    if len(detection):
        scale_coords(img.shape[2:], detection[:, :4], img0.shape, kpt_label=False)

    detection = np.array(detection)
    bbox = detection[:, :4]
    confs = detection[:, 4]


    face_imgs = []
    for x_min, y_min, x_max, y_max in bbox:
        x_min = int(max(x_min, 0))
        y_min = int(max(y_min, 0))
        x_max = int(min(x_max, img0.shape[1]))
        y_max = int(min(y_max, img0.shape[0]))

        face_img = img0[y_min: y_max, x_min: x_max]

        # cv2.imwrite('tmp.png', face_img)
        # img = cv2.imread('tmp.png', cv2.IMREAD_COLOR)
        # img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        #
        # cropped_face_t = img2tensor(img / 255., bgr2rgb=True, float32=True)
        # normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        # cropped_face_t = cropped_face_t.unsqueeze(0).to('cpu')
        #
        # with torch.no_grad():
        #     output = face_sr(cropped_face_t, w=w, adain=True)[0]
        #     restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
        # del output
        # torch.cuda.empty_cache()
        #
        # face_img = restored_face.astype('uint8')
        # cv2.imwrite(f'_/{i}.png', face_img)
        # face_img = cv2.imread(f'_/{i}.png')
        # i += 1

        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
        face_img = cv2.resize(face_img, (64, 64))
        face_imgs.append(face_img)

    if len(face_imgs) == 0:
        continue
    face_imgs = np.reshape(face_imgs, [-1, 64, 64, 3])
    mask_results = mask_classificator.predict(face_imgs, verbose=0)

    for (x_min, y_min, x_max, y_max), conf, mask_result in zip(bbox, confs, mask_results):
        c = 1 - mask_result.argmax()
        score = mask_result.max()

        pred_annotations.append({'id': pred_id,
                                 'image_id': image_id,
                                 'category_id': int(c),
                                 'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                                 'score': float(conf * score)})
        pred_id += 1


with open('prediction/school/eff_b3_original_data_epoch_6.json', 'w') as f:
    json.dump(pred_annotations, f, indent=4)
