from pathlib import Path

from tqdm import tqdm
import pandas as pd
import numpy as np
import torchvision.ops.boxes as bops
import torch
import pickle

from src.detect_faces import detect, save_crops


# bounding_boxes, confidences, paths, shapes = detect(source='../dataset/VGG-Face2/data/test/**/*.jpg',
#                                                     weights='../models/yolov7-lite-s.pt',
#                                                     img_size=256,
#                                                     conf_threshold=0.25,
#                                                     iou_threshold=0.45)
#
#
# with open('bounding_boxes.pkl', 'wb') as f:
#     pickle.dump(bounding_boxes, f)
# with open('confidences.pkl', 'wb') as f:
#     pickle.dump(confidences, f)
# np.save('paths.npy', [str(x) for x in paths])
# np.save('shapes.npy', [np.array(x) for x in shapes])


with open('bounding_boxes.pkl', 'rb') as f:
    bounding_boxes = pickle.load(f)
with open('confidences.pkl', 'rb') as f:
    confidences = pickle.load(f)
paths = [Path(x) for x in np.load('paths.npy')]
shapes = np.load('shapes.npy')


ann = pd.read_csv('../dataset/VGG-Face2/meta/bb_landmark/loose_bb_test.csv', index_col='NAME_ID')
assert np.array_equal(ann.index.values, np.sort(ann.index.values))

xs = ann.X.values
ys = ann.Y.values
ws = ann.W.values
hs = ann.H.values

res_bb, res_p = [], []


for bboxes, c, p, s in tqdm(zip(bounding_boxes, confidences, paths, shapes)):
    i = np.searchsorted(ann.index, str(p)[-19:-4])
    gt_bbox = np.asarray([xs[i] / s[1], ys[i] / s[0], (xs[i] + ws[i]) / s[1], (ys[i] + hs[i]) / s[0]])

    ious = []
    for bbox in bboxes:
        box1 = torch.tensor(np.array([bbox]), dtype=torch.float)
        box2 = torch.tensor(np.array([gt_bbox]), dtype=torch.float)
        iou = bops.box_iou(box1, box2)
        ious.append(iou.numpy())

    ious = np.array(ious)
    if not np.sum(ious > 0.5) == 1:
        print('!!!Error', ious, p)
        continue

    res_bb.append(np.array([bboxes[np.argmax(ious)]]))
    res_p.append(p)

save_crops(res_bb, res_p, Path('../dataset/VGG-Face2/data/crops'))
