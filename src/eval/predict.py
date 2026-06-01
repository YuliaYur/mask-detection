"""Run the face-mask detection pipeline over a dataset and return COCO predictions.

Two-stage (default): YOLOv7-Face detects faces, then the Keras classifier labels each crop
mask / no-mask. With a CodeFormer checkpoint the crops are restored first (three-stage).
With ``use_gt_boxes`` the detector is skipped and the ground-truth boxes are classified
instead (isolates the classifier from the detector).

This is a library used by the `python -m src.eval` endpoint.
"""

from pathlib import Path


def _load_detector(weights, img_size, device):
    import torch

    from yolo7_face.models.experimental import attempt_load
    from yolo7_face.utils.general import check_img_size
    from yolo7_face.utils.torch_utils import select_device

    torch.set_num_threads(1)  # few cores here; avoids extra thread-pool memory
    dev = select_device(device)
    model = attempt_load(str(weights), map_location=dev)
    stride = int(model.stride.max())
    return model, dev, stride, check_img_size(img_size, s=stride)


def _detect(model, dev, stride, img_size, img0, conf, iou):
    import numpy as np
    import torch

    from yolo7_face.utils.datasets import letterbox
    from yolo7_face.utils.general import non_max_suppression, scale_coords

    img = letterbox(img0, img_size, stride=stride, auto=False)[0]
    img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))  # BGR->RGB, HWC->CHW
    img = torch.from_numpy(img).to(dev).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():  # inference only: don't build the autograd graph / retain activations
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf, iou, kpt_label=5)
    detection = pred[0]
    if len(detection):
        scale_coords(img.shape[2:], detection[:, :4], img0.shape, kpt_label=False)

    detection = np.array(detection)
    if detection.size == 0:
        return np.empty((0, 4)), np.empty((0,))
    return detection[:, :4], detection[:, 4]


def _load_codeformer(ckpt, dev):
    import torch

    from code_former.basicsr.utils.registry import ARCH_REGISTRY

    net = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(dev)
    net.load_state_dict(torch.load(str(ckpt))["params_ema"])
    net.eval()
    return net


def _restore_face(net, face_bgr, weight, dev):
    import cv2
    import torch
    from torchvision.transforms.functional import normalize

    from code_former.basicsr.utils import img2tensor, tensor2img

    img = cv2.resize(face_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
    tensor = img2tensor(img / 255.0, bgr2rgb=True, float32=True)
    normalize(tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    tensor = tensor.unsqueeze(0).to(dev)
    with torch.no_grad():
        output = net(tensor, w=weight, adain=True)[0]
        restored = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
    return restored.astype("uint8")


def run_predictions(
    ground_truth: dict,
    images_dir,
    classifier_path,
    detector_path=None,
    use_gt_boxes: bool = False,
    codeformer_path=None,
    codeformer_weight: float = 0.5,
    img_size: int = 640,
    conf: float = 0.25,
    nms_iou: float = 0.45,
    classifier_size: int = 64,
    device: str = "cpu",
) -> list:
    """Detect (or take ground-truth) faces, classify them, and return COCO predictions."""
    import cv2
    import keras
    import numpy as np
    from tqdm import tqdm

    if not use_gt_boxes and detector_path is None:
        raise ValueError("detector_path is required unless use_gt_boxes=True.")

    images_dir = Path(images_dir)
    classifier = keras.models.load_model(str(classifier_path))

    detector = dev = stride = det_img_size = None
    if not use_gt_boxes:
        detector, dev, stride, det_img_size = _load_detector(detector_path, img_size, device)

    codeformer = None
    if codeformer_path is not None:
        import torch

        if dev is None:
            dev = torch.device(device)
        codeformer = _load_codeformer(codeformer_path, dev)

    boxes_by_image = {}
    if use_gt_boxes:
        for ann in ground_truth["annotations"]:
            boxes_by_image.setdefault(ann["image_id"], []).append(ann["bbox"])

    predictions = []
    pred_id = 0
    for img_ann in tqdm(ground_truth["images"]):
        image_id = img_ann["id"]
        img0 = cv2.imread(str(images_dir / img_ann["file_name"]))
        if img0 is None:
            continue

        if use_gt_boxes:
            boxes = np.array(boxes_by_image.get(image_id, []), dtype=float)
            confidences = np.ones(len(boxes))
        else:
            boxes, confidences = _detect(detector, dev, stride, det_img_size, img0, conf, nms_iou)

        if len(boxes) == 0:
            continue

        face_imgs = []
        for x_min, y_min, x_max, y_max in boxes:
            x_min, y_min = int(max(x_min, 0)), int(max(y_min, 0))
            x_max, y_max = int(min(x_max, img0.shape[1])), int(min(y_max, img0.shape[0]))
            face = img0[y_min:y_max, x_min:x_max]
            if codeformer is not None:
                face = _restore_face(codeformer, face, codeformer_weight, dev)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (classifier_size, classifier_size))
            face_imgs.append(face)

        # Direct call instead of classifier.predict(): predict() retraces and leaks memory
        # when called in a loop with a varying batch size. Numerically identical for inference.
        batch = np.reshape(face_imgs, [-1, classifier_size, classifier_size, 3]).astype("float32")
        results = classifier(batch, training=False).numpy()

        for box, conf_score, result in zip(boxes, confidences, results):
            predictions.append(
                {
                    "id": pred_id,
                    "image_id": image_id,
                    "category_id": 1 - int(result.argmax()),
                    "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    "score": float(conf_score) * float(result.max()),
                }
            )
            pred_id += 1

    return predictions
