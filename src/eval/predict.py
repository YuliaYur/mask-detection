"""The face-mask detection pipeline: detect faces, then classify each as mask / no-mask.

``MaskDetector`` loads the YOLOv7-Face detector and the EfficientNetV2-B3 classifier (plus an
optional CodeFormer restorer) once and reuses them across images -- it is the engine behind both
the ``python -m src.eval`` and ``python -m src.detect`` endpoints. ``run_predictions`` drives it
over a whole dataset and returns COCO-format predictions for mAP scoring.
"""

from pathlib import Path


class MaskDetector:
    """A loaded two/three-stage mask-detection pipeline.

    Construct once with the model weights, then call :meth:`detect` and :meth:`classify` per
    image. Pass ``detector_path=None`` to skip detection (e.g. to classify ground-truth boxes);
    pass ``codeformer_path`` to restore faces before classifying (the three-stage variant).
    """

    def __init__(
        self,
        classifier_path,
        detector_path=None,
        codeformer_path=None,
        img_size: int = 640,
        conf: float = 0.25,
        nms_iou: float = 0.45,
        classifier_size: int = 64,
        codeformer_weight: float = 0.5,
        device: str = "cpu",
    ):
        import keras

        self.conf = conf
        self.nms_iou = nms_iou
        self.classifier_size = classifier_size
        self.codeformer_weight = codeformer_weight

        self.classifier = keras.models.load_model(str(classifier_path))
        self._device = None
        self._detector = self._stride = self._det_img_size = None
        if detector_path is not None:
            self._load_detector(detector_path, img_size, device)
        self._codeformer = None
        if codeformer_path is not None:
            self._load_codeformer(codeformer_path, device)

    @property
    def has_detector(self) -> bool:
        return self._detector is not None

    # -- model loading -------------------------------------------------------------------

    def _load_detector(self, weights, img_size, device):
        import torch

        from yolo7_face.models.experimental import attempt_load
        from yolo7_face.utils.general import check_img_size
        from yolo7_face.utils.torch_utils import select_device

        torch.set_num_threads(1)  # few cores here; avoids extra thread-pool memory
        self._device = select_device(device)
        self._detector = attempt_load(str(weights), map_location=self._device)
        self._stride = int(self._detector.stride.max())
        self._det_img_size = check_img_size(img_size, s=self._stride)

    def _load_codeformer(self, ckpt, device):
        import torch

        from code_former.basicsr.utils.registry import ARCH_REGISTRY

        if self._device is None:
            self._device = torch.device(device)
        net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self._device)
        net.load_state_dict(torch.load(str(ckpt))["params_ema"])
        net.eval()
        self._codeformer = net

    # -- inference -----------------------------------------------------------------------

    def detect(self, img0):
        """Detect faces in a BGR image; return ``(boxes Nx4 xyxy, confidences N)``."""
        import numpy as np
        import torch

        from yolo7_face.utils.datasets import letterbox
        from yolo7_face.utils.general import non_max_suppression, scale_coords

        if self._detector is None:
            raise RuntimeError("No detector loaded (construct with detector_path to detect faces).")

        img = letterbox(img0, self._det_img_size, stride=self._stride, auto=False)[0]
        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))  # BGR->RGB, HWC->CHW
        img = torch.from_numpy(img).to(self._device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():  # inference only: don't build the autograd graph / retain activations
            pred = self._detector(img, augment=False)[0]
            pred = non_max_suppression(pred, self.conf, self.nms_iou, kpt_label=5)
        detection = pred[0]
        if len(detection):
            scale_coords(img.shape[2:], detection[:, :4], img0.shape, kpt_label=False)

        detection = np.array(detection)
        if detection.size == 0:
            return np.empty((0, 4)), np.empty((0,))
        return detection[:, :4], detection[:, 4]

    def classify(self, img0, boxes):
        """Classify the given face boxes in a BGR image; return softmax scores ``N x 2``."""
        import numpy as np

        if len(boxes) == 0:
            return np.empty((0, 2))
        batch = self._prepare_faces(img0, boxes)
        # Direct call instead of classifier.predict(): predict() retraces and leaks memory when
        # called in a loop with a varying batch size. Numerically identical for inference.
        return self.classifier(batch, training=False).numpy()

    def _prepare_faces(self, img0, boxes):
        """Crop each face (optionally CodeFormer-restored) to an RGB classifier-size batch."""
        import cv2
        import numpy as np

        size = self.classifier_size
        face_imgs = []
        for x_min, y_min, x_max, y_max in boxes:
            x_min, y_min = int(max(x_min, 0)), int(max(y_min, 0))
            x_max, y_max = int(min(x_max, img0.shape[1])), int(min(y_max, img0.shape[0]))
            face = img0[y_min:y_max, x_min:x_max]
            if self._codeformer is not None:
                face = self._restore_face(face)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (size, size))
            face_imgs.append(face)
        return np.reshape(face_imgs, [-1, size, size, 3]).astype("float32")

    def _restore_face(self, face_bgr):
        import cv2
        import torch
        from torchvision.transforms.functional import normalize

        from code_former.basicsr.utils import img2tensor, tensor2img

        img = cv2.resize(face_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
        tensor = img2tensor(img / 255.0, bgr2rgb=True, float32=True)
        normalize(tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        tensor = tensor.unsqueeze(0).to(self._device)
        with torch.no_grad():
            output = self._codeformer(tensor, w=self.codeformer_weight, adain=True)[0]
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
    """Run the pipeline over a COCO ground-truth dict and return COCO predictions.

    With ``use_gt_boxes`` the detector is skipped and the ground-truth boxes are classified
    instead (isolates the classifier from the detector).
    """
    import cv2
    import numpy as np
    from tqdm import tqdm

    if not use_gt_boxes and detector_path is None:
        raise ValueError("detector_path is required unless use_gt_boxes=True.")

    detector = MaskDetector(
        classifier_path,
        detector_path=None if use_gt_boxes else detector_path,
        codeformer_path=codeformer_path,
        img_size=img_size,
        conf=conf,
        nms_iou=nms_iou,
        classifier_size=classifier_size,
        codeformer_weight=codeformer_weight,
        device=device,
    )

    images_dir = Path(images_dir)
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
            boxes, confidences = detector.detect(img0)

        if len(boxes) == 0:
            continue

        results = detector.classify(img0, boxes)
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
