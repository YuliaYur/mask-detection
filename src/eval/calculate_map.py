from podm.metrics import get_pascal_voc_metrics, MetricPerClass, get_bounding_boxes
from podm import coco_decoder


with open('annotation/school/annotation.json') as fp:
    gold_dataset = coco_decoder.load_true_object_detection_dataset(fp)
with open('prediction/school/prediction_both82_upd.json') as fp:
    pred_dataset = coco_decoder.load_pred_object_detection_dataset(fp, gold_dataset)

gt_BoundingBoxes = get_bounding_boxes(gold_dataset)
pd_BoundingBoxes = get_bounding_boxes(pred_dataset)
results = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, .5)

for cls, metric in results.items():
    print('Class', cls)
    label = metric.label
    print('ap', metric.ap)
    # print('precision', metric.precision)
    # print('interpolated_recall', metric.interpolated_recall)
    # print('interpolated_precision', metric.interpolated_precision)
    print('tp', metric.tp)
    print('fp', metric.fp)
    print('num_groundtruth', metric.num_groundtruth)
    print('num_detection', metric.num_detection)
    print()


mAP = MetricPerClass.mAP(results)
print(mAP)
