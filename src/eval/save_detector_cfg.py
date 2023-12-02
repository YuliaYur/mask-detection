import json
from pathlib import Path


path_to_annotation = Path('prediction/aizoo/prediction_both82.json')
with open(path_to_annotation, 'r') as f:
    ann = json.load(f)

for x in ann:  # ['annotations']:
    x['category_id'] = 0
    # x['score'] = 1

with open(path_to_annotation.parent / (path_to_annotation.stem + '_detector.json'), 'w') as f:
    json.dump(ann, f, indent=4)
