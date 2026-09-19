"""Generate segmentation and ruler inputs for workflow and performance panels."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from . import settings as s

def test_ids():
    split = json.loads(s.data_path('split').read_text(encoding='utf-8'))
    return [int(value) for value in split['splits']['test']['image_ids']]

def example_ids():
    data = json.loads(s.data_path('annotations').read_text(encoding='utf-8'))
    name = s.current()['example_image']
    matches = [int(item['id']) for item in data['images'] if item['file_name'] == name]
    if len(matches) != 1:
        raise ValueError(f'Example must identify exactly one annotated image: {name}')
    return matches

def write_rows(file, rows, fields):
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def segmentation(ids):
    from seg_discrete_overlap import train_eval as runtime
    cfg = s.current()
    device = runtime.resolve_device(cfg.get('device', 'auto'))
    model, payload = runtime.load_inference_model(s.path(cfg['checkpoint']), device)
    dataset = runtime.CocoFullResolutionSegmentationEvaluationDataset(
        s.data_path('annotations'), s.data_path('images'), image_ids=ids, normal_name='normal')
    if dataset.mapping.to_dict() != payload['mapping']:
        raise ValueError('Checkpoint and dataset class mappings differ.')
    output = s.output() / 'segmentation'
    output.mkdir(parents=True, exist_ok=True)
    size = int(cfg.get('image_size', payload['args'].get('image_size', 512)))
    metrics = runtime.export_predictions(model, dataset, device, output, size)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    sufficient_statistics(output, dataset.mapping.channel_to_name)

def sufficient_statistics(output, names):
    manifest = json.loads((output / 'mask_manifest.json').read_text(encoding='utf-8'))
    rows = []
    for image in manifest['images']:
        with np.load(output / image['prediction']['probability_archive']) as archive:
            prediction = archive['probabilities'].astype(np.float64)
        with np.load(output / image['ground_truth']['probability_archive']) as archive:
            target = archive['probabilities'].astype(np.float64)
        for channel, name in enumerate(names):
            p, y = prediction[channel], target[channel]
            p_mass, y_mass = float(p.sum()), float(y.sum())
            error, denominator = abs(p_mass - y_mass), p_mass + y_mass
            both_zero = denominator <= 1e-12
            rows.append({'image_id': int(image['image_id']), 'file_name': image['file_name'],
                         'height': int(image['height']), 'width': int(image['width']),
                         'channel': channel, 'class_name': name,
                         'intersection': float((p*y).sum()), 'prediction_square': float((p*p).sum()),
                         'target_square': float((y*y).sum()), 'prediction_effective_pixels': p_mass,
                         'target_effective_pixels': y_mass, 'absolute_error_effective_pixels': error,
                         'area_denominator_effective_pixels': denominator,
                         'area_smape_percent': 0.0 if both_zero else 200*error/denominator,
                         'area_smape_status': 'both_zero_defined_as_zero' if both_zero else 'defined',
                         'status': 'valid'})
    if not rows:
        raise ValueError('No exported observations.')
    write_rows(output / 'per_image_sufficient_statistics.csv', rows, list(rows[0]))

def ruler(ids, *, config=None, output=None):
    from woundquant.ruler import RulerNet
    cfg = config or s.current()
    destination = (output or s.output()) / 'ruler'
    source = json.loads(s.data_path('annotations').read_text(encoding='utf-8'))
    images = {int(item['id']): item for item in source['images']}
    model = RulerNet(s.path(cfg['ruler_model']))
    predictions, points = [], []
    for image_id in ids:
        item = images[image_id]
        with Image.open(s.data_path('images') / item['file_name']) as image:
            scale, detected, status = model.predict(image)
        predictions.append({'image_id': image_id, 'file_name': item['file_name'],
                            'predicted_pixel_per_cm': scale, 'detected_point_count': len(detected), 'status': status})
        points.extend({'image_id': image_id, 'file_name': item['file_name'], 'point_index': i,
                       'x_original_pixels': x, 'y_original_pixels': y} for i, (x, y) in enumerate(detected))
    write_rows(destination / 'predictions.csv', predictions,
               ['image_id', 'file_name', 'predicted_pixel_per_cm', 'detected_point_count', 'status'])
    write_rows(destination / 'points.csv', points,
               ['image_id', 'file_name', 'point_index', 'x_original_pixels', 'y_original_pixels'])

