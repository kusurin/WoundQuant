"""Export tissue-area measurements for the configured workflow image."""
import numpy as np
import pandas as pd
from _common import settings as s

def main():
    output = s.output()
    stats = pd.read_csv(output / 'segmentation/per_image_sufficient_statistics.csv')
    scales = pd.read_csv(output / 'ruler/predictions.csv')
    rows = stats.merge(scales[['image_id', 'predicted_pixel_per_cm', 'status']],
                       on='image_id', how='left', validate='many_to_one', suffixes=('', '_ruler'))
    valid = (rows.status_ruler == 'valid') & np.isfinite(rows.predicted_pixel_per_cm) & (rows.predicted_pixel_per_cm > 0)
    rows['area_cm2'] = np.nan
    rows.loc[valid, 'area_cm2'] = rows.loc[valid, 'prediction_effective_pixels'] / rows.loc[valid, 'predicted_pixel_per_cm'] ** 2
    rows[['image_id', 'file_name', 'class_name', 'prediction_effective_pixels',
          'predicted_pixel_per_cm', 'area_cm2', 'status_ruler']].to_csv(output / 'areas.csv', index=False)

