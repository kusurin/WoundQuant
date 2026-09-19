"""Select representative image IDs by physical-area error percentile."""
import numpy as np
import pandas as pd
from _common import settings as s

def _select_percentile_cases(records: pd.DataFrame) -> pd.DataFrame:
    valid = records[records.analysis_status == "valid"]
    scores = valid[["image_id", "file_name", "total_area_smape_percent"]].drop_duplicates()
    scores = scores.sort_values("total_area_smape_percent").reset_index(drop=True)
    targets = s.current()["percentiles"]
    if len(scores) < len(targets):
        raise ValueError("Not enough valid images to select distinct percentile cases.")
    selected_indices = set()
    rows = []
    for percentile in targets:
        target_value = float(np.percentile(scores.total_area_smape_percent, percentile))
        candidates = (scores.total_area_smape_percent - target_value).abs().sort_values().index
        selected_index = next(int(index) for index in candidates if int(index) not in selected_indices)
        selected_indices.add(selected_index)
        row = scores.loc[selected_index].to_dict()
        row.update({"target_percentile": percentile, "target_smape_percent": target_value})
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    records = pd.read_csv(s.output("fig4_performance") / "area_errors.csv")
    selected = _select_percentile_cases(records)
    output = s.output()
    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "selected_cases.csv", index=False)
    records[records.image_id.isin(selected.image_id)].to_csv(output / "selected_case_areas.csv", index=False)
