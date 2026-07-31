"""Fit the preregistered train-only geometry logistic-regression diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics.classification import classification_metrics
from tools.io_utils import atomic_csv_dump, atomic_json_dump


def geometry_features(mask_path: str) -> list[float]:
    mask=np.asarray(Image.open(mask_path).convert("L")) > 0; height,width=mask.shape; ys,xs=np.where(mask)
    if not len(xs): raise ValueError(f"Empty mask: {mask_path}")
    left,right,top,bottom=xs.min(),xs.max()+1,ys.min(),ys.max()+1; box_w=right-left; box_h=bottom-top
    return [box_w/width, box_h/height, (box_w*box_h)/(width*height), (left+box_w/2)/width, (top+box_h/2)/height, box_w/box_h]


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--output-dir",default="outputs/cxt_fish/context_sanity/geometry_only"); args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text()); metadata=pd.read_csv(cfg["metadata_path"]); split=pd.read_csv(cfg["track_split_path"]); records=metadata.merge(split[["image_path","split"]],on="image_path",validate="one_to_one")
    train=records[records.split=="train"].reset_index(drop=True); val=records[records.split=="val"].reset_index(drop=True); class_ids=sorted(records.species_id.astype(str).unique()); mapping={label:index for index,label in enumerate(class_ids)}
    x_train=np.asarray([geometry_features(path) for path in train.mask_path]); y_train=np.asarray([mapping[str(label)] for label in train.species_id]); x_val=np.asarray([geometry_features(path) for path in val.mask_path]); y_val=np.asarray([mapping[str(label)] for label in val.species_id])
    model=Pipeline([("scale",StandardScaler()),("classifier",LogisticRegression(C=1.0,solver="lbfgs",max_iter=2000,random_state=407,multi_class="multinomial"))]); model.fit(x_train,y_train); prediction=model.predict(x_val)
    frame=pd.DataFrame({"image_path":val.image_path,"group_id":val.group_id,"target":y_val,"prediction":prediction}); frame["correct"]=frame.target==frame.prediction
    metrics=classification_metrics(y_val,prediction,list(range(len(class_ids)))) | {"class_ids":class_ids,"seed":407,"feature_schema":["w_over_W","h_over_H","box_area_over_image_area","x_center_over_W","y_center_over_H","w_over_h"],"partition":"val","internal_test_read":False,"outer_folds_read":False}
    root=Path(args.output_dir); atomic_csv_dump(frame,root/"per_image_val.csv"); atomic_json_dump(metrics,root/"metrics_val.json"); print(json.dumps(metrics,indent=2))


if __name__ == "__main__": main()
