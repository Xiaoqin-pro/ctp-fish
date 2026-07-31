"""Evaluate one frozen Gate-0 checkpoint without accessing locked outer folds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.f4k_dataset import F4KDataset
from metrics.classification import classification_metrics
from metrics.track_metrics import cluster_bootstrap_mean, track_balanced_accuracy
from models.resnet_classifier import build_resnet18
from tools.io_utils import atomic_csv_dump, atomic_json_dump


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--checkpoint",required=True); parser.add_argument("--split",choices=["image","track"],required=True); parser.add_argument("--partition",choices=["val","test"],default="test"); parser.add_argument("--mask-variant",choices=["original","foreground_only","background_only"],default="original"); parser.add_argument("--evaluation-name"); parser.add_argument("--dry-run",action="store_true"); parser.add_argument("--overwrite",action="store_true"); args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    if args.dry_run: print({"checkpoint":args.checkpoint,"split":args.split,"partition":args.partition,"mask_variant":args.mask_variant,"outer_folds_locked":True}); return
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False); metadata=pd.read_csv(cfg["metadata_path"]); manifest=pd.read_csv(cfg[f"{args.split}_split_path"])
    records=metadata.merge(manifest[["image_path","split"]],on="image_path",validate="one_to_one"); records=records[records.split==args.partition].reset_index(drop=True); class_ids=state["class_ids"]
    transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(int(cfg["image_size"])),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]); dataset=F4KDataset(records,transform,mask_variant=args.mask_variant,class_ids=class_ids)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); loader=DataLoader(dataset,batch_size=int(cfg["batch_size"]),shuffle=False,num_workers=2,pin_memory=device.type=="cuda")
    model=build_resnet18(len(class_ids)); model.load_state_dict(state["model"]); model.to(device).eval(); target=[]; prediction=[]; paths=[]; groups=[]
    with torch.no_grad():
        for images,labels,image_paths,group_ids in loader:
            output=model(images.to(device)); target.extend(labels.tolist()); prediction.extend(output.argmax(1).cpu().tolist()); paths.extend(image_paths); groups.extend(group_ids)
    frame=pd.DataFrame({"image_path":paths,"group_id":groups,"target":target,"prediction":prediction}); frame["correct"]=frame.target==frame.prediction
    track_accuracy,per_track=track_balanced_accuracy(frame); ci=cluster_bootstrap_mean(per_track,seed=3407); metrics=classification_metrics(np.asarray(target),np.asarray(prediction),list(range(len(class_ids)))) | {"track_balanced_accuracy":track_accuracy,"track_balanced_accuracy_ci95":list(ci),"class_ids":class_ids,"partition":args.partition}
    if args.evaluation_name and Path(args.evaluation_name).name != args.evaluation_name: raise ValueError("evaluation-name must be a simple directory name.")
    output=Path(args.checkpoint).parent/(f"evaluation_{args.evaluation_name}" if args.evaluation_name else "evaluation")
    metrics["mask_variant"]=args.mask_variant; metrics["evaluation_name"]=args.evaluation_name
    atomic_csv_dump(frame,output/f"per_image_{args.partition}.csv",args.overwrite); atomic_csv_dump(per_track,output/f"per_track_{args.partition}.csv",args.overwrite); atomic_json_dump(metrics,output/f"metrics_{args.partition}.json",args.overwrite); print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
