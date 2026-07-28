"""Fixed nearest-neighbour similarity audit for image- versus trajectory-level splits."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.io_utils import atomic_csv_dump, atomic_json_dump


def phash(path: str) -> np.uint64:
    image=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    if image is None: raise ValueError(f"Unable to read image: {path}")
    coeff=cv2.dct(np.float32(cv2.resize(image,(32,32))))[:8,:8].reshape(-1); median=np.median(coeff[1:]); bits=coeff>median
    value=0
    for bit in bits: value=(value<<1)|int(bit)
    return np.uint64(value)


class _Images(Dataset):
    def __init__(self, paths: list[str], transform): self.paths=paths; self.transform=transform
    def __len__(self): return len(self.paths)
    def __getitem__(self,index): return self.transform(Image.open(self.paths[index]).convert("RGB")),index


def resnet_features(paths: list[str], device: torch.device) -> np.ndarray:
    weights=ResNet18_Weights.IMAGENET1K_V1; model=resnet18(weights=weights); model.fc=torch.nn.Identity(); model.eval().to(device)
    loader=DataLoader(_Images(paths,weights.transforms()),batch_size=128,shuffle=False,num_workers=2,pin_memory=device.type=="cuda")
    output=np.empty((len(paths),512),dtype=np.float32)
    with torch.no_grad():
        for batch,indices in loader:
            output[indices.numpy()]=model(batch.to(device,non_blocking=True)).cpu().numpy()
    output/=np.maximum(np.linalg.norm(output,axis=1,keepdims=True),1e-12); return output


def _nearest_phash(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    table=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8); best_distance=np.empty(len(test),np.int16); best_index=np.empty(len(test),np.int64)
    for start in range(0,len(test),128):
        block=test[start:start+128]; xor=np.bitwise_xor(block[:,None],train[None,:]).view(np.uint8).reshape(len(block),len(train),8); distance=table[xor].sum(axis=2); local=distance.argmin(axis=1); best_distance[start:start+len(block)]=distance[np.arange(len(block)),local]; best_index[start:start+len(block)]=local
    return best_index,best_distance


def _nearest_cosine(train: np.ndarray, test: np.ndarray, device: torch.device) -> tuple[np.ndarray,np.ndarray]:
    train_t=torch.from_numpy(train).to(device); index=[]; similarity=[]
    for start in range(0,len(test),512):
        scores=torch.from_numpy(test[start:start+512]).to(device)@train_t.T; values,indices=scores.max(dim=1); index.extend(indices.cpu().tolist()); similarity.extend(values.cpu().tolist())
    return np.asarray(index),np.asarray(similarity)


def audit_one(metadata: pd.DataFrame, split: pd.DataFrame, name: str, output: Path, device: torch.device) -> pd.DataFrame:
    joined=metadata.merge(split[["image_path","split"]],on="image_path",validate="one_to_one"); train=joined[joined.split=="train"].reset_index(drop=True); test=joined[joined.split=="test"].reset_index(drop=True)
    train_paths=train.image_path.tolist(); test_paths=test.image_path.tolist(); train_hash=np.asarray([phash(path) for path in train_paths],dtype=np.uint64); test_hash=np.asarray([phash(path) for path in test_paths],dtype=np.uint64)
    p_index,p_distance=_nearest_phash(train_hash,test_hash); train_feature=resnet_features(train_paths,device); test_feature=resnet_features(test_paths,device); f_index,f_similarity=_nearest_cosine(train_feature,test_feature,device)
    result=pd.DataFrame({"test_image_path":test_paths,"test_group_id":test.group_id,"test_species_id":test.species_id,"phash_train_image_path":[train_paths[index] for index in p_index],"phash_train_group_id":train.group_id.iloc[p_index].to_numpy(),"phash_distance":p_distance,"feature_train_image_path":[train_paths[index] for index in f_index],"feature_train_group_id":train.group_id.iloc[f_index].to_numpy(),"feature_cosine_similarity":f_similarity})
    result["phash_same_group"] = result.test_group_id==result.phash_train_group_id; result["feature_same_group"] = result.test_group_id==result.feature_train_group_id
    atomic_csv_dump(result,output/f"nearest_neighbor_{name}.csv",overwrite=True); return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--metadata",required=True); parser.add_argument("--image-split",required=True); parser.add_argument("--track-split",required=True); parser.add_argument("--output-dir",default="outputs/gate0/similarity"); parser.add_argument("--dry-run",action="store_true"); args=parser.parse_args()
    if args.dry_run: print("Would run pHash and fixed ImageNet ResNet18 nearest-neighbour audit."); return
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True); metadata=pd.read_csv(args.metadata); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image=audit_one(metadata,pd.read_csv(args.image_split),"image_level",output,device); track=audit_one(metadata,pd.read_csv(args.track_split),"track_level",output,device)
    summary={"device":str(device),"image_level":{"mean_phash_distance":float(image.phash_distance.mean()),"mean_feature_cosine":float(image.feature_cosine_similarity.mean()),"phash_same_track_fraction":float(image.phash_same_group.mean()),"feature_same_track_fraction":float(image.feature_same_group.mean())},"track_level":{"mean_phash_distance":float(track.phash_distance.mean()),"mean_feature_cosine":float(track.feature_cosine_similarity.mean()),"phash_same_track_fraction":float(track.phash_same_group.mean()),"feature_same_track_fraction":float(track.feature_same_group.mean())}}
    atomic_json_dump(summary,output/"similarity_summary.json",overwrite=True); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
