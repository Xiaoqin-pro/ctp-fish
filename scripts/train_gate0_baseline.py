"""Plain ResNet18 Gate-0 baseline; locked outer folds are always rejected."""
from __future__ import annotations
import argparse, csv, json, os, random, sys, tempfile
from functools import partial
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.f4k_dataset import F4KDataset
from models.resnet_classifier import build_resnet18
from tools.reproducibility import seed_everything


def _atomic_torch_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent, suffix=".tmp") as handle: temporary = Path(handle.name)
    torch.save(payload, temporary); os.replace(temporary, destination)


def _seed_worker(worker_id: int, seed: int) -> None:
    worker_seed = seed + worker_id
    random.seed(worker_seed); np.random.seed(worker_seed)


def _transforms(image_size: int):
    normalize=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    train=transforms.Compose([transforms.RandomResizedCrop(image_size),transforms.RandomHorizontalFlip(),transforms.ColorJitter(.1,.1,.1,.05),transforms.ToTensor(),normalize])
    evaluation=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(image_size),transforms.ToTensor(),normalize])
    return train,evaluation


def _run_epoch(model, loader, optimizer, scaler, device, amp: bool, train: bool) -> tuple[float,float]:
    model.train(train); loss_sum=correct=count=0; criterion=nn.CrossEntropyLoss()
    for images, targets, _, _ in loader:
        images,targets=images.to(device,non_blocking=True),targets.to(device,non_blocking=True)
        if train: optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type=="cuda"):
            logits=model(images); loss=criterion(logits,targets)
        if not torch.isfinite(loss): raise FloatingPointError("Non-finite training loss.")
        if train: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        loss_sum += loss.item()*len(targets); correct += (logits.argmax(1)==targets).sum().item(); count += len(targets)
    return loss_sum/count,correct/count

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--split",choices=["image","track"],required=True); parser.add_argument("--seed",type=int); parser.add_argument("--resume"); parser.add_argument("--outer-folds"); parser.add_argument("--mask-variant",choices=["original","foreground_only","background_only"],default="original"); parser.add_argument("--run-name"); parser.add_argument("--dry-run",action="store_true"); args=parser.parse_args()
    if args.outer_folds: raise ValueError("Outer evaluation folds are locked during Gate-0.")
    cfg=yaml.safe_load(Path(args.config).read_text()); seed=int(args.seed if args.seed is not None else cfg["seeds"][0]); seed_everything(seed)
    if args.dry_run: print({"model":"torchvision ResNet18 ImageNet", "split":args.split, "epochs":cfg["epochs"], "seed":seed, "mask_variant":args.mask_variant, "forbidden_methods":"no reweighting, no trajectory sampler, no contrastive/prototype method"}); return
    metadata=pd.read_csv(cfg["metadata_path"]); split_path=Path(cfg[f"{args.split}_split_path"]); split=pd.read_csv(split_path)
    if split_path.resolve()==Path(cfg["outer_folds_path"]).resolve(): raise ValueError("Outer evaluation folds are locked during Gate-0.")
    records=metadata.merge(split[["image_path","split"]],on="image_path",validate="one_to_one"); class_ids=sorted(records.species_id.astype(str).unique()); train_tf,eval_tf=_transforms(int(cfg["image_size"]))
    train_set=F4KDataset(records[records.split=="train"],train_tf,mask_variant=args.mask_variant,class_ids=class_ids); val_set=F4KDataset(records[records.split=="val"],eval_tf,mask_variant=args.mask_variant,class_ids=class_ids)
    batch=int(cfg["batch_size"]); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); generator=torch.Generator().manual_seed(seed)
    common=dict(batch_size=batch, num_workers=2, pin_memory=device.type=="cuda", worker_init_fn=partial(_seed_worker, seed=seed), generator=generator)
    train_loader=DataLoader(train_set,shuffle=True,**common); val_loader=DataLoader(val_set,shuffle=False,**common)
    model=build_resnet18(len(class_ids)).to(device); optimizer=AdamW(model.parameters(),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"])); scheduler=CosineAnnealingLR(optimizer,T_max=int(cfg["epochs"])); scaler=torch.amp.GradScaler(device.type,enabled=bool(cfg["amp"]) and device.type=="cuda")
    run_name=args.run_name or f"resnet18_{args.split}_seed{seed}"
    if Path(run_name).name != run_name: raise ValueError("run-name must be a simple directory name.")
    output=Path("outputs/gate0/background_audit")/run_name if args.run_name else Path("outputs/gate0/baselines")/run_name
    output.mkdir(parents=True,exist_ok=True); history=[]; best=-1.; patience=0; start=1
    if args.resume:
        state=torch.load(args.resume,map_location=device,weights_only=False); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"]); history=state["history"]; best=state["best_val_accuracy"]; start=int(state["epoch"])+1
    for epoch in range(start,int(cfg["epochs"])+1):
        train_loss,train_acc=_run_epoch(model,train_loader,optimizer,scaler,device,bool(cfg["amp"]),True); val_loss,val_acc=_run_epoch(model,val_loader,optimizer,scaler,device,bool(cfg["amp"]),False); scheduler.step(); row={"epoch":epoch,"train_loss":train_loss,"train_accuracy":train_acc,"val_loss":val_loss,"val_accuracy":val_acc,"lr":optimizer.param_groups[0]["lr"]}; history.append(row)
        if val_acc>best:
            best=val_acc; patience=0; _atomic_torch_save({"model":model.state_dict(),"class_ids":class_ids,"epoch":epoch,"best_val_accuracy":best,"config":cfg},output/"best.pt")
        else: patience+=1
        _atomic_torch_save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"epoch":epoch,"history":history,"best_val_accuracy":best},output/"last.pt")
        pd.DataFrame(history).to_csv(output/"training_curve.csv",index=False)
        print(json.dumps(row))
        if patience>=int(cfg["early_stopping_patience"]): break
    (output/"run_metadata.json").write_text(json.dumps({"split":args.split,"seed":seed,"device":str(device),"batch_size":batch,"class_ids":class_ids,"mask_variant":args.mask_variant,"audit_run":bool(args.run_name)},indent=2))
if __name__=="__main__": main()
