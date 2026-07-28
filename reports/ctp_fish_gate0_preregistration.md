# CTP-Fish Gate-0 preregistration

## Scope

Gate-0 is a feasibility audit, not a method experiment. It examines Fish4Knowledge
metadata, the data-derived F4K-16T subset, image-level and trajectory-level development
splits, split similarity, a plain ImageNet ResNet18 baseline, and optional official-mask
background correlation.

## Locked decisions

- `group_id = species_id::tracking_id_raw`; `tracking_id_raw` alone is never assumed global.
- F4K-16T includes every and only class with at least 15 independent groups.
- Development splits are 70/15/15 with seed 3407. The outer three-fold manifest is locked
  and cannot be read by Gate-0 training.
- The baseline is ImageNet-pretrained torchvision ResNet18 at 224x224, AdamW, cosine
  schedule, 40 epochs, and no class reweighting, resampling, masks, contrastive loss,
  prototype loss, or trajectory module.
- Image and trajectory metrics are both reported. Bootstrap samples groups, never frames.
- Background views are online and only run after official masks are successfully matched.

## Interpretive boundary

Trajectory-group isolation prevents direct frames from the same continuous trajectory from
crossing development sets. It does not establish camera-, location-, time-, or scene-level
independence. No similarity result may alter the F4K-16T rule.

## Stop rule

After Gate-0, report one of `PROCEED`, `PROCEED_WITH_CAUTION`, or `STOP` from the predefined
structural evidence. No contrastive, prototype, sampler, or other paper method is implemented
without a human decision.
