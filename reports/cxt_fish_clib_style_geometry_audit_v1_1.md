# CLIB-style geometry audit v1.1

Status: **technical audit complete; formal training not authorized yet**.

The audit used the first two deterministically ordered records per frozen class
(32 images total) from
`outputs/cxt_fish/final_outer_manifests/fold_1/outer_train.csv`. It read only
outer-train RGB and official masks. No model prediction, inner-dev, outer-test,
calibration or official TEST data was read.

Manifest SHA-256:

```text
727d4ca7bd215a2231d86fc94d94d6c725d9ecd821c4a8c4e5cf720973d1f2da
```

## Fixed geometry results

| Quantity | Result |
|---|---:|
| records | 32 |
| classes | 16 |
| mean subject-box mask recall | 0.4991 |
| mean subject-box/mask IoU | 0.4748 |
| zero subject-mask recall | 0 / 32 |
| mask centroid inside subject box | 32 / 32 |
| corner mosaic containing any mask pixel | 0 / 32 |

The fixed contact sheet is stored locally at
`outputs/cxt_fish/clib_style_geometry_audit_v1_1/contact_sheet_32.png`.

## Decision boundary

The audit shows that the center crop is not empty and its centroid rule behaves
as intended, while the four-corner mosaic contains no annotated fish in this
sample. However, the average mask recall is only about 50%, and several contact
sheet examples retain only a central portion of the fish. This is not a model
result, but it is enough semantic uncertainty that the ratio interpretation
must not be treated as validated paper reproduction.

Therefore v1.1 remains a **deterministic CLIB-inspired interpretation** and
formal nine-cell training stays paused until the crop rule is either confirmed
from the original implementation/formula or explicitly accepted as this
pre-registered adaptation. No ratio was changed based on a classification
metric.
