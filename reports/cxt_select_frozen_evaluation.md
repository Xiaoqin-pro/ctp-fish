# CXT-Select frozen evaluation

## Scope

This is a zero-training exploratory evaluation on the current track-level
validation partition. The partition is explicitly not an independent
confirmation set because it has been used during development. Internal test,
outer folds, and official test were not accessed. F1 remains the prediction
model; F3 is used only as a complementary disagreement source.

Three frozen pairs were evaluated: F1/F3 at seeds 3407, 2026, and 17. The
evaluation cached original, foreground, same-class swap, and cross-class swap
logits for both models. Only original RGB F1/F3 outputs entered risk scores.

## Primary comparison

The primary comparison was JS divergence versus F1 MSP risk. Higher risk means
the sample is reviewed/rejected first. The following values are per-seed.

| Seed | Score | Clean AUROC | Clean AUPRC | Clean AURC | Cross-swap AUPRC | DAR-flip AUPRC |
|---:|---|---:|---:|---:|---:|---:|
| 3407 | F1 MSP | 0.9729 | 0.3389 | 0.000162 | 0.4100 | 0.3277 |
| 3407 | JS | 0.9539 | 0.1251 | 0.000278 | 0.3950 | 0.3101 |
| 2026 | F1 MSP | 0.9676 | 0.4097 | 0.000239 | 0.4263 | 0.2972 |
| 2026 | JS | 0.9553 | 0.3151 | 0.000327 | 0.4026 | 0.2732 |
| 17 | F1 MSP | 0.9793 | 0.3915 | 0.000184 | 0.4006 | 0.2705 |
| 17 | JS | 0.9780 | 0.3792 | 0.000194 | 0.3809 | 0.2584 |

JS has higher clean AURC (worse) on all three seeds and lower cross-swap and
DAR-flip AUPRC on all three seeds. Label disagreement is substantially weaker
than both continuous scores and is not a viable replacement.

## Paired trajectory bootstrap

The bootstrap resampled `(species_id, group_id)` clusters within species for
2,000 paired replicates per seed. Differences are JS minus MSP, so negative
cross-event AUPRC differences and positive AURC differences favor MSP.

| Seed | Clean AURC difference | Cross-swap AUPRC difference | DAR-flip AUPRC difference |
|---:|---:|---:|---:|
| 3407 | +0.000113 (CI −0.000061, +0.000349) | −0.0147 (CI −0.0382, +0.0099) | −0.0173 (CI −0.0373, +0.0030) |
| 2026 | +0.000089 (CI −0.000066, +0.000247) | −0.0234 (CI −0.0362, −0.0108) | −0.0238 (CI −0.0351, −0.0132) |
| 17 | +0.000010 (CI −0.000033, +0.000045) | −0.0197 (CI −0.0322, −0.0072) | −0.0128 (CI −0.0221, −0.0044) |

The direction is not a JS improvement. Clean AURC does not favor JS, and the
context-event detection metrics favor MSP on two or three seeds.

## Decision

CXT-Select does **not** pass its exploratory gate. The disagreement between F1
and F3 is not a reliable risk score under this protocol; it does not improve
clean selective ranking and does not consistently detect cross-swap or
donor-attraction events. No score weighting, temperature tuning, third model,
new loss, or new risk head is authorized.

This closes the method-development sequence. The positive result that remains
frozen is F1 foreground-sufficiency training together with the track-aware and
context-swap evaluation protocol. The next work is manuscript consolidation,
statistical presentation, and—only after the full paper protocol is frozen—the
single final external/test evaluation.

## Provenance

- Branch: `experiment/cxt-fish-phase3-selective`.
- Protocol commit: `36096ff`.
- Evaluation outputs: local `outputs/cxt_fish/phase3_selective/`.
- Compact committed outputs: `experiments/cxt_select_detection_metrics.csv` and `experiments/cxt_select_bootstrap.json`.
- Current validation: exploratory only.
- Internal test accessed: false.
- Outer folds accessed: false.
- Official test accessed: false.
