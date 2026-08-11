# CXT-Fish claims and limits (frozen)

This file freezes the claims permitted in the final manuscript and the claims
that must not be made after the frozen post-development confirmation.

## Permitted claims

- Random image-level splits can be optimistic when correlated frames from the
  same group are shared across development and evaluation; group-disjoint
  evaluation is stricter.
- F4K-16T models show measurable dependence on contextual information under
  the fixed foreground and donor-context composite intervention tests.
- On the frozen ResNet18 post-development, group-disjoint confirmation
  analysis, foreground-sufficiency training
  raises cross-class-swap macro-F1 by 7.24 percentage points in the
  frozen equal-weight cell-mean summary. The frozen outer protocol treated
  cross-class donor-context-composite macro-F1 as a secondary robustness
  metric; formal interval reporting is restricted to this final robustness
  outcome.
- Across five post-hoc deterministic donor realizations under the same frozen
  donor-construction rule, the mean CXT-Fish--F0 cross-composite effect
  remained positive (+7.74 to +8.84 percentage points). These sensitivity
  analyses do not replace the frozen seed-3407 result.
- In the post-hoc F0-2RGB control, CXT-Fish exceeded an ordinary-RGB
  two-view supervision-matched control by 6.75 percentage points on
  cross-composite macro-F1 (95% exploratory interval, +5.66 to +7.85
  percentage points), indicating that generic duplicated supervised exposure
  alone does not explain the full robustness gain.
- ResNet18 clean macro-F1 changes by -0.21 percentage points on average; a
  clean-accuracy improvement claim is not permitted.
- DAR-flip is lower for CXT-Fish in all nine paired ResNet18 cells.
- Masks are used during training-view construction only; inference uses an
  ordinary RGB image and one model forward.
- MobileNetV3-Large shows average foreground and donor-context composite
  improvement, with heterogeneous cross-swap effects across folds and a
  2.94-point clean
  macro-F1 decrease; it is an architecture-sensitivity supplement.
- The clean--robustness trade-off is architecture-sensitive.

## Prohibited claims

- Universal accuracy improvement.
- Universal or significant tail-class improvement.
- Backbone-agnostic or cross-backbone validity.
- Clean performance preserved on every backbone.
- Beating CLIB.
- Contrastive learning is generally ineffective.
- Background bias has been eliminated.
- Cross-dataset, cross-camera, or cross-sea generalization.

## Evidence boundaries

- The historical development split was drawn from the same corpus as the
  outer folds; some historical development groups necessarily occur in the
  later outer-test partitions. The outer result is therefore a frozen,
  post-development group-disjoint confirmation analysis, not a fully blind
  nested evaluation or method-selection-independent test.
- Route C is a fixed two-stage mechanism control, not a faithful CLIB
  reproduction and not an exhaustive contrastive-learning comparison.
- MobileNetV3 uses one pre-specified/frozen seed and is not a second main
  confirmation.
- Group IDs are trajectory/group proxies, not verified biological individual
  identities.
- The official Fish4Knowledge test set was not accessed.
- These claims are based on frozen outputs; no further model experiment is
  authorized in this repository version.
