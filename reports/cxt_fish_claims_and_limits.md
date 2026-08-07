# CXT-Fish claims and limits (frozen)

This file freezes the claims permitted in the final manuscript and the claims
that must not be made after the outer confirmation.

## Permitted claims

- Random image-level splits can be optimistic when correlated frames from the
  same group are shared across development and evaluation; group-disjoint
  evaluation is stricter.
- F4K-16T models show measurable dependence on contextual information under
  the fixed foreground and context-swap tests.
- On the frozen ResNet18 outer confirmation, foreground-sufficiency training
  raises cross-class-swap macro-F1 by 7.24 percentage points in the
  pre-registered cell-mean summary.
- ResNet18 clean macro-F1 changes by -0.21 percentage points on average; a
  clean-accuracy improvement claim is not permitted.
- DAR-flip is lower for CXT-Fish in all nine paired ResNet18 cells.
- Masks are used during training-view construction only; inference uses an
  ordinary RGB image and one model forward.
- MobileNetV3-Large shows the same foreground and cross-swap robustness
  direction, but with a 2.94-point clean macro-F1 decrease; it is an
  architecture-sensitivity supplement.
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

- Route C is a fixed two-stage mechanism control, not a faithful CLIB
  reproduction and not an exhaustive contrastive-learning comparison.
- MobileNetV3 uses one registered seed and is not a second main confirmation.
- Group IDs are trajectory/group proxies, not verified biological individual
  identities.
- The official Fish4Knowledge test set was not accessed.
- These claims are based on frozen outputs; no further model experiment is
  authorized in this repository version.
