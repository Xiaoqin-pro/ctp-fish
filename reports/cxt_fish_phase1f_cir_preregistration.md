# CIR-Fish Phase 1F: zero-training gate

This pilot tests **Cross-Track Context-Residual Intervention (CIR-Fish)**
without training a new model. The frozen Phase 1C F1 checkpoint is used on the
validation partition only. Internal test, calibration, and outer folds remain
locked.

For donor image (j), the detached residual is

\[
c_j = z_j^{orig} - z_j^{fg}.
\]

For recipient (i), the diagnostic injected logits are

\[
\tilde z_{i\leftarrow j}=z_i^{fg}+\operatorname{sg}(c_j).
\]

The gate also runs the existing fixed raw-RGB cross-class context swap. Because
the official masks are single-fish masks, this is described as **non-target
context/interference exchange**, not as a pure fish-free background swap.

## Frozen gate checks

1. Donor residual specificity: one-vs-rest AUROC of donor-class residual scores,
   reported per class and overall.
2. Injection response: recipient accuracy, donor attraction, conditional
   donor attraction, donor-vs-recipient margin, and prediction agreement.
3. Deterministic residual shuffle control preserving the residual distribution.
4. Intervention alignment:
   \[
   q^{real}_{ij}=[z^{swap}_{ij,y_j}-z^{swap}_{ij,y_i}]
   -[z^{fg}_{i,y_j}-z^{fg}_{i,y_i}],
   \quad q^{CIR}_{ij}=c_j[y_j]-c_j[y_i].
   \]
   Spearman alignment and AUROC for real DAR-flip events are recorded.
5. Paired scene-cluster bootstrap uses recipient `group_id` clusters and 1,000
   fixed replicates with seed 3407.

## Pass rule

The zero-training gate passes only if all fixed checks pass:

- overall residual donor AUROC >= 0.60;
- CIR score AUROC for real DAR-flip >= 0.60;
- q-CIR/q-real Spearman correlation > 0;
- true-minus-shuffle donor-attraction bootstrap 95% CI lower bound > 0;
- at least half of the 23 declared classes meet residual AUROC >= 0.60;
- at least 95% of fixed pairs are valid.

No I1/I2 training is authorized by this document. If the gate fails, the CIR
method is closed without changing the frozen CXT-Fish results.
