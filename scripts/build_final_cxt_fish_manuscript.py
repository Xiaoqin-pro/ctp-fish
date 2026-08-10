"""Build the final IMTS-ready manuscript from the supplied DOCX and frozen artifacts."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"E:\xiazai\google\CXT-Fish_IMTS_Final_ConstructValidity_Manuscript.docx")
OUT = ROOT / "paper" / "CXT-Fish_IMTS_Submission_Ready_v5.docx"
FIG_DIR = ROOT / "reports" / "figures" / "manuscript_submission_v4"


def replace_paragraph(d: Document, index: int, text: str) -> None:
    p = d.paragraphs[index]
    p.text = text


def insert_after(p: Paragraph, text: str, style: str = "Normal") -> Paragraph:
    new_p = OxmlElement("w:p")
    p._p.addnext(new_p)
    new_para = Paragraph(new_p, p._parent)
    new_para.style = style
    new_para.add_run(text)
    return new_para


def replace_text_in_runs(d: Document, old: str, new: str) -> None:
    """Replace a short metadata string without disturbing paragraph layout."""
    parts = list(d.paragraphs)
    parts.extend(cell.paragraphs for table in d.tables for row in table.rows for cell in row.cells)
    for section in d.sections:
        parts.extend(section.header.paragraphs)
        parts.extend(section.footer.paragraphs)
    for group in parts:
        for p in group if isinstance(group, list) else [group]:
            for run in p.runs:
                if old in run.text:
                    run.text = run.text.replace(old, new)


def build() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE, OUT)
    d = Document(OUT)

    # Updated correspondence metadata requested for the submission-ready version.
    replace_text_in_runs(d, "3497925919@qq.com", "xiao.qin@sdust.edu.cn")

    replace_paragraph(d, 5,
        "Underwater fish-recognition benchmarks derived from video contain correlated observations that can blur the distinction between frame recognition and generalization to new recorded groups. We audit this issue in Fish4Knowledge and construct F4K-16T, comprising 27,133 images from 16 species selected under a fixed minimum-recorded-group criterion. The protocol combines recorded-group-disjoint evaluation with foreground and donor-context interventions that probe predictive non-subject/contextual signals. We instantiate a simple label-level foreground-sufficiency objective: a shared classifier predicts the species from both the ordinary RGB image and a mask-derived view with suppressed surrounding detail, while inference remains one ordinary RGB image and one forward pass. After the method definitions were fixed, we conducted a three-fold group-disjoint re-evaluation on the same Fish4Knowledge-derived corpus previously used during method development. CXT-Fish increased cross-class donor-context-composite macro-F1 from 0.5189 to 0.5913. The equal-weight paired improvement was 7.24 percentage points, with a conditional paired group-cluster bootstrap 95% interval of 6.13–8.20 points; mean clean macro-F1 changed from 0.9577 to 0.9556. Post-hoc controls showed positive effects across five additional deterministic donor assignments, a +6.83-point advantage over an ordinary-RGB two-view control (exploratory interval, +5.71–+7.87), and a +10.48-point difference after donor-subject suppression under the same frozen pairings (exploratory interval, +9.38–+11.61). These results support foreground sufficiency as a targeted intervention for the tested synthetic, mask-defined context perturbations rather than as a general clean-accuracy, context-invariance, or external-domain solution.")

    replace_paragraph(d, 14,
        "The contributions are threefold. First, we provide a quantitative, integrated reliability audit for Fish4Knowledge-derived recognition, combining recorded-group-disjoint evaluation, nearest-neighbour correlation audits, group-aware summaries, and controlled context interventions. Second, we instantiate a simple label-level foreground-sufficiency objective for underwater fish recognition: segmentation-derived information is privileged during training, but the auxiliary view and mask disappear at inference. This is an application-specific instantiation, not a claim to introduce a new learning paradigm. Third, we provide a layered evidence package in which the main same-corpus frozen group-disjoint re-evaluation is separated from post-hoc donor-realization, two-view, donor-subject-suppression, architecture-sensitivity, and mechanism-stress controls. The main ResNet18 result is a +7.24-point cross-class donor-context-composite macro-F1 change with a −0.21-point clean macro-F1 change; the paper therefore makes a targeted robustness claim rather than a general accuracy claim.")
    replace_paragraph(d, 16,
        "Fig. 1 Study design and evidence hierarchy. The solid path is the main evidence chain from correlation audit to the same-corpus frozen group-disjoint re-evaluation. Dashed branches denote post-hoc validity controls and boundary analyses conducted after the main method definitions were fixed; they are not independent validation or a second blind evaluation. CXT-Fish changes training, not the ordinary RGB inference pathway.")

    replace_paragraph(d, 23,
        "Training-time privileged information is a broader learning setting in which auxiliary information is available during learning but absent at prediction (Vapnik and Vashist 2009; Lopez-Paz et al. 2016). Foreground masking, subject–context recomposition, and background perturbation are established ways to probe or reduce contextual reliance (Torralba and Efros 2011; Geirhos et al. 2020; Xiao et al. 2021). In underwater fish recognition, CLIB is a close domain-specific contrastive approach that uses subject/background views to reduce background influence (Yan et al. 2024). We therefore do not claim the first use of segmentation-derived training information, foreground/context manipulation, or contrastive learning. The narrower distinction here is a label-level sufficiency objective coupled to a recorded-group-aware reliability protocol and a layered construct-validity package. The objective does not force original and foreground representations to coincide, and the paper does not infer that label-level supervision is generally superior to contrastive learning.")
    replace_text_in_runs(d, "In underwater fish recognition, CLIB", "Earlier live-fish recognition work already used trajectory-aware separation to keep images from one trajectory sequence out of both training and testing, together with trajectory-level voting (Huang et al. 2015). In underwater fish recognition, CLIB")
    replace_paragraph(d, 33, "3.2 Evidence stages and same-corpus group-disjoint re-evaluation")
    replace_paragraph(d, 35,
        "For each outer fold, held-out evaluation groups are disjoint from all groups used to fit the model and select its checkpoint in that fold. Test-fold images are never used for optimization or checkpoint selection. We therefore call the main experiment a same-corpus frozen group-disjoint re-evaluation: it estimates robustness across held-out recorded groups after the method definitions were fixed, but it is not method-development-independent holdout validation, a fully blind nested evaluation, or independent external validation. Historical development and the final folds are partitions of the same Fish4Knowledge-derived corpus, so this evidence boundary is stated explicitly.")
    replace_paragraph(d, 35,
        "The three outer folds are loaded from the frozen group manifest using its fixed split seed. For each fold, the designated test groups are held out first. The remaining groups are partitioned into outer-train and fold-local inner-development groups by stable hash ordering separately within each species, using the fixed 0.15 inner-development fraction with at least one development group and at least one fitting group per species. Outer-train, inner-development, and outer-test groups are mutually disjoint, and both fitting and inner-development manifests contain all 16 frozen species. The same fold-local manifests are shared by F0 and CXT-Fish; model seeds do not change the split. Held-out evaluation groups are therefore disjoint from all groups used to fit the model and select its checkpoint in that fold. Test-fold images are never used for optimization or checkpoint selection. We call the main experiment a same-corpus frozen group-disjoint re-evaluation: it estimates robustness across held-out recorded groups after the method definitions were fixed, but it is not method-development-independent holdout validation, a fully blind nested evaluation, or independent external validation. Historical development and the final folds are partitions of the same Fish4Knowledge-derived corpus, so this evidence boundary is stated explicitly.")
    replace_paragraph(d, 37,
        "Fig. 2 Group-aware evaluation changes the performance estimate. (a) Mean macro-F1 and group-balanced accuracy under image-level and group-disjoint splitting. (b) Head, mid, and tail per-class F1. (c) Same-group nearest-neighbour fractions across the two protocols.")
    replace_paragraph(d, 45,
        "Let x_r denote a recipient image, x_d a donor image, m_r the feathered recipient mask, and R(.) the deterministic resizing/alignment operation that maps the donor to the recipient canvas. In the frozen implementation, both images are converted to RGB, the donor is directly resized to the recipient width and height with PIL bilinear interpolation without aspect-ratio padding, and the recipient mask is resized with nearest-neighbour interpolation before Gaussian feathering. The uint8 composite is rounded and clipped on the recipient canvas before ordinary resizing, tensor conversion, and ImageNet normalization. The donor-context composite is constructed as x_dc = m_r * x_r + (1 - m_r) * R(x_d). For each recipient and intervention type, exactly one donor is selected from the same held-out outer fold. Same-class donors share the recipient species but have a different recorded group; cross-class donors have both a different species and a different recorded group.")
    replace_paragraph(d, 40,
        "The diagnostic views were defined before fitting their separate classifiers. Foreground-only retains RGB pixels inside the official subject mask and replaces the outside region with the fixed foreground-view construction used for CXT-Fish, namely Gaussian-blurred RGB with a three-pixel feather. Background-only suppresses the annotated subject region according to the frozen matched-view construction while retaining the non-primary region; it is evaluated with a separately trained diagnostic classifier. Constant-fill replaces the annotated subject region with neutral RGB (128,128,128); inpainted background applies OpenCV Telea inpainting with radius 3 after one 3 x 3 elliptical dilation of the mask; shuffled-mask background fills the non-primary region using a deterministically assigned mask from a different recorded group; mask-only copies the official binary fish mask into three channels; and geometry-only uses six normalized mask-box features: width, height, area, centre x, centre y, and aspect ratio. These separate models quantify information available in each constructed view rather than the response of the ordinary-RGB F0 model to an out-of-distribution input.")
    replace_paragraph(d, 42,
        "Fig. 3 Context-diagnostic sanity checks on the historical development validation partition. Several views retain substantial predictive non-subject/contextual signal. We reserve the stronger shortcut-susceptibility interpretation for the conflicting-context interventions rather than treating every contextual cue as ecologically spurious.")
    replace_paragraph(d, 86,
        "The near-equality of constant-fill and shuffled-mask conditions argues against the recipient fish-hole shape being the sole source of the background diagnostic signal. The lower but still substantial inpainted-background result is consistent with both residual environmental/camera context and artifacts induced by subject removal. These experiments demonstrate substantial predictive non-subject/contextual signal availability, but not a causal claim that the natural background alone determines species identity. We reserve the stronger shortcut-susceptibility interpretation for failures exposed by the conflicting-context interventions (Fig. 3).")

    replace_paragraph(d, 53,
        "For an RGB training sample x, the auxiliary foreground view x_fg is constructed from the same paired geometric augmentation as the ordinary view, then retains the official subject-mask pixels while replacing the outside region with the fixed Gaussian-blurred RGB context and three-pixel feather used throughout the frozen protocol. The shared classifier receives both views in one concatenated 2B forward. The objective is L = CE(f(x), y) + lambda_fg CE(f(x_fg), y), with lambda_fg = 1.0. Ordinary augmentation comprises RandomResizedCrop with scale 0.08–1.0 and aspect ratio 3/4–4/3, followed by a shared 0.5 horizontal flip and shared ColorJitter (brightness, contrast, saturation 0.1; hue 0.05), bilinear resize, tensor conversion, and ImageNet normalization. Foreground sufficiency is used operationally for this label-supervised context-suppressed view; it does not denote a formal proof that the foreground alone contains all predictive information.")
    replace_paragraph(d, 54,
        "Foreground sufficiency does not require original and foreground representations to coincide. Instead, it means that the context-suppressed view remains independently label-predictive under the auxiliary cross-entropy term. The term therefore describes the training intervention and its operational evaluation, not a formal subject-only identifiability claim.")
    replace_paragraph(d, 58,
        "The principal recognition summaries are macro-F1, tail-class F1, and group-balanced accuracy. Macro-F1 gives equal weight to species but remains image-weighted within each species; it is not an equal-weight recorded-group estimand. Tail classes are defined once from the frozen development frequency tiers. Group-balanced accuracy first averages correctness within each recorded group and then averages over groups, reducing domination by long recordings. Image-level accuracy and weighted F1 are retained as secondary descriptive measures. A post-hoc species–group-balanced robustness sensitivity is reported separately.")
    replace_paragraph(d, 61,
        "The frozen outer protocol recorded clean macro-F1, tail F1, and group-balanced accuracy as primary recognition metrics and donor-context quantities as secondary robustness measures. For the final manuscript synthesis, we report a conditional bootstrap interval for one focal robustness contrast: the paired difference in cross-class donor-context-composite macro-F1. We report this single interval rather than presenting a correlated family of robustness measures as independent inferential endpoints. This reporting focus does not upgrade the same-corpus design to independent validation.")
    replace_paragraph(d, 63,
        "Uncertainty is estimated with 5,000 paired group-cluster bootstrap replicates. Within every replicate, recorded groups are resampled with replacement separately within fold and ground-truth species. The same sampled groups are used for paired methods and across the three pre-specified seeds before macro-F1 is recomputed for each fold-seed cell; the nine paired cell differences are then averaged with equal weight. The resulting conditional paired group-cluster bootstrap interval quantifies held-out-group resampling uncertainty conditional on the completed development process, the frozen trained models and seeds, and the frozen seed-3407 donor realization. It does not estimate method-selection uncertainty, optimization-seed uncertainty, alternative donor-generation mechanisms, or external-dataset variability. The recorded-group counts for each fold × species stratum are reported in Supplementary Table S2 to make the finite-cluster structure explicit.")
    replace_paragraph(d, 63,
        "Uncertainty is estimated with 5,000 paired group-cluster bootstrap replicates. The cluster unit is the recipient recorded group; donor identities and donor reuse are held fixed through the complete frozen recipient-donor manifest rather than resampled as a second clustering dimension. Within every replicate, recipient recorded groups are resampled with replacement separately within fold and ground-truth species. The same sampled groups are used for paired methods and across the three pre-specified seeds before macro-F1 is recomputed for each fold-seed cell; the nine paired cell differences are then averaged with equal weight. The resulting conditional paired group-cluster bootstrap interval quantifies held-out-group resampling uncertainty conditional on the completed development process, the fixed outer-fold allocation and fold × species stratum sizes, the frozen trained checkpoints and three seeds, and the complete seed-3407 recipient-donor manifest. It does not estimate alternative fold assignments, method-selection uncertainty, optimization-seed population uncertainty, alternative donor-generation mechanisms, higher-level acquisition-unit variability, or external-dataset variability. The recorded-group counts for each fold × species stratum are reported in Supplementary Table S2 to make the finite-cluster structure explicit.")
    replace_paragraph(d, 64, "3.8 Post-hoc validity controls")
    replace_paragraph(d, 97,
        "Fig. 6 Main same-corpus frozen ResNet18 re-evaluation. (a) Mean macro-F1 across ordinary, foreground-sufficient, same-class-composite, and cross-class-composite views. (b) Equal-weight paired effects across nine fold–seed cells. The cross-class donor-context-composite difference is accompanied by the manuscript's conditional 5,000-replicate paired group-cluster bootstrap interval; lower DAR-flip is favourable.")
    replace_paragraph(d, 95,
        "The effect was substantially larger under conflicting donor context. Cross-class donor-context-composite macro-F1 increased from 0.5189 to 0.5913. The equal-weight paired improvement was 7.24 percentage points. The 5,000-replicate paired group-cluster bootstrap yielded a conditional 95% interval of +6.13 to +8.20 points. Eight of nine cells improved. DAR-flip decreased from 0.2146 to 0.1792 and was lower for CXT-Fish in all nine paired cells. The main frozen result is therefore reduced susceptibility to conflicting donor context, not improved ordinary-view recognition (Table 5; Fig. 6).")

    replace_paragraph(d, 101,
        "The frozen seed-3407 donor assignment yielded the manuscript effect of +7.24 points. Using the same frozen checkpoints and donor-construction mechanism, all five alternative deterministic donor realizations produced positive equal-weight effects: +8.00, +8.84, +8.45, +7.98, and +7.74 points for seeds 4101–4105. The mean effect across the five alternatives was +8.20 points (SD 0.44), with four realizations favourable in eight of nine cells and one in all nine cells. These are post-hoc donor-realization sensitivity results, not a second primary endpoint.")
    replace_paragraph(d, 106,
        "F0-2RGB nearly matched the clean performance of CXT-Fish (0.9551 versus 0.9556) while retaining duplicated supervised exposure, a concatenated 2B model forward, two cross-entropy terms, and comparable BatchNorm exposure. However, its cross-class donor-context-composite macro-F1 was 0.5230, compared with 0.5913 for CXT-Fish. The corrected observed equal-weight paired difference was +6.83 percentage points. A 5,000-replicate species-stratified paired group-cluster bootstrap, sharing each fold-level group draw across both methods and all three seeds, gave an exploratory conditional percentile interval of +5.71 to +7.87 points. The observed point estimate is computed from the nine paired fold-seed effects; this control is post-hoc and does not modify the main result.")
    replace_paragraph(d, 112,
        "We therefore repeated the frozen-pair evaluation after suppressing the donor subject while retaining the same recipient, donor assignment, recipient mask, checkpoint, and compositing geometry. The equal-weight CXT-Fish–F0 difference was +10.48 percentage points, with an exploratory 5,000-replicate conditional percentile interval of +9.38 to +11.61 points. All nine fold–seed cells remained positive, although the magnitude varied substantially across cells. A post-hoc donor-species-equal-weight correctness sensitivity also remained positive; detailed donor-species and residual-bin heterogeneity are reported in Supplementary Tables S7–S8. This analysis narrows the visible-donor-subject explanation but does not establish natural-background or external-domain robustness.")
    replace_paragraph(d, 118,
        "Table 8 Architecture-sensitivity summary. Values are means across the three outer folds for the one pre-specified MobileNet seed; differences are computed from unrounded cell-level values. MobileNet is architecture-sensitivity evidence only, with cross-class composite effects positive in 2/3 folds.")
    replace_paragraph(d, 119,
        "At the pre-specified MobileNet seed, foreground-sufficiency training improved mean foreground macro-F1 by 10.22 points and cross-class donor-context-composite macro-F1 by 4.82 points, while DAR-flip decreased by 4.73 points. Foreground improved in all three folds and DAR-flip decreased in all three; cross-class composite improved in two of three folds and decreased in one. However, ordinary-view macro-F1 decreased by 2.94 points and same-class-composite macro-F1 by 4.09 points. MobileNetV3-Large is therefore architecture-sensitivity evidence with heterogeneous cross-class effects, not cross-backbone validation or backbone-agnostic evidence (Table 8; Fig. 8).")
    replace_paragraph(d, 121,
        "Fig. 8 Architecture-sensitivity boundary. (a) MobileNetV3-Large fold-level effects show foreground gains in 3/3 folds, cross-class-composite gains in 2/3 folds, and heterogeneous clean cost. (b) Clean versus cross-class-composite macro-F1 for ResNet18 and MobileNetV3-Large. The MobileNet result is a one-seed sensitivity analysis, not a second main evaluation.")

    replace_paragraph(d, 124,
        "The strongest conclusion is deliberately narrow. In the main same-corpus frozen ResNet18 group-disjoint re-evaluation, foreground-sufficiency training improved performance under the tested conflicting donor-context intervention without improving mean clean recognition. The 7.24-point cross-class donor-context-composite gain, together with lower DAR-flip in all nine paired cells and positive group-weighted sensitivities, supports a targeted robustness interpretation rather than a general accuracy claim.")
    replace_paragraph(d, 128,
        "The F0-2RGB result provides the most direct mechanism-oriented control in the current evidence package. Simply doubling supervised RGB exposure with matched compute does not reproduce the cross-composite improvement, even though clean performance is almost identical to CXT-Fish. The historical F2/F3 ablations and the supplementary Route C experiment similarly show that stronger representation-level constraints did not automatically yield a better clean–robustness profile under the tested protocols. These observations should not be generalized into a claim that representation invariance or contrastive learning is inferior. They show only that the final label-sufficiency instantiation addresses the target failure more favourably in this study.")
    replace_paragraph(d, 136,
        "The practical attraction of CXT-Fish is that the intervention is confined to training. The deployed classifier has the ordinary RGB inference graph of its baseline backbone: one image, one classifier, and one forward pass. Segmentation masks are not needed at inference, and no donor selection or second view is generated online. The structural claim is therefore that CXT-Fish does not introduce a segmentation dependency into the recognition-stage inference graph. It does not claim that the full marine-monitoring pipeline is unchanged: these experiments begin after fish crops are available and do not evaluate detector errors, tracking failures, multi-fish overlap, crop errors, or downstream ecological aggregation from raw video.")
    replace_paragraph(d, 137,
        "CXT-Fish nevertheless assumes foreground masks during training. These masks are privileged annotation rather than a free deployment resource and may increase data-preparation cost where only species labels or bounding boxes are available. The present study does not test automatically generated masks or quantify sensitivity to mask quality. We also do not claim a measured latency, energy, or memory advantage because no hardware-specific deployment benchmark was frozen; training is more expensive because each recipient contributes two supervised views in one concatenated forward.")
    replace_paragraph(d, 139,
        "External validity. The principal evidence is limited to the 16-class F4K-16T subset of Fish4Knowledge. Group-disjoint evaluation prevents direct crossing of recorded groups inside each evaluation cell, but it does not establish independence across cameras, sites, dates, water conditions, species inventories, or data sources. Higher-level acquisition units such as camera, session, deployment, or date could not be verified from the frozen recognition metadata and may remain shared across recorded groups. Independent multi-site or cross-dataset replication is required before broader generalization claims can be made.")
    replace_paragraph(d, 141,
        "Evaluation design. Historical development and the main group-disjoint folds partition the same corpus. Although each final fold prevents direct recorded-group overlap between fitting, checkpoint selection, and evaluation, the analysis does not undo method selection performed earlier on the corpus and is not equivalent to a fully blind nested evaluation or external validation. The conditional bootstrap interval quantifies held-out-group resampling uncertainty conditional on the frozen trained models, the completed development process, and the frozen seed-3407 donor realization. The least represented fold × species strata contain relatively few recorded groups, so percentile cluster intervals should be interpreted as approximate finite-sample uncertainty summaries.")
    replace_paragraph(d, 142,
        "Boundary evidence. Donor-realization, F0-2RGB, donor-subject-suppressed, and group-weighted analyses are post-hoc validity controls, not new independent endpoints. MobileNetV3-Large is evaluated with one pre-specified seed and has heterogeneous cross-class effects, so it does not estimate across-seed or backbone-independent uncertainty. Route C is a fixed mechanism stress control rather than a faithful CLIB reproduction or exhaustive contrastive comparison. Only the cross-class donor-context-composite macro-F1 difference in the main ResNet18 re-evaluation receives the manuscript's conditional bootstrap interval; all newer control intervals are explicitly exploratory.")
    replace_paragraph(d, 144,
        "Video-derived underwater recognition data require evaluation units that reflect their correlation structure. In Fish4Knowledge, keeping recorded groups disjoint yielded lower class-balanced performance estimates and exposed substantial predictive information outside the annotated subject. CXT-Fish addresses the resulting vulnerability with a simple label-level foreground-sufficiency instantiation that leaves the recognition-stage deployment pathway unchanged. In the main same-corpus frozen ResNet18 group-disjoint re-evaluation, cross-class donor-context-composite macro-F1 improved by 7.24 percentage points while mean clean macro-F1 changed by −0.21 points. Post-hoc controls further showed positive effects across donor realizations, a gain beyond generic two-view RGB supervision, and persistence after donor-subject evidence was suppressed under frozen pairings. The evidence therefore supports foreground sufficiency as a targeted intervention for the tested synthetic, mask-defined context perturbations—not as a general accuracy enhancer, context-invariance guarantee, or external-domain solution.")
    replace_paragraph(d, 146,
        "Code, configurations, split manifests, audit reports, figures, and compact result artifacts are publicly available at https://github.com/Xiaoqin-pro/ctp-fish. This manuscript is grounded in the frozen reviewer-control and construct-validity snapshot at commit 11624e0f804b8f69dbd878b04a9f430bab03c055. The original main outer result and donor manifest remain unchanged; later packages add only reviewer-motivated controls, inference-only construct checks, statistical reanalysis, and reproducible figure sources. The evidence package reports 88 passing tests and no access to the official Fish4Knowledge TEST partition. The principal inference is conditional on the completed development process, frozen models/seeds, and frozen donor realization.")

    # Captions and table notes: make estimands and ± explicit.
    replace_paragraph(d, 88, "Table 4 Historical development ablations (development validation only). Values are means ± sample SD across the three fixed development seeds; they are descriptive and do not define the main frozen estimand.")
    replace_paragraph(d, 92, "Table 5 Main same-corpus frozen ResNet18 group-disjoint re-evaluation across three folds × three seeds. Values are equal-weight means ± sample SD across the nine fold–seed cells. Paired differences are computed from unrounded cell-level values. The cross-class composite interval is the conditional paired group-cluster bootstrap interval reported for the focal robustness contrast.")
    replace_paragraph(d, 109, "Fig. 7 Post-hoc validity controls. (a) The CXT-Fish minus F0 cross-class effect remains positive for the frozen seed-3407 donor assignment and five alternative deterministic donor realizations. (b) F0-2RGB nearly matches clean performance but not the donor-context robustness of CXT-Fish. (c) The CXT-Fish–F0 effect remains positive after donor-subject suppression under frozen pairings. These controls are post-hoc and exploratory; they do not constitute independent validation.")

    replace_text_in_runs(d,
        "Code, configurations, split manifests, audit reports, figures, and compact result artifacts are publicly available at https://github.com/Xiaoqin-pro/ctp-fish.",
        "Code, configurations, split manifests, audit reports, figures, and compact result artifacts are publicly available at https://github.com/Xiaoqin-pro/ctp-fish. The public repository does not redistribute trained checkpoints or full per-image prediction outputs; complete training and per-image replay additionally require an authorized local copy of the Fish4Knowledge-derived data under the original access terms.")

    # Keep table terminology aligned with the manuscript-level estimand.
    # Remove the inherited manual page break so the short reproducibility table
    # is not separated from its heading by a sparse page.
    d.paragraphs[148].paragraph_format.page_break_before = None
    # Start the short reproducibility section with its table rather than leaving
    # its heading and a few lines stranded at the foot of the preceding page.
    d.paragraphs[146].paragraph_format.page_break_before = None
    for row in d.tables[4].rows:
        if row.cells and row.cells[0].text.strip() == "Cross-class composite macro-F1":
            row.cells[-1].text = "Conditional bootstrap interval: +6.13 to +8.20 pp"
    for row in d.tables[8].rows:
        if row.cells and row.cells[0].text.strip() == "Main ResNet bootstrap":
            row.cells[1].text = "Conditional main bootstrap interval"

    # Add the two requested no-training analyses after all index-based edits.
    # Normalize the wording after the post-hoc paragraph is inserted below.
    p95 = d.paragraphs[95]
    insert_after(p95,
        "Because the main macro-F1 is class-balanced but image-weighted within species, we additionally computed a post-hoc group-weighted robustness sensitivity from the frozen per-image predictions. For cross-class composites, recorded-group-balanced accuracy changed from 0.6979 ± 0.0278 for F0 to 0.7426 ± 0.0161 for CXT-Fish, an equal-weight nine-cell difference of +4.47 percentage points (8/9 favourable cells). A stricter species–group-balanced accuracy changed from 0.5271 ± 0.0156 to 0.5944 ± 0.0269, a +6.73-point difference (9/9 favourable cells). These are descriptive sensitivity estimands, not replacements for the primary macro-F1 result; the full cell table and definitions are in Supplementary Table S1.")

    replace_text_in_runs(d, "not replacements for the primary macro-F1 result", "not replacements for the main frozen macro-F1 result")
    replace_text_in_runs(d, "remains independently label-predictive", "remains separately label-predictive")

    # Keep the diagnostic question aligned with the operational meaning of
    # foreground sufficiency: fine surrounding detail is suppressed, rather
    # than all non-subject information being removed.
    for table in d.tables:
        for row in table.rows:
            if row.cells and row.cells[0].text.strip() == "Foreground-sufficient":
                if len(row.cells) >= 3:
                    row.cells[2].text = "Does label predictiveness persist after fine-context suppression?"

    # Add verified privileged-information references before the declarations block.
    ref_anchor = next((p for p in d.paragraphs if p.text.startswith("Zhao J, Dong X")), None)
    if ref_anchor is not None:
        insert_after(ref_anchor, "Lopez-Paz D, Bottou L, Schölkopf B, Vapnik V (2016) Unifying distillation and privileged information. In: International Conference on Learning Representations.")
        insert_after(ref_anchor, "Vapnik VN, Vashist A (2009) A new learning paradigm: learning using privileged information. Neural Networks 22:544–557. https://doi.org/10.1016/j.neunet.2009.06.042")

    if ref_anchor is not None:
        insert_after(ref_anchor, "Huang PX, Boom BJ, Fisher RB (2015) Hierarchical classification with reject option for live fish recognition. Machine Vision and Applications 26(1):89–102. https://doi.org/10.1007/s00138-014-0641-2")

    # Reorder only the explicit Reference-style paragraphs and the two added
    # references; declarations and other manuscript text are never touched.
    ref_heading = next((p for p in d.paragraphs if p.text.strip() == "References"), None)
    if ref_heading is not None:
        paras = d.paragraphs
        start = next((i for i, p in enumerate(paras) if p._p is ref_heading._p), None)
        if start is not None:
            ref_paras = [
                p for p in paras[start + 1:]
                if p.style.name == "Reference" or p.text.startswith(("Huang PX", "Lopez-Paz", "Vapnik"))
            ]
            if ref_paras:
                parent = ref_paras[0]._p.getparent()
                for p in ref_paras:
                    parent.remove(p._p)
                for p in sorted(ref_paras, key=lambda p: p.text.strip().casefold()):
                    parent.append(p._p)

    # Final page-flow controls for the two short tables called out during QA.
    # Table 3 must follow its section heading rather than being pushed away by
    # a stale manual break; Table 7 is moved as a complete short table.
    for p in d.paragraphs:
        if p.text.strip() == "Table 3 Frozen implementation settings":
            p.paragraph_format.page_break_before = None
            p.paragraph_format.keep_with_next = True
        elif p.text.strip().startswith("3.10 Implementation details"):
            p.paragraph_format.page_break_before = None
            p.paragraph_format.keep_with_next = True
        elif p.text.strip().startswith("4.6 Generic two-view supervision"):
            p.paragraph_format.page_break_before = None
            p.paragraph_format.keep_with_next = False
        elif p.text.strip().startswith("Table 7 Post-hoc"):
            p.paragraph_format.page_break_before = None
            p.paragraph_format.keep_with_next = False
        elif p.text.strip().startswith("4.9 Architecture sensitivity"):
            # Keep the short architecture table from leaving only its repeated
            # header row at the foot of the preceding page.
            p.paragraph_format.page_break_before = True
            p.paragraph_format.keep_with_next = True
        elif p.text.strip().startswith("Table 8 Architecture-sensitivity"):
            p.paragraph_format.page_break_before = None
            p.paragraph_format.keep_with_next = True

    if len(d.tables) > 6:
        table7 = d.tables[6]
        for row in table7.rows:
            tr_pr = row._tr.get_or_add_trPr()
            if not any(child.tag.endswith("cantSplit") for child in tr_pr):
                tr_pr.append(OxmlElement("w:cantSplit"))

    if len(d.tables) > 5:
        table6 = d.tables[5]
        for row in table6.rows:
            tr_pr = row._tr.get_or_add_trPr()
            if not any(child.tag.endswith("cantSplit") for child in tr_pr):
                tr_pr.append(OxmlElement("w:cantSplit"))

    # Apply final pagination controls after all insertions have stabilized
    # paragraph indices.
    for p in d.paragraphs:
        if p.text.startswith("7 Reproducibility and data/code availability"):
            p.paragraph_format.page_break_before = None
            p.paragraph_format.keep_with_next = False
        elif p.text.startswith("4.5 The robustness effect is not confined to one donor assignment"):
            p.paragraph_format.page_break_before = True
            p.paragraph_format.keep_with_next = True
        elif p.text.startswith("Table 9 Key reproducibility assets"):
            p.paragraph_format.page_break_before = None

    # Final prose pass. The structure follows the compact problem-design-result-
    # boundary rhythm common in high-impact empirical papers while preserving
    # every frozen numerical result and evidence tier.
    final_prose = {
        "Underwater fish-recognition benchmarks derived from video contain correlated observations":
            "Video-derived underwater fish benchmarks contain correlated observations: adjacent frames from one recording are not independent evidence for a new recording. We audited this problem in Fish4Knowledge and constructed F4K-16T, a fixed 27,133-image, 16-species protocol with 8,680 recorded groups. The study combines recorded-group-disjoint evaluation with foreground and donor-context interventions that test whether predictions remain stable when non-recipient evidence conflicts with the fish label. We then instantiated a simple label-level foreground-sufficiency objective. During training, one shared classifier predicts the species from both ordinary RGB and a mask-derived view in which fine surrounding detail is suppressed; at the recognition stage, after fish crops are available, inference still uses one ordinary RGB image and one forward pass. After fixing the method and evaluation definitions, we performed a three-fold, three-seed group-disjoint re-evaluation on the same Fish4Knowledge-derived corpus used during development. Cross-class donor-context-composite macro-F1 increased from 0.5189 to 0.5913, an equal-weight paired difference of 7.24 percentage points. Conditional on the completed development process, frozen fold allocation, trained checkpoints, seeds, and donor manifest, the paired group-cluster bootstrap 95% interval was 6.13 to 8.20 points. Mean clean macro-F1 changed from 0.9577 to 0.9556. Bounded post-hoc controls remained positive across five alternative donor assignments, exceeded an ordinary-RGB two-view control by 6.83 points, and persisted after donor-subject suppression. The evidence supports a targeted improvement under the tested synthetic, mask-defined intervention family, not clean-accuracy improvement, natural-context invariance, external-domain generalization, or end-to-end monitoring-system validation.",
        "The contributions are threefold.":
            "This study makes three contributions. First, it provides an integrated reliability audit for Fish4Knowledge-derived recognition, linking recorded-group-disjoint evaluation, nearest-neighbour correlation analysis, group-aware summaries, and controlled context interventions. Second, it instantiates a simple label-level foreground-sufficiency objective in which segmentation-derived information is privileged during training but absent from the recognition-stage inference graph. This is an application-specific intervention, not a new learning paradigm. Third, it separates the main same-corpus frozen group-disjoint re-evaluation from bounded post-hoc controls of donor assignment, duplicated RGB supervision, donor-subject evidence, architecture sensitivity, and representation-level stress. This evidence hierarchy supports a targeted robustness claim: cross-class donor-context-composite macro-F1 increased by 7.24 points, whereas clean macro-F1 changed by -0.21 points.",
        "F2 did not improve the target clean-robustness profile":
            "F2 did not improve the target clean-robustness profile. F3 achieved the highest mean development cross-composite score, but it added a representation-consistency assumption and produced a less favourable clean-robustness balance. F1 was selected because it most directly instantiated the label-level study hypothesis with no additional representation constraint; it was not selected by maximizing cross-composite macro-F1. F2 and F3 were retained as diagnostic ablations, and the F1 definition was frozen before the three-fold outer re-evaluation. The outer analysis therefore compares only the frozen F0 and F1 definitions (Table 4).",
        "The effect was substantially larger under conflicting donor context.":
            "The clearest separation appeared under conflicting donor context. Cross-class donor-context-composite macro-F1 increased from 0.5189 to 0.5913, an equal-weight paired improvement of 7.24 percentage points. The 5,000-replicate paired group-cluster bootstrap yielded a conditional 95% interval of +6.13 to +8.20 points under the frozen fold allocation, checkpoints, seeds, and donor manifest. Eight of nine fold-seed cells improved. DAR-flip decreased from 0.2146 to 0.1792 and was lower for CXT-Fish in all nine cells. Thus, the main frozen result is higher performance under the tested synthetic conflicting-context intervention, not better ordinary-view recognition (Table 5; Fig. 6).",
        "The strongest conclusion is deliberately narrow.":
            "The main result is deliberately narrow. In the same-corpus frozen ResNet18 group-disjoint re-evaluation, foreground-sufficiency training improved performance under the tested synthetic conflicting donor-context intervention without improving mean clean recognition. The 7.24-point cross-class-composite gain, lower DAR-flip in all nine paired cells, and positive group-weighted sensitivities jointly support a targeted robustness interpretation. They do not establish natural-context invariance, external-domain generalization, or a universal accuracy benefit.",
        "The post-hoc controls sharpen the interpretation of that result":
            "The post-hoc controls narrow three alternative explanations. The effect remained positive across five additional deterministic donor assignments, making one favourable seed-3407 pairing an unlikely sole explanation. CXT-Fish also exceeded the supervision- and compute-matched ordinary-RGB two-view control by 6.83 points, although the control does not match every augmentation-correlation detail. Finally, the effect persisted after visible donor-subject evidence was suppressed while pairings and checkpoints remained fixed. These analyses strengthen construct validity within the frozen intervention family; they do not create an independent evaluation.",
        "Video-derived underwater recognition data require evaluation units":
            "For video-derived underwater recognition, the evaluation unit is part of the scientific claim. In Fish4Knowledge, keeping recorded groups disjoint lowered class-balanced performance estimates and revealed substantial predictive information outside the annotated subject. CXT-Fish addresses one resulting failure mode with a simple label-level foreground-sufficiency intervention that leaves recognition-stage inference unchanged. In the same-corpus frozen ResNet18 re-evaluation, cross-class donor-context-composite macro-F1 increased by 7.24 points while clean macro-F1 changed by -0.21 points. The effect remained positive across donor assignments, beyond an ordinary-RGB two-view control, and after donor-subject suppression under frozen pairings. These findings support targeted robustness to the tested synthetic, mask-defined intervention family. They do not establish a general accuracy gain, context invariance, external-domain transfer, or end-to-end deployment reliability."
    }
    for p in d.paragraphs:
        for start, replacement in final_prose.items():
            if p.text.startswith(start):
                p.text = replacement
                break

    # Replace embedded figures with reproducible, corrected outputs.
    d.save(OUT)
    mapping = {
        "image9.png": FIG_DIR / "fig1_protocol.png",
        "image2.png": FIG_DIR / "fig2_gate0.png",
        "image3.png": FIG_DIR / "fig3_context.png",
        "image4.png": FIG_DIR / "fig4_donor_intervention.png",
        "image5.png": FIG_DIR / "fig5_method.png",
        "image6.png": FIG_DIR / "fig6_main_results.png",
        "image10.png": FIG_DIR / "fig7_controls.png",
        "image8.png": FIG_DIR / "fig8_architecture_sensitivity.png",
    }
    tmp = OUT.with_suffix(".tmp.docx")
    with zipfile.ZipFile(OUT, "r") as zin, zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            name = Path(item.filename).name
            if item.filename.startswith("word/media/") and name in mapping:
                data = mapping[name].read_bytes()
            zout.writestr(item, data)
    tmp.replace(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
