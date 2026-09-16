# Reviewer 2 Comments (received 2026-09-12, from reviewer2.docx)

Recommendation: Major revision

1. Novelty framing overstated ("first" semantic segmentation approach for colony-scale
   bleaching assessment) given CoralSCOP and related literature. Clarify that the novelty
   is in the combination of SAM2 + bleaching status + morpho-taxonomic grouping + ASV
   top-down imagery, not a broad first-in-field claim.

2. Data section: annotations were SAM2-assisted, which may bias evaluation (since the
   pipeline itself relies on SAM2). Report inter-annotator agreement, QC procedures,
   correction rules, and any independent expert audit. Clarify the "non-coral" class
   definition (negative masks weren't explicitly annotated; they become a major training
   component via synthetic generation).

3. Experimental design under-validated: single reef system, single ASV platform, one
   camera geometry, held-out test set from the same domain. Add baseline comparisons
   (SAM2 alone, CoralSCOP, U-Net/DeepLab, Mask R-CNN, non-ensemble ResNet) and ablations
   (dual SAM2 config, synthetic non-coral generation, ov ersampling, augmentation, ensemble
   weighting, mask consolidation).

4. Train/validation/held-out/test terminology is confusing; possible over-optimistic
   reporting. Abstract's 90.9%/95.8%/94.3% appear to be from Table 4's augmented
   validation set (1,762 masks), while ecological evaluation uses the 111 held-out images.
   Explicitly state that no masks/images/augmentations/hyperparameters from the final
   111-image test set were used anywhere upstream (SAM2 tuning, CNN training, ensemble
   weighting, threshold selection).

5. [SCOPE OF THIS REVISION PASS] Method description has logical ambiguities:
   - Section 3.2.1's mask consolidation (uses classification confidence) creates an
     apparent ordering conflict with Figure 4 (implies classify -> reject non-coral ->
     consolidate) vs. Sections 3.2.1/3.3 (implies consolidate -> classify). Needs an
     unambiguous rewrite of the pipeline sequence. [NOT YET ADDRESSED]
   - Ensemble aggregation: non-coral class weight not reported -- FIXED (nu_noncoral =
     0.25, justified via a priori ~25% coral cover prior, boosts recall).
   - Optimization details incomplete -- FIXED (completed the MM derivation for the
     w_m update, added closed-form multiplicative update, renormalization step, and a
     methods+results paragraph comparing 6 ensemble-combination schemes).
   - Convexity claim needs justification -- FIXED (removed "novel convex ensemble
     optimization scheme" framing; the joint (alpha, W) problem is explicitly non-convex,
     only individual conditional updates are; reframed around interpretability instead).

6. Results should be interpreted more conservatively: pixel accuracy >90% can be inflated
   by dominant background/easy non-coral pixels. Add IoU, Dice, F1, per-class P/R,
   macro-averaged metrics, confidence intervals. Treat LCC/BCC underestimation as a
   substantive limitation, not a minor residual bias.

7. Figure quality: Figures 4, 5 too dense/small text. Figure 13 panels hard to read.
   Figure 15 negative depth values need clarification. Figure 1 needs scale/depth/quality
   context. Figure 9 needs clearer legends. Table 1 should clarify whether "average coral
   cover" includes both healthy+bleached; Table 4 should define whether P/R are
   coral-vs-noncoral, macro-averaged, or mask-level aggregated.

8. Related Work lacks recent underwater/UAV monitoring literature: cites specific papers
   to add (Holothurians monitoring UAV 2025; Mussel-YOLO EI 2025; RBL-YOLO MPB 2024;
   multi-label coral condition monitoring MPB 2026; multi-scale coral condition/Drupella
   MPB 2026; Cost-Effective AUAV monitoring MER 2026).

9. Copyediting: numerous typos ("heatstress," "stake holders," "over fishing," "land
   based," "long term," "limited shallower reefs," "photodata," "yielding to the
   complexity," "proceeding section," "intersection-over-union The," "parameter in 2,"
   "helps balances," "use to model," "the proposed pipelines ability," "proposed
   pipelines reproduces," "may reflects") and an overly informal acknowledgement
   ("A massive thank you goes [to] Aidan Quigley"). Also: "Beijorn" should be "Beijbom,"
   "Marcohorn" should be "Macrohon," SAM2 appears cited with the original SAM paper's
   reference in places, and CoralMask/USIS10K/CoralSCOP references appear mismatched or
   duplicated.

---
**Status as of 2026-09-12, pass 2 (figures/tables refresh):**

- **Item 5** (ensemble/EM math): addressed in pass 1. Non-coral weight (0.25) now
  reported and justified. Pipeline-ordering ambiguity (Sec 3.2.1 vs Fig 4) still open.
- **Item 6** (IoU/Dice/F1/macro/CIs): mostly addressed. New Section 4.x
  ("Segmentation Quality: IoU and Dice") adds per-taxonomy pixel IoU/Dice/precision/
  recall (Table `tab:iou-dice-taxonomy`), a NeuralReefer-vs-CoralSCOP comparison
  (Table `tab:coralscop-comparison`), an IoU exceedance figure, and a conditional-IoU-
  by-LCC-quartile table/discussion. Bootstrap CIs (already coded in data_viz.R's
  per-taxonomy section) not yet pulled into the paper table itself -- still open.
- **Item 7** (figure quality): Figures 4 and 5 converted to `sidewaysfigure` on their
  own landscape page. Figure 13 commented out. Figure 9 now has one shared legend
  (regenerated `pred.png`s from the current model, `show_labels=False`) instead of a
  caption-only color disclaimer. Figure 1's caption now states pixel dimensions and
  the survey's camera-depth-below-surface range; **altitude/GSD/physical scale is not
  recoverable** -- the metadata has no camera-to-substrate distance or EXIF, so no
  scale bar could be added (flagged to user, not fabricated). Figure 15's negative-
  depth sign-convention concern (depth below water surface, not distance to
  substrate) is NOT yet addressed -- still open. Table 1/Table 4 metric-definition
  clarifications requested by this item are also NOT yet addressed.
- **All figures/tables refreshed to current data** (separate from reviewer comments,
  per direct user request): Figs 2,3,6,7,8,9,10,11,12,14,15 and Tables 4,5 all now
  trace to the Sep-12 `data/performance/inference/*` run (previously a mix of Aug and
  Sep vintages); abstract headline stats and Figure 7/8 captions corrected to match.
- **Items 1-4, 8, 9**: still fully outstanding, not touched this pass.

**Status as of 2026-09-12, pass 3 (user's reconciliation notes):**

- **Comment 4** (mask accuracy vs. pixelwise segmentation accuracy ambiguity): now fully
  addressed. Abstract rewritten to lead with pipeline-level LCC IoU/Dice (0.674/0.781,
  vs. CoralSCOP's 0.487/0.614) as the primary claim, with mask-level classification
  accuracy reported as a clearly-labeled secondary result. Methods section (data-
  partitioning paragraph) now explicitly states Table 4 evaluates mask-level
  classification only, while pipeline quality is measured via IoU/Dice on the 111 test
  images. New paragraph in Section "Segmentation Quality: IoU and Dice" makes the causal
  point explicit: good classification (Table 4) cannot fix coral SAM2 never found in the
  first place, so segmentation quality bounds and drives the LCC bias.
- **Comment 5, pipeline-ordering ambiguity**: resolved. Verified against the actual code
  (`segmenter.py`) that the true order is propose -> classify -> reject non-coral ->
  consolidate (Figure 4 was already correct; the prose in "SAM2-Based Segmentation" and
  "Coral Cover Estimation" asserted the reverse and has been rewritten to match the code
  and figure, with an explicit note that the subsections' narrative grouping differs from
  execution order).
- **Two of my own pass-1 edits were factually wrong and are now corrected**: Figure 4's
  caption ("cropped, normalized...") and Table 4's note ("deterministically preprocessed
  (non-augmented)...") both incorrectly described live inference and OOS validation as
  augmentation-free -- verified against `filter.py`/`transforms.py` that both actually
  use the same stochastic `MASK_TRANSFORM_AUGMENT`, just under a fixed seed. Also
  corrected Section 3.2.2's claim that augmentation is resampled per individual mask
  every epoch -- it's actually shared per chunk of up to 1,000 masks.
- **New Appendix "Robustness to Spatial Dependence"** (`app:spatial`): recomputed LCC
  IoU/Dice/accuracy excluding test images within 2m and 5m of a training image.
  Reassuring result: negligible change at 2m (96 of 111 images), and no degradation at
  the stricter 5m threshold (39 images) either -- inconsistent with spatial leakage
  materially inflating the reported numbers.
- **Methods**: added an explicit stacked-generalization (Wolpert 1992) justification for
  the disjoint submodel/ensemble/test-set partitioning, and named the mask-level
  exchangeability assumption explicitly.
- **Limitations/Future Work additions**: TOLERANCE=0.2 label-noise caveat (synthetic
  non-coral masks with 0-20% real-coral overlap); point-prompted SAM2 (CoralSCOP-style)
  as a segmentation alternative; test-time-augmentation ensembling as a future direction
  (noting the current single-draw-per-mask choice is for compute efficiency); corrected
  the ensemble-comparison paragraph to state precisely that the neural meta-learner wins
  on recall/F2 (not merely "competitive"), with a new limitations paragraph on the
  interpretability-vs-performance tradeoff this implies.
- **Comment 6/7 conditional-IoU/degenerate-cases/exceedance-curve content** (added pass
  2) was spot-checked against the user's detailed notes and already covers items 9/9b/9c
  adequately -- no changes needed there.

**New, unplanned findings from this pass, worth the user's attention:**
- `data_viz.R` had a broken file path and a confusion-matrix parser bug (missing
  `dtype` suffix in the current file format) -- both were silently producing wrong/NA
  output before this pass; both are now fixed.
- Several `\revadd`/`\revdel` edits from pass 1 would have failed to compile at all
  (math content escaping `$...$` inside `\revdel`, an entire `align` environment
  wrapped in `\revdel`, a paragraph break inside a `\textcolor` argument) -- all fixed.
  `\revdel` is only safe for short (a few words) deletions; longer deletions in this
  pass were dropped silently (kept only the red `\revadd` replacement, no strikethrough
  of the old text) to avoid the same class of bug.
- Table 3's "Epochs = 30" vs. `scripts/config.py`'s `EPOCHS = 100` was not
  independently verified (plausibly explained by early stopping, `PATIENCE = 5`, but
  no training log was available to confirm the actual stopped-at epoch).

---
**Status as of 2026-09-13, pass 4 (visual/notation/structure polish):**

- **Figure theme unification** (item 7, continued): all figures now share one theme
  (`nf.theme`/`nf.theme.single`/`nf.theme.facet` in `data_viz.R`) matching the ensemble
  heatmap's look (serif "cm" font, `#F7F7F7` panel, bold titles/axes) -- a prior-session
  bug (`nf.theme.single <- nf.theme %+replace% theme(text=element_text(size=34))`) had
  silently dropped the custom font family via `%+replace%`'s whole-element replacement;
  fixed by switching to `+` and re-specifying `family="cm"`. Figure 2 (annotation-count
  histogram) got its first-ever reproducible R code this pass. Density-plot color
  borders removed; Figure 11's LOESS smoother replaced with a quadratic `lm` fit (no SE
  ribbon); confusion-matrix heatmaps (Figs 7-8) re-themed and given pretty class-name
  labels via a shared `pretty_class_name()`.
- **Figure 9** (item 7): root-caused and fixed the GT/predicted color-scheme mismatch --
  `CoralSegmenter._create_color_map` draws from unseeded `np.random.rand`, so `gt.png`
  and `pred.png` were built with different colors in different runs. `render_examples.py`
  now seeds once, builds one `CoralSegmenter`, and renders both `gt.png` (via
  `get_gt_masks`) and `pred.png` through the same `color_map`, both with per-mask text
  labels restored (`show_labels=True`). Legend whitespace cropped.
- **Figure/table reordering per user request**: accuracy-density figure moved to appear
  right after the LCC calibration figure (before BCC); depth-vs-accuracy figure and its
  discussion moved to a new Appendix B (with an explicit sign-convention note addressing
  item 7's "Figure 15 negative depth values" comment); Table 8 (conditional-IoU-by-
  LCC-quartile) commented out as redundant with the IoU exceedance figure, its findings
  kept as prose; Table 7 restructured to match Table 4's In-Sample/Out-of-Sample
  column-panel style, "NeuralReefer (single submodel, no ensemble)" renamed "Single CNN".
- **Notation/math additions**: numbered pixel-set IoU/Dice equations added (Eq. 5-6,
  `TP_k/FP_k/FN_k` matching the code's `_pixel_confusion_metrics` exactly); all 7
  `\mathcal{M}_\text{...}` mask-set notation occurrences removed and rephrased as prose
  (verified none appeared inside actual equations); CNN architecture (Table 3) given an
  explicit "this is standard/conventional, not a novelty point" justification paragraph;
  EM/MM derivation (6 labeled equations) moved to a new Appendix C, with a high-level
  E-step/M-step/MM-surrogate summary left in the main text; added a precise argument
  (citing Lipton, Elkan & Narayanaswamy 2014 on Fβ-optimal Bayes classifiers) for why
  maximizing F2 is consistent with minimizing the Eq. 4 loss, given ν_noncoral=0.25=1/4
  matches F2's β²=4 recall-weighting.
- **Item 6/7 residual items now addressed**: LCC explicitly clarified (at its first
  definition and again near Figure 11) to include bleached coral cover, per the
  reviewer's comment that this wasn't clear from the prose alone.
- **Table A.9 (now A.8 after Table 8's removal)**: note rewritten to explicitly name the
  method as a DBSCAN-style fixed-radius kd-tree search (`dbscan::frNN`), and the R code
  itself was switched from a plain O(n_test x n_train) loop to actually using `frNN`
  (two-set query), so the paper's methodology description is now literally accurate
  (verified identical contamination counts, 15/105 at 2m and 72/105 at 5m, before/after
  the code switch).
- **Found and fixed a pre-existing bug** (not from this pass, but newly noticed):
  `\ref{app:spatial}` (and other `app:*` labels) already expand to "Appendix~A" via
  elsarticle's appendix numbering, so every existing `Appendix~\ref{app:spatial}` in the
  paper was rendering as "Appendix Appendix A" -- removed the redundant literal
  "Appendix~" text everywhere (5 occurrences) so `\ref{app:...}` alone renders correctly.
- **New reference added**: `lipton2014fbeta` (Lipton, Elkan & Narayanaswamy, ECML PKDD
  2014, "Optimal Thresholding of Classifiers to Maximize F1 Measure") -- a real,
  verifiable paper, added for the F2/loss-weighting justification above.
- Full `pdflatex`+`bibtex`+`pdflatex`x2 cycle: 43 pages, zero errors, zero undefined
  references/citations, no overfull hboxes exceeding 20pt.

---
**Status as of 2026-09-13, pass 5 (Figure 1 rebuild + remaining theme polish):**

- **Figure 1 rebuilt as a true scatter-of-images plot** (pgfplots/TikZ): panels now sit
  at their literal acquisition time (x) and logged depth below the surface (y), with real
  axis ticks/labels, replacing the old fixed 3x3 grid. The original 9 example photos
  clustered too tightly in time/depth to plot this way without overlapping, so a new set
  of 9 was selected from the full 552-image annotated pool via farthest-point sampling in
  physical (cm) space (capping depth at -9m to keep 4 extreme outlier readings from
  compressing the rest of the scale) -- verified zero bounding-box overlap at the chosen
  2.4cm thumbnail size before finalizing. `pgfplots` added to the preamble.
- **Root-caused two real ggplot2 4.0.2 rendering quirks** via isolated reproduction
  scripts (not visible in this project's own docs/changelog, so worth flagging if it
  recurs elsewhere): (1) `element_rect(fill=...)` with no explicit `colour` now draws a
  visible near-black stroke by default -- this, not any intentional border setting, was
  the entire "black border with grid lines poking through it" issue; fixed by adding
  `colour = NA` to `panel.background` and adding a deliberate `panel.border` (which
  ggplot always renders after `panel.grid`, so it now frames the grid cleanly instead of
  being painted over). (2) `geom_point`'s default shape (19) renders a visibly darker rim
  around each point under `alpha < 1`, even for isolated non-overlapping points with no
  fill/stroke aesthetics set; `shape = 16` eliminates it completely. Both fixes are now in
  `nf.theme` / the two affected `geom_point` calls (Figure 11, Appendix Figure B).
- **Other theme edits**: grid line width reduced (`panel.grid.major/minor linewidth`
  0.3/0.15, down from an unset default that scaled to ~1.6 at `base_size=36`);
  `nf.theme.single`/`nf.theme.facet` text bumped 34→40pt; confusion-matrix heatmaps
  (Figs 7-8, and the ensemble heatmap Fig 6) bumped to 44pt base text and annotation
  numbers 7.5→11pt, with `panel.border=element_blank()` added since heatmaps shouldn't
  have the new border; `aspect.ratio=1` added to `nf.theme.facet` to force square facet
  panels (Figures 2, 11, 13), with `ggsave` widths increased accordingly and
  `legend.box.spacing` tightened.
- **Figure 9's gt/pred cell overflow fixed**: `predictions.tex`'s `\qualCellImg` macro
  only constrained image *height*, so a gt/pred panel whose bbox-cropped aspect ratio
  wasn't perfectly square could overflow its column width; added a matching `width=
  \qualColW` constraint (with `keepaspectratio` picking whichever bound is tighter).
- Full `pdflatex`+`bibtex`+`pdflatex`x2 cycle re-verified after all of the above: 43
  pages, zero errors, zero undefined references/citations.

---
**Status as of 2026-09-13, pass 6 (fine-tuning the pass-5 theme changes):**

- **Figures 2/13**: pass 5's `legend.box.spacing` was set too tight (4pt, effectively
  none); increased to 10pt for a small but visible gap between the facet panels and the
  legend.
- **Figure 7** (3x3 bleaching confusion matrix): `plot_confusion_heatmap()` now takes
  `text_size`/`annot_size`/`axis_x_rel`/`axis_y_rel` parameters so a small matrix (far
  more room per cell) can use much larger text than a large one -- Figure 7 now renders
  at 60pt base text / 20pt cell annotations (vs. Figure 8's 44pt/11pt), with the axis
  label/tick sizes also bumped.
- **Figure 9 row-size inconsistency root-caused and fixed at the source**: gt.png/pred.png
  were saved via `fig.savefig(..., bbox_inches='tight', pad_inches=0)`
  (`segmenter.py::show_masks`), which crops to the rendered content's extent --
  including per-mask text labels, whose bounding boxes can extend past the 1024x1024
  image near the edges by a different amount per image/mask layout. This made saved
  figures inconsistently non-square (aspect ratios measured 1.02-1.15 across the 4
  examples) even though the source image is always exactly square. Fix: drop
  `bbox_inches='tight'` entirely and save the fixed `figsize=(10, 10)` canvas as-is --
  every output is now verified exactly 3000x3000px (1:1) across all 4 examples' gt and
  pred images. (This is a shared method also used by `main.py`/`train.py`'s own
  diagnostic dumps, so they get the same consistency fix for free.)
- **Figure 11**: point alpha raised 0.5 -> 0.75 (per "increase opaqueness by 50%"); trend
  reverted from the pass-4 quadratic fit back to a plain linear `lm` (`y ~ x`) per this
  round's explicit request -- caption updated to say "linear" again.
- **Figure B.15 (Appendix) gained a second panel**: now a 2-panel faceted figure (Full
  Range | Zoomed to depth 0 to -7.5m and accuracy 75-100%), styled with `nf.theme.facet`
  to match Figures 2/13. The zoomed window contains 92.0% of all images (n=494/537 with
  valid depth+accuracy); a linear trend fitted per split in that panel only, both
  close to flat. Caption updated to describe both panels and cite the 92.0% figure.
- Full `pdflatex`+`bibtex`+`pdflatex`x2 cycle re-verified: 42 pages, zero errors, zero
  undefined references/citations.
- **Follow-up**: og.jpg fills 100% of its own square frame with zero internal margin,
  while gt/pred.png (after pass 6's `bbox_inches='tight'` removal) have a genuine,
  measured 5% margin baked in on every side (matplotlib axes occupy exactly
  `[0.05,0.05,0.9,0.9]` of the figure) -- so their visible photo content fills only 90%
  of the nominal image size. `predictions.tex`'s `\qualOGcell` was displaying og at 0.92
  (close, but not quite matching); reduced to 0.90 to exactly match gt/pred's effective
  visible-content scale, so the coral photo now appears the same apparent size across
  all three columns in every row. Recompiled clean (42 pages, zero errors).

---
**Status as of 2026-09-13, pass 7 (small refinements):**

- **Confusion matrices (Figs 7-8)**: cells that are exactly zero now render blank/
  transparent (`fill = NA`, no tile color) with no "0.00" annotation, instead of showing
  the palette's lightest color plus a zero label. Cells that merely *round* to 0.00 but
  aren't exactly zero (e.g. Non-Coral's tiny off-diagonal misclassification rates) still
  display normally, per the literal "equal zero" request.
- **`geom_point` size increased 50%** (0.75 -> 1.125) on Figure 11 and the Appendix
  depth-vs-accuracy figure (both panels).
- Full `pdflatex`+`bibtex`+`pdflatex`x2 cycle re-verified: 42 pages, zero errors, zero
  undefined references/citations.

---
**Status as of 2026-09-13, pass 8 (axis/tick/strip text sizes + longest-table question):**

- **Axis tick labels and facet subpanel titles bumped up a bit everywhere**: both
  defaulted to `theme_minimal`'s `rel(0.8)` of the base text size (confirmed via
  `calc_element()`, e.g. 32pt vs. 40pt axis titles on the facet theme) -- `nf.theme`'s
  `axis.text` and `nf.theme.facet`'s new `strip.text` are now explicit `rel(0.95)`,
  cascading through every plot (single, facet, and the confusion-matrix/ensemble-heatmap
  theme, which layers its own further axis-text `rel()` multiplier on top of this same
  base). A modest, proportional increase everywhere rather than a per-figure tweak.
- **User question answered**: Table 2 (SAM2 Tuned Parameters) is confirmed, by direct
  empirical measurement (`\settowidth` on each table's content typeset at its own
  natural/unconstrained width, stripping `\revadd`/`\revdel` wrappers first), to still
  be the widest table in the paper -- 643.5pt natural width vs. the next-widest (Table 4,
  Model Performance) at 473.1pt. This is the table stored in `\perftablebox`
  (`main.tex` ~line 171) that every other table's `\wd\perftablebox`-sized `tabular*`
  references for a shared column width; the mechanism's own code comment warns this can
  change as content is edited, and it hasn't -- no action needed, just confirmed current.
  Full ranking (natural pt): T2 643.5 > T4 473.1 > T1 434.0 > T5 419.2 > T7 413.7 >
  T-A.8 407.9 > T3 364.6 > T6 324.4.
- Full `pdflatex`+`bibtex`+`pdflatex`x2 cycle re-verified: 42 pages, zero errors, zero
  undefined references/citations.

---
**Status as of 2026-09-13, pass 9 (Table 2 narrowing -- big win for every table's size):**

- **Table 2 header text, not its data, turned out to be the dominant width driver**:
  "Large Configuration"/"Small Configuration" (17 chars each, bold) were wider than any
  actual value in those columns (widest data value: "10,000" at 6 chars). Shortened to
  "Large"/"Small" with a new table note clarifying these denote the large-/small-object
  SAM2 configurations (Section 3.2.1), not model sizes; also tightened the column
  gutters around the last two columns (`@{\hspace{4pt}}` in place of default
  `\tabcolsep` at those boundaries) and added a `\midrule` before the "Postprocessing
  and Augmentation Parameters" panel (previously missing).
- **Net effect, measured directly**: `\wd\perftablebox` (the shared reference width
  every other table's `tabular*` stretches to) dropped from 643.5pt to 468.9pt --
  almost exactly `\linewidth` (469.8pt). Since every table is `\resizebox{\linewidth}
  {!}{...}`-wrapped, this means the old oversized reference was forcing a substantial
  *downscale* on literally every table in the paper; now that reference is ~1:1 with
  the page width, so every table renders at very close to its natural, undiminished
  font size. Visually confirmed a dramatic legibility improvement on Tables 2, 4, and 7.
  (Page count grew 42->44 as a direct, expected consequence of every table now taking
  up more vertical space.)
- **Double rule after column headers**: recommended against it. `booktabs` (used
  throughout this document for `\toprule`/`\midrule`/`\bottomrule`) is deliberately
  minimalist and doesn't provide a double-rule primitive; a single `\midrule` after the
  header (already the pattern in every other table here) is the modern
  scientific-table convention this package embodies. Left as a single rule for
  consistency with the rest of the paper.
- Full `pdflatex`+`bibtex`+`pdflatex`x2 cycle re-verified: 44 pages, zero errors, zero
  undefined references/citations.

---
**Status as of 2026-09-13, pass 10 (final grammar/logic/notation skim):**

Full read-through of `main.tex` (via a forked agent, read-only investigation) hunting
for grammar errors, logical inconsistencies, and notation introduced but used only
once. Findings and fixes:

- **Grammar**: fixed a run-on sentence missing a period (Table 1 notes), an awkward
  "as a backbone for each layer" phrase (dropped the dangling clause), redundant
  "$\kappa$ number of genera" -> "$\kappa$ genera", a missing preposition ("area
  available reef substrate" -> "area of available reef substrate"), and an awkward
  "classes...classes for each class" repetition in the Related Work novelty claim.
- **Logical inconsistencies**: (1) "As defined earlier, we let $\mathcal{D} = ...$" --
  $\mathcal{D}$ was never actually named earlier (and, per the notation check below, is
  never used again either) -- dropped the unused name via `\revdel`. (2) Table 5's
  "decompose additively... by construction" note overclaimed exactness: the Marginal
  Sums rows' Bleached+Healthy don't sum to the displayed Total (off by up to 0.0007,
  larger than four-decimal rounding alone would explain) -- added an explicit rounding
  caveat to the table note rather than silently leaving the overclaim. (3) A stale code
  comment claimed `\perftablebox` holds Table 4's content; it's actually held Table 2's
  content since it was introduced -- fixed the comment (no functional change).
- **Notation used only once**: removed $\mathcal{Y}$ and the one-hot vector
  $\bm{e}_{y_i}$ (Section 3.2.2) -- never referenced again, and the actual loss equation
  uses $\mathbbm{1}(y_i=k)$ on the scalar index instead, so both were pure clutter;
  removed the undefined, never-reused, never-valued threshold $\tau_\text{area}$
  (Section 3.2.1) in favor of plain prose ("a minimum area threshold"); fixed a stray
  bold $\bm{x}_i$ (used exactly once) to match the plain $x_i$ used everywhere else for
  the same quantity; renamed the Limitations section's local $C \in \{0,1\}$
  (coral/non-coral indicator) to $S$ to resolve a collision with the unrelated,
  much-more-heavily-used $C$ = channel count from Section 3.2.2's $\mathbb{R}^{C\times
  H\times W}$ notation. Left the Limitations section's $\kappa$/$G$/$B$/$S$ dimensionality
  illustration and the loss function's $\mathcal{L}(\bm\alpha,\bm W)$ name alone --
  both are used multiple times within their own local scope / are standard convention,
  not vestigial one-off notation.
- Full `pdflatex`+`bibtex`+`pdflatex`x2 cycle re-verified: 43 pages, zero errors, zero
  undefined references/citations. Visually spot-checked the highest-risk edit (a
  `\revdel` wrapping math content, historically fragile in this document) and confirmed
  correct strikethrough/red-addition rendering.

---
**Status as of 2026-09-13, pass 11 (figure/table placement-specifier consistency):**

- Checked Elsevier's actual submission/artwork policy (Ecological Informatics uses the
  standard Elsevier/ScienceDirect system): LaTeX manuscripts are explicitly permitted to
  embed figures at their point of discussion via normal `\includegraphics`/`\begin{figure}`
  commands (not required to defer to the end) -- confirmed we're following the sanctioned
  workflow. Separately, individual figure source files are still required at final
  submission regardless of embedding (submission-portal step, not a `.tex` change).
- **Found and fixed 3 figures with no placement specifier at all** (bare `\begin{figure}`,
  Figures 2, 3, and 6) -- inconsistent with the other 13 figures and 9 tables, which all
  use `[htbp]`. A bare specifier falls back to the class default, which typically omits
  "here" as an option, so these could have floated further from their point of discussion
  than every other figure in the paper. Standardized all three to `[htbp]` for consistency.
  The two `sidewaysfigure`s intentionally use `[p]` (dedicated landscape page) and were
  left as-is; the two `[!ht]` grep hits are inside historical provenance comments, not
  live code.
- Full `pdflatex`+`bibtex`+`pdflatex`x2 cycle re-verified: 43 pages, zero errors, zero
  undefined references/citations.
