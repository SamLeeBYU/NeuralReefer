library(tidyverse)
library(readxl)
library(dbscan)
library(igraph)
library(jsonlite)

metadata = readxl::read_xlsx("data/metadata/Day3_Photo_MetaData_sr4.xlsx")

eval.dat = read_csv("data/performance/coral_segmenter_predictions.v.1.0.csv")

# --- Train/Test Independence Check ---------------------------------------
# eval.dat above was scored over every image in data/train (train ∪ test,
# 552 rows), so before trusting any "test set" statistic we need to know
# which images are actually test images, and whether any test image sits
# within SPATIAL_RADIUS_M of a train image. Nearby/overlapping photos have
# correlated coral cover, so a test image within that radius of a train
# image violates the train/test independence assumption (see
# scripts/train.py::enforce_spatial_independence, which now prevents this
# for splits built going forward -- this section audits/cleans up eval runs,
# like this v1.0 one, that predate that check).
#
# Set EXCLUDE_INDEPENDENCE_VIOLATIONS <- TRUE to drop the violating images
# from eval.dat before any stats/plots below are computed.

SPATIAL_RADIUS_M <- 2  # meters; matches config.SPATIAL_RADIUS
EXCLUDE_INDEPENDENCE_VIOLATIONS <- FALSE

split.dat <- read_csv("data/performance/train_test_split_metadata.csv", show_col_types = FALSE) %>%
  select(image_id, split) %>%
  distinct(image_id, .keep_all = TRUE)

eval.dat <- eval.dat %>% left_join(split.dat, by = "image_id")

coords <- eval.dat %>% select(NorthPhoto_UTM, EastPhoto_UTM)
has_coords <- complete.cases(coords) & !is.na(eval.dat$split)

eval.dat$violates_independence <- FALSE

# Fixed-radius nearest-neighbor search via a kd-tree (dbscan::frNN) instead
# of an O(N^2) all-pairs distance matrix, then connected components
# (igraph) over the resulting "within radius" graph: if a test image is
# only indirectly close to a train image (through a chain of other nearby
# photos), it still leaks, so any cluster touching both splits should be
# treated as a violation, not just directly-adjacent pairs.
coord_mat <- as.matrix(coords[has_coords, ])
nn <- dbscan::frNN(coord_mat, eps = SPATIAL_RADIUS_M)

edges <- do.call(rbind, lapply(seq_along(nn$id), function(i) {
  nbrs <- nn$id[[i]]
  nbrs <- nbrs[nbrs > i]  # de-duplicate: keep each edge once
  if (length(nbrs) == 0) return(NULL)
  cbind(i, nbrs)
}))

g <- igraph::graph_from_data_frame(
  d = as.data.frame(edges),
  vertices = data.frame(name = seq_len(nrow(coord_mat))),
  directed = FALSE
)
cluster_id <- igraph::components(g)$membership

split_sub <- eval.dat$split[has_coords]
violating_local <- logical(length(cluster_id))
for (cl in unique(cluster_id)) {
  members <- which(cluster_id == cl)
  if (length(members) < 2) next
  splits_in_cluster <- unique(split_sub[members])
  if ("train" %in% splits_in_cluster && "test" %in% splits_in_cluster) {
    test_members <- members[split_sub[members] == "test"]
    violating_local[test_members] <- TRUE
  }
}
eval.dat$violates_independence[has_coords] <- violating_local

violating_ids <- eval.dat %>% filter(violates_independence) %>% pull(image_id)
n_missing_coords <- sum(!has_coords & eval.dat$split == "test", na.rm = TRUE)

cat(length(violating_ids), "test image(s) violate the", SPATIAL_RADIUS_M, "m independence assumption:\n")
print(violating_ids)
if (n_missing_coords > 0) {
  cat(n_missing_coords, "test image(s) lack usable GPS coordinates and could not be checked.\n")
}
# ---------------------------------------------------------------------------

# --- IoU / F1 / Dice for Live Coral Cover, derived from the performance CSV ---
# The CSV only stores per-image scalar `accuracy`, `coral_cover` (true), and
# `coral_cover_pred` -- not the underlying pixel masks -- but for a single
# foreground class (coral vs. non-coral) those three numbers fully
# determine the 2x2 confusion matrix, so IoU/F1/Dice can be recovered with
# no mask data at all:
#
#   P = coral_cover      = (TP + FN) / N   (true coral proportion)
#   Q = coral_cover_pred = (TP + FP) / N   (predicted coral proportion)
#   A = accuracy         = (TP + TN) / N   (pixel agreement, both classes)
#   1 = TP + FP + FN + TN                  (proportions of N sum to 1)
#
# Solving the linear system for the intersection (TP) gives:
#   TP = (A + P + Q - 1) / 2
# from which IoU = TP / (P + Q - TP) and Dice = F1 = 2*TP / (P + Q)
# (Dice and F1 are the same quantity for a single foreground class).
#
# One wrinkle: `accuracy` (Segmenter.accuracy) is computed over the full
# 1024x1024 frame, while `coral_cover`/`coral_cover_pred` (Segmenter.coral_cover)
# exclude the fixed CROP_SPACE border pixels from their denominator. That
# border is never coral in either mask, so it contributes only true
# negatives -- it's subtracted out below to put accuracy on the same
# (cropped) denominator as P and Q before solving.

CROP_SPACE_PX <- 7130
IMG_AREA_PX <- 1024 * 1024

eval.dat <- eval.dat %>%
  mutate(
    accuracy_cropped = (accuracy * IMG_AREA_PX - CROP_SPACE_PX) / (IMG_AREA_PX - CROP_SPACE_PX),
    tp_lcc = (accuracy_cropped + coral_cover + coral_cover_pred - 1) / 2,
    fp_lcc = coral_cover_pred - tp_lcc,
    fn_lcc = coral_cover - tp_lcc,
    tn_lcc = 1 - coral_cover - coral_cover_pred + tp_lcc,
    iou_lcc = tp_lcc / (coral_cover + coral_cover_pred - tp_lcc),
    dice_lcc = 2 * tp_lcc / (coral_cover + coral_cover_pred),
    f1_lcc = dice_lcc  # equivalent to Dice for a single foreground class
  )

n_invalid_lcc <- sum(eval.dat$tp_lcc < 0 | eval.dat$fp_lcc < 0 |
                        eval.dat$fn_lcc < 0 | eval.dat$tn_lcc < 0, na.rm = TRUE)
if (n_invalid_lcc > 0) {
  cat(n_invalid_lcc, "image(s) produced a negative confusion-matrix component",
      "(accuracy/coverage figures were mutually inconsistent for that row);",
      "IoU/Dice/F1 for those rows should be treated with caution.\n")
}

cat("Live Coral Cover segmentation quality (derived from accuracy + coverage):\n")
print(
  eval.dat %>%
    summarise(
      mean_iou = mean(iou_lcc, na.rm = TRUE),
      median_iou = median(iou_lcc, na.rm = TRUE),
      mean_dice_f1 = mean(dice_lcc, na.rm = TRUE),
      median_dice_f1 = median(dice_lcc, na.rm = TRUE)
    )
)
# ---------------------------------------------------------------------------

# Reports the effect of removing independence violators, including IoU/Dice/F1
# alongside accuracy/bias metrics.
summarize_independence_effect <- function(df, label) {
  df$big_mask <- (abs(df$accuracy - df$coral_cover) <= 0.1) & df$coral_cover < 0.1
  out <- tibble(
    set = label,
    n = nrow(df),
    prop_big_mask = mean(df$big_mask, na.rm = TRUE),
    mean_accuracy = mean(df$accuracy, na.rm = TRUE),
    median_accuracy_adj = median(df$accuracy[df$big_mask == 0], na.rm = TRUE),
    mean_cover_bias = mean(df$coral_cover_pred - df$coral_cover, na.rm = TRUE),
    mean_bleached_bias = mean(df$pct_bleached_pred - df$pct_bleached_true, na.rm = TRUE)
  )
  if ("iou_lcc" %in% colnames(df)) {
    out$mean_iou_lcc <- mean(df$iou_lcc, na.rm = TRUE)
    out$mean_dice_f1_lcc <- mean(df$dice_lcc, na.rm = TRUE)
  }
  out
}

independence_effect <- bind_rows(
  summarize_independence_effect(eval.dat, "all images (current)"),
  summarize_independence_effect(eval.dat %>% filter(!violates_independence), "all images, violators removed"),
  summarize_independence_effect(eval.dat %>% filter(split == "test"), "test split only"),
  summarize_independence_effect(eval.dat %>% filter(split == "test", !violates_independence), "test split, violators removed")
)
print(independence_effect)

if (EXCLUDE_INDEPENDENCE_VIOLATIONS) {
  eval.dat <- eval.dat %>% filter(!violates_independence)
}
# ---------------------------------------------------------------------------

# --- Custom GGplot Theme -------------------------------------------------

library(sysfonts)

font_add("cm", regular="fonts/cmunrm.ttf")
showtext::showtext_auto()

nf.theme <- theme_minimal(base_size = 36, base_family = "cm") +
  theme(
    #axis.text.x = element_text(angle = 75, hjust = 1),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    axis.text = element_text(face = "bold"),
    panel.background = element_rect(fill = "#F7F7F7"),
    panel.grid.major = element_line(color = "#E3E3E3"),
    panel.grid.minor = element_line(color = "#F0F0F0"),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

# ------------------------------------------------------------------------

#big masks
tolerance = 0.1
eval.dat$big_mask = (abs(eval.dat$accuracy-eval.dat$coral_cover) <= tolerance) & eval.dat$coral_cover < 0.1

#What proportion of our data did we accept a big mask for?
mean(eval.dat$big_mask)

#Adjusted accuracy
median(eval.dat$accuracy[eval.dat$big_mask == 0])

#True coral covers
coral.cover.true <- eval.dat[, c("coral_cover", colnames(eval.dat)[str_detect(colnames(eval.dat), "true")])] %>%
  as.matrix()

coral.cover.true[, 2:ncol(coral.cover.true)] <- sweep(
  coral.cover.true[, 2:ncol(coral.cover.true)],
  1,
  coral.cover.true[, 1],
  "/"
)
coral.cover.true[is.na(coral.cover.true)] <- 0

coral.cover.pred <- eval.dat[, colnames(eval.dat)[str_detect(colnames(eval.dat), "pred")]] %>%
  as.matrix()

coral.cover.pred[, 2:ncol(coral.cover.pred)] <- sweep(
  coral.cover.pred[, 2:ncol(coral.cover.pred)],
  1,
  coral.cover.pred[, 1],
  "/"
)

coral.cover.pred[is.na(coral.cover.pred)] <- 0
coral.cover.pred[is.infinite(coral.cover.pred)] <- 0

bias.mat <- coral.cover.pred-coral.cover.true
apply(bias.mat, 2, function(x)mean(x, na.rm=T))
apply(bias.mat, 2, function(x)median(x, na.rm=T))

#Biases by bleaching/genus
bleached.cover.biases <- (1-coral.cover.pred[,9:ncol(coral.cover.pred)])-(1-coral.cover.true[,9:ncol(coral.cover.true)])
colMeans(bleached.cover.biases)

################################################################################

true.cols <- c("coral_cover", colnames(eval.dat)[str_detect(colnames(eval.dat), "true")])
pred.cols <- colnames(eval.dat)[str_detect(colnames(eval.dat), "pred")]
classes = c("coral_cover", "pct_bleached", 
            sapply(true.cols[3:length(true.cols)], function(x){
              str_split(x, "__") %>% .[[1]] %>% last()
            })
)
classes[9:length(classes)] = str_c(classes[9:length(classes)], ":healthy")

coral.cover <- tibble(
  type = rep(c("True", "Pred"), each=prod(dim(bias.mat))),
  class = rep(c(unname(classes), unname(classes)), each=nrow(bias.mat)),
  value = c(coral.cover.true, coral.cover.pred)
)

ggplot(coral.cover, aes(x = value, fill = type, color = type)) +
  geom_density(alpha = 0.4) +
  facet_wrap(~class, scales = "free") +
  labs(title = "True vs Predicted Coral Cover Densities",
       x = "Coral Cover", y = "Density") +
  nf.theme

################################################################################

library(showtext)
library(patchwork)

# Create the output directory if it doesn't exist
save_dir <- "figures/segmentation.v.1.0"
dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)

# Custom plot size (in inches), dpi 100 → 1200x800 px
plot_width <- 4
plot_height <- 8/3
dpi_val <- 300

# ---- Plot 1: Accuracy vs Depth ----
p1 <- ggplot(eval.dat, aes(x = Depth_WaterSurface, y = accuracy)) +
  geom_point(alpha = 0.6) +
  labs(x = "Depth (m)", y = "Accuracy", title = "Accuracy vs Depth") +
  guides(fill = "none", color = "none") +
  nf.theme

ggsave(filename = file.path(save_dir, "accuracy_vs_depth.png"),
       plot = p1, width = plot_width, height = plot_height, dpi = dpi_val)

# ---- Plot 2: Histogram of Accuracies ----
p2 <- ggplot(eval.dat, aes(x = accuracy)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  geom_vline(aes(xintercept = median(accuracy, na.rm = TRUE)),
             color = "black", linetype = "dashed") +
  labs(x = "Accuracy", y = "Count", title = "Histogram of Accuracies") +
  guides(fill = "none", color = "none") +
  nf.theme

ggsave(filename = file.path(save_dir, "accuracy_histogram.png"),
       plot = p2, width = plot_width, height = plot_height, dpi = dpi_val)

# ---- Plot 3: Histogram of LCC IoU ----
p3 <- ggplot(eval.dat, aes(x = iou_lcc)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  geom_vline(aes(xintercept = median(iou_lcc, na.rm = TRUE)),
             color = "black", linetype = "dashed") +
  labs(x = "IoU", y = "Count", title = "Histogram of Live Coral Cover IoU") +
  guides(fill = "none", color = "none") +
  nf.theme

ggsave(filename = file.path(save_dir, "lcc_iou_histogram.png"),
       plot = p3, width = plot_width, height = plot_height, dpi = dpi_val)

# ---- Plot 4: Histogram of LCC Dice / F1 ----
p4 <- ggplot(eval.dat, aes(x = dice_lcc)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  geom_vline(aes(xintercept = median(dice_lcc, na.rm = TRUE)),
             color = "black", linetype = "dashed") +
  labs(x = "Dice / F1", y = "Count", title = "Histogram of Live Coral Cover Dice/F1") +
  guides(fill = "none", color = "none") +
  nf.theme

ggsave(filename = file.path(save_dir, "lcc_dice_f1_histogram.png"),
       plot = p4, width = plot_width, height = plot_height, dpi = dpi_val)

# ---- Class-wise Density and Bias Plots ----
unique_classes <- unique(coral.cover$class)

for (class_name in unique_classes) {
  dat.subset <- coral.cover %>% filter(class == class_name)
  
  # Density Plot
  density_plot <- ggplot(dat.subset, aes(x = value, fill = type, color = type)) +
    geom_density(alpha = 0.4) +
    coord_cartesian(xlim = c(0, 1)) +
    labs(x = "Coral Cover", y = "Density", title = str_c("True vs Predicted Coral Cover - ", class_name)) +
    guides(fill = "none", color = "none") +
    nf.theme
  
  # Save Density Plot
  ggsave(filename = file.path(save_dir, str_c("coral_cover_density_", str_replace_all(class_name, "[:/ ]", "_"), ".png")),
         plot = density_plot, width = plot_width, height = plot_height, dpi = dpi_val)
  
  # Bias Plot
  true_vals <- dat.subset %>% filter(type == "True") %>% pull(value)
  pred_vals <- dat.subset %>% filter(type == "Pred") %>% pull(value)
  bias <- pred_vals - true_vals
  bias_df <- tibble(bias = bias)
  
  bias_plot <- ggplot(bias_df, aes(x = bias)) +
    geom_histogram(bins = 30, fill = "salmon", color = "black") +
    geom_vline(aes(xintercept = median(bias, na.rm = TRUE)),
               color = "black", linetype = "dashed") +
    labs(x = "Bias (Predicted - True)", y = "Count", title = str_c("Histogram of Coral Cover Bias - ", class_name)) +
    guides(fill = "none", color = "none") +
    nf.theme
  
  # Save Bias Plot
  ggsave(filename = file.path(save_dir, str_c("coral_cover_bias_", str_replace_all(class_name, "[:/ ]", "_"), ".png")),
         plot = bias_plot, width = plot_width, height = plot_height, dpi = dpi_val)
}

################################################################################

# --- Genus / Bleaching Discrimination: Jaccard index, F1, Dice from confusion_matrix.txt ---
# IMPORTANT terminology note: this is NOT the same kind of quantity as the
# pixel-area-based segmentation "IoU" computed above for Live Coral Cover,
# even though both use the identical |A intersect B| / |A union B| formula.
# Here, the "set" being intersected/unioned is a set of CLASSIFIED MASK
# INSTANCES (each unit is one candidate mask, already segmented -- there is
# no notion of spatial/boundary overlap involved at all), whereas pixel IoU
# measures agreement between two spatial regions. To avoid the two being
# confused for the same measurement, this instance-count version is called
# the Jaccard index here (matching sklearn.metrics.jaccard_score's usage
# for multiclass classification), and "IoU" is reserved for the pixel-level
# LCC metrics elsewhere in this file.
#
# Unlike the coverage CSV, a confusion matrix already contains the full joint
# count for every (true, predicted) class pair, so per-class TP/FP/FN read
# directly off it -- no missing-information problem, no algebra needed:
#   TP_c = M[c, c]              (correctly classified masks of class c)
#   FP_c = colSum(c) - TP_c     (masks predicted c but truly something else)
#   FN_c = rowSum(c) - TP_c     (masks truly c but predicted something else)
#   Jaccard_c = TP_c / (TP_c + FP_c + FN_c);  F1_c = Dice_c = 2*TP_c / (2*TP_c + FP_c + FN_c)
#
# Genus- and bleaching-level matrices are obtained by summing groups of rows/
# columns of the raw 17-class (genus x bleach-status, + noncoral) matrix
# together (e.g. all "<genus>:bleached" + "<genus>:healthy" cells collapse
# into one "<genus>" cell) before applying the same formula.

CONFUSION_MATRIX_PATH <- "models/filter_res34_5_v18/confusion_matrix.txt"

# Parses the `print(confusion_matrix(...))` + `labels: {...}` dump written by
# filter.py into a labeled matrix. Numbers are pulled only from within the
# array([[ ... ]], dtype block so "int64" in the dtype suffix is never
# mistaken for matrix entries.
parse_confusion_matrix <- function(path) {
  txt <- paste(readLines(path), collapse = " ")

  array_match <- regmatches(txt, regexpr("array\\(\\[\\[(.*?)\\]\\],\\s*dtype", txt, perl = TRUE))
  nums <- as.numeric(unlist(regmatches(array_match, gregexpr("-?\\d+", array_match))))

  labels_match <- regmatches(txt, regexpr("labels:\\s*(\\{.*\\})\\s*$", txt, perl = TRUE))
  labels_json <- sub("^labels:\\s*", "", labels_match)
  label_map <- fromJSON(labels_json)

  k <- length(label_map)
  cm <- matrix(nums, nrow = k, ncol = k, byrow = TRUE)
  class_names <- names(sort(unlist(label_map)))  # order by index 0..k-1
  dimnames(cm) <- list(true = class_names, pred = class_names)
  cm
}

# Per-class TP/FP/FN/precision/recall/F1/Jaccard from a (possibly collapsed)
# confusion matrix. `jaccard` here is the instance-count Jaccard index, NOT
# pixel-level IoU -- see the note above.
confusion_matrix_metrics <- function(cm) {
  tibble(
    class = rownames(cm),
    tp = diag(cm),
    fp = colSums(cm) - diag(cm),
    fn = rowSums(cm) - diag(cm)
  ) %>%
    mutate(
      precision = tp / (tp + fp),
      recall = tp / (tp + fn),
      f1 = 2 * tp / (2 * tp + fp + fn),
      jaccard = tp / (tp + fp + fn)
    )
}

# Sums groups of rows/cols together to collapse a fine-grained confusion
# matrix (e.g. genus:bleached / genus:healthy) into coarser classes.
collapse_confusion_matrix <- function(cm, groups) {
  group_names <- names(groups)
  out <- matrix(0, nrow = length(groups), ncol = length(groups),
                dimnames = list(group_names, group_names))
  for (gi in group_names) {
    for (gj in group_names) {
      out[gi, gj] <- sum(cm[groups[[gi]], groups[[gj]], drop = FALSE])
    }
  }
  out
}

cm_raw <- parse_confusion_matrix(CONFUSION_MATRIX_PATH)

genus_groups <- list(
  branching_acropora = c("branching_acropora:bleached", "branching_acropora:healthy"),
  finger_acropora    = c("finger_acropora:bleached", "finger_acropora:healthy"),
  finger_porites     = c("finger_porites:bleached", "finger_porites:healthy"),
  knob_porites       = c("knob_porites:bleached", "knob_porites:healthy"),
  mounding_          = c("mounding_:bleached", "mounding_:healthy"),
  oddball            = c("oddball:bleached", "oddball:healthy"),
  pocillopora_       = c("pocillopora_:bleached", "pocillopora_:healthy"),
  table_acropora     = c("table_acropora:bleached", "table_acropora:healthy"),
  noncoral           = "noncoral"
)
cm_genus <- collapse_confusion_matrix(cm_raw, genus_groups)
genus_metrics <- confusion_matrix_metrics(cm_genus)

bleach_groups <- list(
  bleached = grep(":bleached$", colnames(cm_raw), value = TRUE),
  healthy  = grep(":healthy$", colnames(cm_raw), value = TRUE),
  noncoral = "noncoral"
)
cm_bleach <- collapse_confusion_matrix(cm_raw, bleach_groups)
# Coral-only subset excludes the noncoral row/col: bleaching status is only
# meaningful for masks that both sides agree are actually coral.
cm_bleach_coral_only <- cm_bleach[c("bleached", "healthy"), c("bleached", "healthy")]
bleach_metrics <- confusion_matrix_metrics(cm_bleach_coral_only)

cat("Genus discrimination (mask-count Jaccard/F1/Dice, from confusion_matrix.txt):\n")
print(genus_metrics)
cat("macro F1:", mean(genus_metrics$f1), " macro Jaccard:", mean(genus_metrics$jaccard),
    " overall accuracy:", sum(diag(cm_genus)) / sum(cm_genus), "\n\n")

cat("Bleaching discrimination, coral-only (mask-count Jaccard/F1/Dice):\n")
print(bleach_metrics)

# ---- Confusion matrix heatmaps (row-normalized: each row sums to 1) ----
plot_confusion_heatmap <- function(cm, title, filename) {
  cm_norm <- cm / rowSums(cm)
  cm_df <- as_tibble(cm_norm, rownames = "true") %>%
    pivot_longer(-true, names_to = "pred", values_to = "prop") %>%
    mutate(true = factor(true, levels = rownames(cm)),
           pred = factor(pred, levels = colnames(cm)))

  p <- ggplot(cm_df, aes(x = pred, y = true, fill = prop)) +
    geom_tile() +
    geom_text(aes(label = sprintf("%.2f", prop)), size = 3) +
    scale_fill_gradient(low = "white", high = "steelblue", limits = c(0, 1)) +
    labs(x = "Predicted", y = "True", title = title, fill = "Row %") +
    theme_minimal(base_size = 14) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(hjust = 0.5, face = "bold"))

  ggsave(filename = file.path(save_dir, filename), plot = p,
         width = plot_width * 2, height = plot_height * 2, dpi = dpi_val)
  p
}

plot_confusion_heatmap(cm_genus, "Genus Discrimination", "confusion_matrix_genus.png")
plot_confusion_heatmap(cm_bleach_coral_only, "Bleaching Discrimination (coral-only)", "confusion_matrix_bleaching.png")
################################################################################

# ---- Ensemble weight heatmaps (one per ensemble_method) --------------------
# Reproduces figures/paper/ensemble_interpretable_weights_heatmap.png's
# style ("Ensemble Responsibility by Class") for every fitted ensemble
# method in data/performance/ensemble_weights.csv (see
# scripts/export_ensemble_weights.py, which must be re-run after
# scripts/retrain_ensemble.py to refresh this CSV). The underlying quantity
# differs by method -- these three (em/adam/reweight) are latent-class
# mixtures with a genuine per-example responsibility, matching the original
# figure exactly; multinomial/linear/nn don't have that structure, so each
# gets the closest natural per-(model, class) quantity for its own
# parameterization instead (see that script's docstring) -- titled and
# color-scaled accordingly rather than mislabeled as "responsibility".

ensemble_dir <- "figures/ensemble"
dir.create(ensemble_dir, showWarnings = FALSE, recursive = TRUE)

ew <- read_csv("data/performance/ensemble_weights.csv", show_col_types = FALSE)

pretty_class_name <- function(x) {
  is_bleached <- str_detect(x, ":bleached$")
  is_healthy  <- str_detect(x, ":healthy$")
  base <- str_remove(x, ":(bleached|healthy)$")
  base <- str_remove(base, "_$")                 # trailing underscore (mounding_, pocillopora_)
  base <- str_to_title(str_replace_all(base, "_", " "))
  base[base == "Noncoral"] <- "Non-Coral"
  out <- base
  out[is_bleached] <- paste0(base[is_bleached], " (Bleached)")
  out[is_healthy]  <- paste0(base[is_healthy], " (Healthy)")
  out
}

ensemble_method_titles <- c(
  em          = "Ensemble Responsibility by Class (EM)",
  adam        = "Ensemble Responsibility by Class (Adam)",
  reweight    = "Ensemble Responsibility by Class (Reweight)",
  multinomial = "Per-Submodel Calibration Weight by Class (Multinomial)",
  linear      = "Per-Submodel Self-Weight by Class (Linear)",
  nn          = "Per-Submodel Input Sensitivity by Class (Neural Net, Approx.)"
)

plot_ensemble_weights <- function(method_df, method_name) {
  has_share <- !all(is.na(method_df$model_share))

  model_labels <- method_df %>%
    distinct(model, model_share) %>%
    arrange(model) %>%
    mutate(label = if (has_share) sprintf("Model %d (%.2f%%)", model, 100 * model_share)
                   else sprintf("Model %d", model))

  df <- method_df %>%
    mutate(
      class_label = pretty_class_name(class_name),
      class_label = factor(class_label, levels = unique(class_label[order(class_name)])),
      model_label = factor(model_labels$label[model], levels = model_labels$label),
      # White text on the darker (upper-half-of-THIS-plot's-own-range) tiles,
      # black otherwise -- each method has its own value range, so the
      # threshold is relative to this plot's own min/max, not a fixed cutoff.
      text_color = ifelse(value > (min(value) + max(value)) / 2, "white", "black")
    )

  p <- ggplot(df, aes(x = class_label, y = model_label, fill = value)) +
    geom_tile(color = "white", linewidth = 0.6) +
    geom_text(aes(label = sprintf("%.2f", value), color = text_color),
              family = "cm", size = 7.5, show.legend = FALSE) +
    scale_color_identity() +
    labs(title = ensemble_method_titles[[method_name]], x = NULL, y = NULL, fill = NULL) +
    nf.theme +
    theme(
      axis.text.x = element_text(angle = 60, hjust = 1, size = rel(1.0)),
      axis.text.y = element_text(size = rel(1.1)),
      panel.grid = element_blank(),
      legend.position = "none"
    )

  # One consistent sequential palette across all six methods (a diverging
  # scale isn't informative here since the fitted values don't straddle a
  # natural midpoint for every method).
  p <- p + scale_fill_distiller(palette = "YlGnBu", direction = 1)

  ggsave(filename = file.path(ensemble_dir, sprintf("ensemble_weights_%s.png", method_name)),
         plot = p, width = 14, height = 6, dpi = dpi_val)
  p
}

for (m in unique(ew$method)) {
  plot_ensemble_weights(ew %>% filter(method == m), m)
}
################################################################################

# --- F1 / Dice / Jaccard for each submodel + the ensemble, from model_performance.txt ---
# Like the confusion-matrix section above (and unlike the pixel-level LCC
# IoU section), this is an instance-count metric, not a spatial one -- so it
# is labeled `jaccard`, not `iou`, for the same reason. It's also a coarser
# collapse than the per-genus/per-bleach Jaccard above: precision/recall in
# model_performance.txt come from filter.py's test()/validate(), which
# collapse ALL 16 non-noncoral classes into one "coral" bucket (a true
# branching_acropora mask predicted as finger_porites still counts as a
# true positive here) -- i.e. "is this mask coral at all", not "which coral".
#
# F1 = Dice = 2*P*R/(P+R) is just the harmonic mean of precision and recall,
# so it needs nothing beyond the precision/recall columns already in the
# file. Jaccard is recoverable too, via the standard Dice<->Jaccard identity
# Jaccard = F1/(2-F1) (equivalently P*R/(P+R-P*R)) -- no raw TP/FP/FN counts
# needed, since precision and recall already summarize that same binary
# (coral vs. noncoral) confusion matrix. This is the same identity used for
# the coral-cover IoU/Dice section above (there derived the other way:
# Dice from IoU) -- the algebra is identical either way, only the
# underlying population (pixels vs. mask instances) differs.
#
# Sanity check: collapsing confusion_matrix.txt's raw 17-class matrix into
# coral-vs-noncoral the same way (sum the full 16x16 coral sub-block, not
# just its diagonal) reproduces the Ensemble out-of-sample row here exactly
# (precision=0.9433, recall=0.9577, on the same 1,762 held-out masks) --
# confirming both files were scored on the same split.

MODEL_PERFORMANCE_PATH <- "models/filter_res34_5_v18/model_performance.txt"

# Parses the fixed-width "Model | Accuracy | Precision | Recall || Accuracy |
# Precision | Recall |" table written by filter.py / generate_filter_reports.py.
parse_model_performance <- function(path) {
  lines <- readLines(path, warn = FALSE)
  data_lines <- lines[str_detect(lines, "^\\s*(\\d+|Ensemble)\\s*\\|")]

  map(data_lines, function(line) {
    fields <- str_split(line, "\\|")[[1]] %>% str_trim()
    fields <- fields[fields != ""]
    tibble(
      model = fields[1],
      in_accuracy = as.numeric(fields[2]),
      in_precision = as.numeric(fields[3]),
      in_recall = as.numeric(fields[4]),
      oos_accuracy = as.numeric(fields[5]),
      oos_precision = as.numeric(fields[6]),
      oos_recall = as.numeric(fields[7])
    )
  }) %>% bind_rows()
}

# Adds `<prefix>_f1`, `<prefix>_dice` (identical to f1), and `<prefix>_jaccard`
# (instance-count Jaccard index, NOT pixel-level IoU) columns computed from
# an existing precision/recall pair.
add_f1_dice_jaccard <- function(df, precision_col, recall_col, prefix) {
  p <- df[[precision_col]]
  r <- df[[recall_col]]
  f1 <- 2 * p * r / (p + r)
  df[[paste0(prefix, "_f1")]] <- f1
  df[[paste0(prefix, "_dice")]] <- f1
  df[[paste0(prefix, "_jaccard")]] <- f1 / (2 - f1)
  df
}

model_performance <- parse_model_performance(MODEL_PERFORMANCE_PATH) %>%
  add_f1_dice_jaccard("in_precision", "in_recall", "in") %>%
  add_f1_dice_jaccard("oos_precision", "oos_recall", "oos")

cat("Per-model / ensemble F1, Dice, Jaccard -- coral-vs-noncoral detection, mask-count based\n")
cat("(derived from precision & recall in model_performance.txt):\n")
print(model_performance %>% select(model, starts_with("in_"), starts_with("oos_")))

ensemble_oos <- model_performance %>% filter(model == "Ensemble")
cat(sprintf(
  "\nEnsemble, out-of-sample coral detection: F1/Dice = %.4f, Jaccard = %.4f\n",
  ensemble_oos$oos_f1, ensemble_oos$oos_jaccard
))
################################################################################

# --- Per-taxonomy PIXEL-level IoU/Dice/precision/recall, macro-averaged, ---
# --- with bootstrap confidence intervals                                 ---
# Genuine spatial-overlap IoU/Dice/precision/recall per taxonomy, macro-
# averaged with bootstrap CIs -- NOT the instance-count Jaccard from
# confusion_matrix.txt above. Computed by scripts/train.py's eval loop
# directly from predicted vs. ground-truth masks (see
# pixel_confusion_metrics/union_mask/taxonomy_records there), one row per
# (image, taxonomy). Requires running that eval loop at least once; this
# section no-ops with a message if the CSV isn't there yet.

TAXONOMY_METRICS_PATH <- "data/performance/coral_segmenter_taxonomy_metrics.v.1.0.csv"

if (!file.exists(TAXONOMY_METRICS_PATH)) {

  cat("\n", TAXONOMY_METRICS_PATH, "not found -- run train.py's eval loop to generate it",
      "before this section can compute pixel-level per-taxonomy IoU/Dice/precision/recall.\n")

} else {

  taxonomy_df <- read_csv(TAXONOMY_METRICS_PATH, show_col_types = FALSE)

  # Nonparametric bootstrap CI for the mean of one metric column, resampling
  # images (the natural iid unit here -- each image contributes one value
  # per taxonomy) with replacement.
  bootstrap_ci <- function(x, n_boot = 2000, seed = 42) {
    set.seed(seed)
    x <- x[!is.na(x)]
    if (length(x) == 0) return(c(low = NA_real_, high = NA_real_))
    boot_means <- replicate(n_boot, mean(sample(x, length(x), replace = TRUE)))
    q <- quantile(boot_means, c(0.025, 0.975))
    c(low = unname(q[1]), high = unname(q[2]))
  }

  # Per-taxonomy mean + 95% CI for one metric, one taxonomy at a time
  # (images are resampled independently per taxonomy here).
  summarize_metric_ci <- function(df, metric) {
    df %>%
      group_by(taxonomy) %>%
      group_modify(~ {
        ci <- bootstrap_ci(.x[[metric]])
        tibble(n_images = nrow(.x), mean = mean(.x[[metric]], na.rm = TRUE),
               ci_low = ci["low"], ci_high = ci["high"])
      }) %>%
      ungroup() %>%
      rename_with(~ paste0(metric, "_", .x), c(mean, ci_low, ci_high))
  }

  # Macro-average across `classes` (unweighted mean of each class's own
  # mean), with its own bootstrap CI -- resampling IMAGES jointly across all
  # classes each draw (not each class independently), since every class's
  # value for a given image comes from the same underlying prediction, so
  # resampling them separately would understate the true correlation and
  # produce an overly narrow (wrong) CI for the macro-average.
  macro_bootstrap <- function(df, metric, classes, n_boot = 2000, seed = 42) {
    set.seed(seed)
    wide <- df %>%
      filter(taxonomy %in% classes) %>%
      select(image_id, taxonomy, value = all_of(metric)) %>%
      pivot_wider(names_from = taxonomy, values_from = value)

    n <- nrow(wide)
    class_mat <- as.matrix(wide[, classes])
    point_est <- mean(colMeans(class_mat, na.rm = TRUE))

    boot_macro <- replicate(n_boot, {
      idx <- sample(seq_len(n), n, replace = TRUE)
      mean(colMeans(class_mat[idx, , drop = FALSE], na.rm = TRUE))
    })
    ci <- quantile(boot_macro, c(0.025, 0.975))
    tibble(metric = metric, macro_mean = point_est, ci_low = unname(ci[1]), ci_high = unname(ci[2]))
  }

  taxonomy_summary <- summarize_metric_ci(taxonomy_df, "iou") %>%
    left_join(summarize_metric_ci(taxonomy_df, "dice_f1"), by = c("taxonomy", "n_images")) %>%
    left_join(summarize_metric_ci(taxonomy_df, "precision"), by = c("taxonomy", "n_images")) %>%
    left_join(summarize_metric_ci(taxonomy_df, "recall"), by = c("taxonomy", "n_images"))

  cat("Per-taxonomy pixel-level IoU/Dice/precision/recall (mean + 95% bootstrap CI over images):\n")
  print(taxonomy_summary, n = Inf)

  # Macro-average over the 8 mutually-exclusive genus classes only --
  # "all_coral", "bleached", and the "<genus>:healthy" rows are different
  # groupings of the same underlying data, not additional classes in that
  # partition, and would double-count if included in the same average.
  genus_classes <- taxonomy_df %>%
    distinct(taxonomy) %>%
    filter(!taxonomy %in% c("all_coral", "bleached"), !str_detect(taxonomy, ":")) %>%
    pull(taxonomy)

  macro_summary <- bind_rows(
    macro_bootstrap(taxonomy_df, "iou", genus_classes),
    macro_bootstrap(taxonomy_df, "dice_f1", genus_classes),
    macro_bootstrap(taxonomy_df, "precision", genus_classes),
    macro_bootstrap(taxonomy_df, "recall", genus_classes)
  )
  cat("\nMacro-averaged pixel-level metrics across the", length(genus_classes),
      "genus classes (mean + 95% bootstrap CI, jointly resampled over images):\n")
  print(macro_summary, n = Inf)
}
################################################################################
