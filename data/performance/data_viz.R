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

split.dat <- read_csv("data/metadata/train_test_split_metadata.csv", show_col_types = FALSE) %>%
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
    # theme_minimal's default axis.text is noticeably smaller than axis
    # titles (rel(0.8) of base size); bump it closer to full size.
    axis.text = element_text(face = "bold", size = rel(0.95)),
    # NOTE: element_rect()'s `colour` defaults to a visible (near-black)
    # stroke in this ggplot2 version when left unspecified -- explicitly
    # setting colour = NA here is required, otherwise panel.background
    # silently draws a border around the panel that panel.grid then paints
    # over at every intersection (the "grid lines go over the border" bug).
    panel.background = element_rect(fill = "#F7F7F7", colour = NA),
    # The real, intentional panel border: panel.border is always drawn
    # after panel.grid in ggplot's rendering order, so this one frames the
    # grid cleanly instead of being painted over by it.
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
    panel.grid.major = element_line(color = "#E3E3E3", linewidth = 0.3),
    panel.grid.minor = element_line(color = "#F0F0F0", linewidth = 0.15),
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

  # The array literal is terminated either by "]], dtype=..." (older dumps,
  # via numpy's own repr) or plain "]])" with no dtype suffix (current
  # confusion_matrix.txt). Capture ONLY the inner "[[ ... ]]" content (group
  # 1) so a "dtype=int64"-style suffix, if present, never contributes stray
  # digits to the number extraction below.
  m <- regexec("array\\(\\[\\[(.*?)\\]\\](?:,\\s*dtype[^)]*)?\\)", txt, perl = TRUE)
  array_inner <- regmatches(txt, m)[[1]][2]
  nums <- as.numeric(unlist(regmatches(array_inner, gregexpr("-?\\d+", array_inner))))

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

# Full 3-way (bleached/healthy/noncoral) version -- this is what Figure 7's
# caption actually describes (recall of each of the three row-classes
# against the full row total, INCLUDING cross-confusion with noncoral),
# not the coral-only 2x2 subset above (which excludes noncoral confusions
# from the denominator entirely and so reads higher).
bleach_metrics_3way <- confusion_matrix_metrics(cm_bleach)
cat("\nBleaching discrimination, full 3-way incl. noncoral (matches Figure 7's caption framing):\n")
print(bleach_metrics_3way)

# Shared display-name formatter (genus_bleach-style key -> "Genus (Status)"),
# used by both the confusion-matrix heatmaps below and the ensemble-weight
# heatmaps further down, so every figure's class labels look identical.
pretty_class_name <- function(x) {
  is_bleached <- str_detect(x, ":bleached$") | str_detect(x, "^bleached$")
  is_healthy  <- str_detect(x, ":healthy$") | str_detect(x, "^healthy$")
  base <- str_remove(x, ":(bleached|healthy)$")
  base <- str_remove(base, "_$")                 # trailing underscore (mounding_, pocillopora_)
  base <- str_to_title(str_replace_all(base, "_", " "))
  base[base == "Noncoral"] <- "Non-Coral"
  out <- base
  out[is_bleached & !x %in% c("bleached")] <- paste0(base[is_bleached & !x %in% c("bleached")], " (Bleached)")
  out[is_healthy & !x %in% c("healthy")] <- paste0(base[is_healthy & !x %in% c("healthy")], " (Healthy)")
  out
}

# ---- Confusion matrix heatmaps (row-normalized: each row sums to 1) ----
# text_size/annot_size are tunable per call: a small (e.g. 3x3) matrix has far
# more room per cell than a large one (e.g. 9x9), so it can and should use
# noticeably larger annotation and axis text than the default.
plot_confusion_heatmap <- function(cm, title, filename, text_size = 44, annot_size = 11,
                                    axis_x_rel = 1.0, axis_y_rel = 1.1) {
  dimnames(cm) <- list(true = pretty_class_name(rownames(cm)), pred = pretty_class_name(colnames(cm)))
  cm_norm <- cm / rowSums(cm)
  cm_df <- as_tibble(cm_norm, rownames = "true") %>%
    pivot_longer(-true, names_to = "pred", values_to = "prop") %>%
    mutate(true = factor(true, levels = rev(rownames(cm))),
           pred = factor(pred, levels = colnames(cm)),
           text_color = ifelse(prop > 0.5, "white", "black"),
           # Zero cells get a blank/transparent tile and no annotation,
           # rather than the palette's lightest color plus a "0.00" label.
           prop_fill = ifelse(prop == 0, NA_real_, prop),
           label_text = ifelse(prop == 0, NA_character_, sprintf("%.2f", prop)))

  # Same visual identity as plot_ensemble_weights (Figure 6): YlGnBu tiles,
  # white gridlines, brightness-switched text color, nf.theme + "cm" font.
  p <- ggplot(cm_df, aes(x = pred, y = true, fill = prop_fill)) +
    geom_tile(color = "white", linewidth = 0.6) +
    geom_text(aes(label = label_text, color = text_color),
              family = "cm", size = annot_size, show.legend = FALSE, na.rm = TRUE) +
    scale_color_identity() +
    scale_fill_distiller(palette = "YlGnBu", direction = 1, limits = c(0, 1), na.value = NA) +
    labs(x = "Predicted", y = "True", title = title, fill = NULL) +
    nf.theme +
    theme(
      text = element_text(size = text_size, family = "cm"),
      axis.text.x = element_text(angle = 60, hjust = 1, size = rel(axis_x_rel)),
      axis.text.y = element_text(size = rel(axis_y_rel)),
      panel.grid = element_blank(),
      panel.border = element_blank(),
      legend.position = "none"
    )

  ggsave(filename = file.path(save_dir, filename), plot = p,
         width = 12, height = 10, dpi = dpi_val)
  p
}

plot_confusion_heatmap(cm_genus, "Genus Discrimination", "confusion_matrix_genus.png")
plot_confusion_heatmap(cm_bleach_coral_only, "Bleaching Discrimination (coral-only)", "confusion_matrix_bleaching.png")
# Full 3-way (bleached/healthy/noncoral) version -- this is the one that
# matches Figure 7's caption (which discusses noncoral rejection alongside
# bleached/healthy recall, not just the coral-only 2x2 subset above).
# It's only a 3x3 matrix, so it has much more room per cell than the 9x9
# genus matrix -- scale annotation/axis text up accordingly.
plot_confusion_heatmap(cm_bleach, "Bleaching Discrimination (incl. Non-Coral)", "confusion_matrix_bleaching_3way.png",
                        text_size = 60, annot_size = 20, axis_x_rel = 1.15, axis_y_rel = 1.25)

# Paper-facing copies (Figures 7 and 8), current-vintage confusion matrices.
dir.create("figures/paper", showWarnings = FALSE, recursive = TRUE)
file.copy(file.path(save_dir, "confusion_matrix_bleaching_3way.png"),
          "figures/paper/cm_bleach_healthy_nonc_annotated.png", overwrite = TRUE)
file.copy(file.path(save_dir, "confusion_matrix_genus.png"),
          "figures/paper/cm_genus_annotated.png", overwrite = TRUE)
cat("\nCopied current confusion-matrix heatmaps -> figures/paper/cm_bleach_healthy_nonc_annotated.png (Fig 7), cm_genus_annotated.png (Fig 8)\n")
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

# pretty_class_name() is defined earlier (shared with plot_confusion_heatmap).
.unused_pretty_class_name_placeholder <- function(x) {
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
  reweight    = "Ensemble Responsibility by Class",
  multinomial = "Per-Submodel Calibration Weight by Class (Multinomial)",
  linear      = "Per-Submodel Self-Weight by Class (Linear)",
  nn          = "Per-Submodel Input Sensitivity by Class (Neural Net, Approx.)"
)
# "reweight" is the production ensemble method (config.ENSEMBLE_METHOD) and is
# the one shown in the paper as plain "Ensemble" (Figure 6) -- its plot title
# above deliberately omits a parenthetical method tag, unlike the other five
# (which are shown, if at all, only as internal comparisons -- see the
# ensemble-optimizer methods+results paragraph in paper/main.tex). Its output
# file is additionally copied to a display-friendly name below.

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
              family = "cm", size = 11, show.legend = FALSE) +
    scale_color_identity() +
    labs(title = ensemble_method_titles[[method_name]], x = NULL, y = NULL, fill = NULL) +
    nf.theme +
    theme(
      text = element_text(size = 44, family = "cm"),
      axis.text.x = element_text(angle = 60, hjust = 1, size = rel(1.0)),
      axis.text.y = element_text(size = rel(1.1)),
      panel.grid = element_blank(),
      panel.border = element_blank(),
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

# Paper-facing copy of the production method's heatmap, named/titled plainly
# as "Ensemble" (Figure 6) -- figures/paper/ensemble_interpretable_weights_heatmap.png.
# (figures/paper defined as PAPER_FIG_DIR further below -- hardcoded here
# since this block runs before that point in the script.)
dir.create("figures/paper", showWarnings = FALSE, recursive = TRUE)
file.copy(
  file.path(ensemble_dir, "ensemble_weights_reweight.png"),
  "figures/paper/ensemble_interpretable_weights_heatmap.png",
  overwrite = TRUE
)
cat("\nCopied ensemble_weights_reweight.png -> figures/paper/ensemble_interpretable_weights_heatmap.png (Figure 6)\n")
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

### ============================================================================
### CURRENT PAPER FIGURES/TABLES -- sourced from the Sep-2025 replicate.py run
### under data/performance/inference/ (annotations_coco_{train,test}_metrics.csv,
### incl. the genus_bleach rows added by scripts/compute_genus_bleach_metrics.py).
### This supersedes, for every paper-facing figure/table below, the v1.0-CSV
### section above (which is kept as-is for historical/audit purposes but no
### longer feeds paper/*.png -- both sections were cross-checked to agree to
### within ~0.1-0.7 percentage points on shared images, the residual being the
### old pipeline's fixed CROP_SPACE_PX border exclusion, which this newer,
### COCO-mask-based scoring does not apply).
###
### Derivation: annotations_coco_*_metrics.csv gives tp_px/fp_px/fn_px per
### (image, taxonomy) on a fixed 1024x1024 frame (verified: every image in
### the COCO exports is exactly 1024x1024), which is sufficient to recover,
### for ANY taxonomy row (lcc / bleached / genus / genus:bleach-status):
###   true_frac = (tp+fn)/N   -- ground-truth coverage fraction of the image
###   pred_frac = (tp+fp)/N   -- predicted coverage fraction of the image
###   accuracy  = 1-(fp+fn)/N -- pixel-level coral-vs-noncoral accuracy (only
###                              meaningful for taxonomy=="lcc", where "coral"
###                              is a full partition of the image into 2 classes)
### iou/dice_f1/precision/recall are already columns in the CSV directly.
### ============================================================================

INFERENCE_DIR <- "data/performance/inference"
PAPER_FIG_DIR <- "figures/paper"
dir.create(PAPER_FIG_DIR, showWarnings = FALSE, recursive = TRUE)

IMG_PX <- 1024 * 1024

tax_train <- read_csv(file.path(INFERENCE_DIR, "annotations_coco_train_metrics.csv"), show_col_types = FALSE) %>% mutate(split = "train")
tax_test  <- read_csv(file.path(INFERENCE_DIR, "annotations_coco_test_metrics.csv"),  show_col_types = FALSE) %>% mutate(split = "test")

tax_all <- bind_rows(tax_train, tax_test) %>%
  rename(image_id = image) %>%
  mutate(
    true_frac = (tp_px + fn_px) / IMG_PX,
    pred_frac = (tp_px + fp_px) / IMG_PX,
    accuracy  = 1 - (fp_px + fn_px) / IMG_PX,
    split = factor(split, levels = c("train", "test"), labels = c("Train", "Test"))
  )

# Metadata join (depth, UTM coords) -- for Fig 15 (depth vs accuracy) and any
# future spatial audit of this run, matched on the full roboflow filename.
split_meta <- read_csv("data/metadata/train_test_split_metadata.csv", show_col_types = FALSE) %>%
  select(filename, Depth_WaterSurface, NorthPhoto_UTM, EastPhoto_UTM)

tax_all <- tax_all %>% left_join(split_meta, by = c("image_id" = "filename"))

# All remaining paper figures reuse `nf.theme` (defined above for the
# ensemble-weight heatmaps) rather than a separate, plainer theme, so every
# ggplot in the paper shares one visual identity (font, panel background,
# grid, bold text). Single-panel plots get base_size scaled down slightly
# from the heatmap's 36pt (tuned for a 14x6in canvas) to suit their smaller
# canvases; faceted plots keep closer to the full size since they're wider.
nf.theme.single <- nf.theme + theme(text = element_text(size = 40, family = "cm"))
nf.theme.facet  <- nf.theme + theme(text = element_text(size = 40, family = "cm")) +
  theme(
    # Force square-ish facet panels (rather than the default rectangle
    # dictated by data range), with a modest (not zero) gap to the legend.
    aspect.ratio = 1,
    legend.box.spacing = unit(10, "pt"),
    # Subpanel titles (e.g. "Train"/"Test", "Full Range"/"Zoomed") default to
    # rel(0.8) of the base size, same as axis text -- bump alongside it.
    strip.text = element_text(size = rel(0.95))
  )

# Two-color qualitative palette reused across every train/test (and
# actual/predicted, NeuralReefer/CoralSCOP) comparison, replacing ggplot's
# default hues for visual consistency with the rest of the paper.
PAL2 <- c("#1B9E77", "#D95F02")

fmt_pct <- function(x) sprintf("%.1f%%", 100 * x)

# ---- Figure 2: manual annotation counts per image, train vs test ----------
TRAIN_DIR <- "yellowfin_segment.v18i.coco-segmentation/train"  # matches scripts/config.py

split_lookup <- read_csv("data/metadata/train_test_split_metadata.csv", show_col_types = FALSE) %>%
  select(filename, split) %>%
  distinct(filename, .keep_all = TRUE)

gt_coco <- fromJSON(file.path(TRAIN_DIR, "_annotations.coco.json"))
ann_counts <- gt_coco$annotations %>% count(image_id, name = "n_annotations")
ann_per_image <- gt_coco$images %>%
  select(id, file_name) %>%
  left_join(ann_counts, by = c("id" = "image_id")) %>%
  mutate(n_annotations = replace_na(n_annotations, 0)) %>%
  left_join(split_lookup, by = c("file_name" = "filename")) %>%
  filter(!is.na(split)) %>%
  mutate(split = factor(split, levels = c("train", "test"), labels = c("Train", "Test")))

med2 <- ann_per_image %>% group_by(split) %>% summarise(m = median(n_annotations))
cat("Fig 2 (annotation counts) medians:\n"); print(med2)
cat("Fig 2 sanity check -- total images:", nrow(ann_per_image), "total annotations:", sum(ann_per_image$n_annotations), "\n")

p_fig2 <- ggplot(ann_per_image, aes(x = n_annotations, fill = split)) +
  geom_histogram(binwidth = 5, color = NA, alpha = 0.7, position = "identity") +
  geom_vline(data = med2, aes(xintercept = m, color = split), linetype = "dashed", linewidth = 1, show.legend = FALSE) +
  scale_fill_manual(values = PAL2) +
  scale_color_manual(values = PAL2) +
  facet_wrap(~split, scales = "free_y") +
  labs(title = "Manual Annotations per Image", x = "Number of Annotations", y = "Count", fill = "Split") +
  nf.theme.facet
ggsave(file.path(PAPER_FIG_DIR, "fig2.png"), p_fig2, width = 12, height = 7, dpi = 300)

# ---- Figure 3: ground-truth LCC density, train vs test ----------------------
lcc_true <- tax_all %>% filter(taxonomy == "lcc")
med3 <- lcc_true %>% group_by(split) %>% summarise(m = median(true_frac, na.rm = TRUE))
cat("Fig 3 (true LCC) medians:\n"); print(med3 %>% mutate(pct = fmt_pct(m)))

p_fig3 <- ggplot(lcc_true, aes(x = true_frac, fill = split)) +
  geom_density(alpha = 0.5, color = NA) +
  geom_vline(data = med3, aes(xintercept = m, color = split), linetype = "dashed", linewidth = 1, show.legend = FALSE) +
  scale_x_continuous(labels = scales::percent) +
  scale_fill_manual(values = PAL2) +
  scale_color_manual(values = PAL2) +
  labs(title = "Live Coral Cover (Ground Truth): Train vs Test",
       x = "Live Coral Cover (% of image area)", y = "Density", fill = "Split") +
  nf.theme.single
ggsave(file.path(PAPER_FIG_DIR, "fig3.png"), p_fig3, width = 8, height = 6, dpi = 300)

# ---- Figure 11: predicted LCC density, train vs test -------------------------
# (Document position, not the historical variable-name numbering below --
# Figure 9 was split into two figures (9: qual-results, 10: qual-results-fail)
# at some point after these variable names were first assigned, shifting
# every subsequent figure's TRUE position by +1 relative to its old name;
# Figures 12/13/14 were also later reordered relative to each other. Variable
# names and output filenames here now match each figure's current position in
# submission/main.tex -- keep them in sync if the document order changes again.)
med_fig11 <- lcc_true %>% group_by(split) %>% summarise(m = median(pred_frac, na.rm = TRUE))
cat("Fig 11 (pred LCC) medians:\n"); print(med_fig11 %>% mutate(pct = fmt_pct(m)))

p_fig11 <- ggplot(lcc_true, aes(x = pred_frac, fill = split)) +
  geom_density(alpha = 0.5, color = NA) +
  geom_vline(data = med_fig11, aes(xintercept = m, color = split), linetype = "dashed", linewidth = 1, show.legend = FALSE) +
  scale_x_continuous(labels = scales::percent) +
  scale_fill_manual(values = PAL2) +
  scale_color_manual(values = PAL2) +
  labs(title = "Predicted Live Coral Cover: Train vs Test",
       x = "Predicted Live Coral Cover (% of image area)", y = "Density", fill = "Split") +
  nf.theme.single
ggsave(file.path(PAPER_FIG_DIR, "fig11.png"), p_fig11, width = 8, height = 6, dpi = 300)

# ---- Figure 12: predicted vs actual LCC, faceted by split, quadratic fit ---
p_fig12 <- ggplot(lcc_true, aes(x = true_frac, y = pred_frac)) +
  # shape=16 (not the default 19) avoids a spurious darker rim that shape 19
  # renders around each point under alpha transparency in this ggplot2
  # version; size halved per requested point-size reduction.
  geom_point(alpha = 0.75, color = PAL2[1], size = 1.125, shape = 16) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40") +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE, color = PAL2[2], linewidth = 1.2) +
  scale_x_continuous(labels = scales::percent) +
  scale_y_continuous(labels = scales::percent) +
  facet_wrap(~split) +
  labs(title = "Calibration of Predicted vs. Observed Live Coral Cover",
       x = "Observed LCC", y = "Predicted LCC") +
  nf.theme.facet
ggsave(file.path(PAPER_FIG_DIR, "fig12.png"), p_fig12, width = 13, height = 7.5, dpi = 300)

# ---- Figure 14: bleached coral cover, true vs pred, train & test -----------
bcc <- tax_all %>% filter(taxonomy == "bleached") %>%
  select(image_id, split, true_frac, pred_frac) %>%
  pivot_longer(c(true_frac, pred_frac), names_to = "type", values_to = "value") %>%
  mutate(type = recode(type, true_frac = "Actual", pred_frac = "Predicted"))

med_fig14 <- bcc %>% group_by(split, type) %>% summarise(m = median(value, na.rm = TRUE), .groups = "drop")
cat("Fig 14 (BCC) medians:\n"); print(med_fig14 %>% mutate(pct = fmt_pct(m)))

p_fig14 <- ggplot(bcc, aes(x = value, fill = type)) +
  geom_density(alpha = 0.5, color = NA) +
  geom_vline(data = med_fig14, aes(xintercept = m, color = type), linetype = "dashed", linewidth = 1, show.legend = FALSE) +
  scale_x_continuous(labels = scales::percent) +
  scale_fill_manual(values = PAL2) +
  scale_color_manual(values = PAL2) +
  facet_wrap(~split) +
  labs(title = "Bleached Coral Cover: True vs. Predicted",
       x = "Bleached Coral Cover (% of image area)", y = "Density", fill = NULL) +
  nf.theme.facet
ggsave(file.path(PAPER_FIG_DIR, "fig14.png"), p_fig14, width = 13, height = 7.5, dpi = 300)

# ---- Figure 13: pixel-level segmentation accuracy density, train vs test ---
med_fig13 <- lcc_true %>% group_by(split) %>% summarise(m = median(accuracy, na.rm = TRUE))
cat("Fig 13 (accuracy) medians:\n"); print(med_fig13 %>% mutate(pct = fmt_pct(m)))

p_fig13 <- ggplot(lcc_true, aes(x = accuracy, fill = split)) +
  geom_density(alpha = 0.5, color = NA) +
  geom_vline(data = med_fig13, aes(xintercept = m, color = split), linetype = "dashed", linewidth = 1, show.legend = FALSE) +
  scale_x_continuous(labels = scales::percent) +
  scale_fill_manual(values = PAL2) +
  scale_color_manual(values = PAL2) +
  labs(title = "Pixel-Level Segmentation Accuracy: Train vs Test",
       x = "Segmentation Accuracy", y = "Density", fill = "Split") +
  nf.theme.single
ggsave(file.path(PAPER_FIG_DIR, "fig13.png"), p_fig13, width = 8, height = 6, dpi = 300)

# ---- Figure B.16 (Appendix): depth vs accuracy, full range + zoomed panel --
depth_acc_valid <- lcc_true %>% filter(!is.na(Depth_WaterSurface), !is.na(accuracy))

ZOOM_DEPTH_RANGE <- c(-7.5, 0)
ZOOM_ACC_RANGE <- c(0.75, 1.0)

depth_acc_zoom_rows <- depth_acc_valid %>%
  filter(Depth_WaterSurface >= ZOOM_DEPTH_RANGE[1], Depth_WaterSurface <= ZOOM_DEPTH_RANGE[2],
         accuracy >= ZOOM_ACC_RANGE[1], accuracy <= ZOOM_ACC_RANGE[2])
pct_in_zoom <- 100 * nrow(depth_acc_zoom_rows) / nrow(depth_acc_valid)
cat(sprintf("\nFig B.16 zoom panel: %.1f%% of test/train images (n=%d of %d) fall within depth %.1f-%.1fm and accuracy %.0f-%.0f%%\n",
            pct_in_zoom, nrow(depth_acc_zoom_rows), nrow(depth_acc_valid),
            ZOOM_DEPTH_RANGE[2], ZOOM_DEPTH_RANGE[1], 100*ZOOM_ACC_RANGE[1], 100*ZOOM_ACC_RANGE[2]))

depth_acc_panel_full <- depth_acc_valid %>% mutate(panel = "Full Range")
depth_acc_panel_zoom <- depth_acc_zoom_rows %>% mutate(panel = "Zoomed")
depth_acc_combined <- bind_rows(depth_acc_panel_full, depth_acc_panel_zoom) %>%
  mutate(panel = factor(panel, levels = c("Full Range", "Zoomed")))

p_figb16 <- ggplot(depth_acc_combined, aes(x = Depth_WaterSurface, y = accuracy, color = split)) +
  geom_point(alpha = 0.75, size = 1.125, shape = 16) +
  geom_smooth(data = depth_acc_combined %>% filter(panel == "Zoomed"),
              method = "lm", formula = y ~ x, se = FALSE, linewidth = 1) +
  scale_color_manual(values = PAL2) +
  scale_y_continuous(labels = scales::percent) +
  facet_wrap(~panel, scales = "free") +
  labs(title = "Pixel-Level Segmentation Accuracy vs. Water Depth",
       x = "Depth (m)", y = "Segmentation Accuracy", color = "Split") +
  nf.theme.facet
ggsave(file.path(PAPER_FIG_DIR, "figb16.png"), p_figb16, width = 13, height = 7.5, dpi = 300)

cat("\nWrote current-vintage Figs 2/3/10/11/12/13/14(appendix) to", PAPER_FIG_DIR, "\n")

# ---- Table 5: bias decomposition by genus x bleach status, train & test ---
genus_bleach <- tax_all %>%
  filter(str_detect(taxonomy, ":")) %>%
  separate(taxonomy, into = c("genus", "status"), sep = ":") %>%
  mutate(bias = pred_frac - true_frac)

table5 <- genus_bleach %>%
  group_by(split, genus, status) %>%
  summarise(mean_bias = mean(bias, na.rm = TRUE), median_bias = median(bias, na.rm = TRUE),
            se_bias = sd(bias, na.rm = TRUE) / sqrt(n()), .groups = "drop")

genus_totals <- genus_bleach %>%
  group_by(split, genus, image_id) %>%
  summarise(bias = sum(bias), .groups = "drop") %>%
  group_by(split, genus) %>%
  summarise(mean_bias = mean(bias, na.rm = TRUE), median_bias = median(bias, na.rm = TRUE),
            se_bias = sd(bias, na.rm = TRUE) / sqrt(n()), .groups = "drop")

marginal_sums <- genus_bleach %>%
  group_by(split, status, image_id) %>%
  summarise(bias = sum(bias), .groups = "drop") %>%
  group_by(split, status) %>%
  summarise(mean_bias = mean(bias, na.rm = TRUE), se_bias = sd(bias, na.rm = TRUE) / sqrt(n()), .groups = "drop")

overall_lcc_bias <- lcc_true %>%
  mutate(bias = pred_frac - true_frac) %>%
  group_by(split) %>%
  summarise(mean_bias = mean(bias, na.rm = TRUE), se_bias = sd(bias, na.rm = TRUE) / sqrt(n()))

cat("\n=== TABLE 5 SOURCE DATA (current Sep-2025 vintage) ===\n")
cat("\nPer (genus, bleach-status) bias:\n"); print(table5, n = Inf)
cat("\nPer-genus totals (bleached+healthy):\n"); print(genus_totals, n = Inf)
cat("\nMarginal sums (per bleach status, all genera):\n"); print(marginal_sums, n = Inf)
cat("\nOverall LCC bias (train/test):\n"); print(overall_lcc_bias, n = Inf)

# ---- Shared helper: pooled TP/FP/FN ratios + image-resampling bootstrap ---
# A ratio of pooled counts (Cityscapes/PASCAL-VOC-style mIoU) is a nonlinear
# function of the summed per-image TP/FP/FN, so SD/sqrt(n) -- valid only for
# the sampling variability of a MEAN of per-image ratios -- does not apply
# to it. Instead we resample images (the actual iid sampling unit) with
# replacement, recompute the pooled ratios from each resampled set of
# per-image counts, and use the SD of those bootstrap replicates as the
# standard error. Used below for Table 6 (per-taxonomy), Table 7
# (NeuralReefer vs. CoralSCOP), and Table A.8 (spatial-robustness).
pixel_pooled_stats <- function(df) {
  TP <- sum(df$tp_px); FP <- sum(df$fp_px); FN <- sum(df$fn_px)
  tibble(
    iou       = ifelse((TP + FP + FN) > 0, TP / (TP + FP + FN),         NA_real_),
    dice      = ifelse((TP + FP + FN) > 0, 2 * TP / (2 * TP + FP + FN), NA_real_),
    precision = ifelse((TP + FP)      > 0, TP / (TP + FP),             NA_real_),
    recall    = ifelse((TP + FN)      > 0, TP / (TP + FN),             NA_real_)
  )
}

# Same as pixel_pooled_stats, plus pooled pixel accuracy (1 - (FP+FN) /
# total pixels), for tables that also report per-image "accuracy" (a
# per-image column already defined elsewhere as 1 - (fp_px+fn_px)/IMG_PX);
# pooling it the same way sums FP+FN across images against the images'
# combined pixel budget (n_images * IMG_PX) rather than averaging per-image
# accuracies directly.
pixel_pooled_stats_acc <- function(df) {
  bind_cols(pixel_pooled_stats(df),
            tibble(accuracy = 1 - sum(df$fp_px + df$fn_px) / (nrow(df) * IMG_PX)))
}

# stat_fn(df) -> a one-row tibble of pooled point statistics for that set of
# per-image rows (pixel_pooled_stats or pixel_pooled_stats_acc above).
# Returns the point estimate plus a bootstrap SE (`se_<stat>`) for each.
image_bootstrap <- function(df, stat_fn, n_boot = 2000, seed = 42) {
  set.seed(seed)
  n <- nrow(df)
  point <- stat_fn(df)
  boot <- map_dfr(seq_len(n_boot), function(b) stat_fn(df[sample.int(n, n, replace = TRUE), , drop = FALSE]))
  se <- boot %>% summarise(across(everything(), ~ sd(.x, na.rm = TRUE)))
  names(se) <- paste0("se_", names(se))
  bind_cols(tibble(n_boot_images = n), point, se)
}

# ---- Per-taxonomy IoU/Dice/Precision/Recall table (test set) ---------------
# Pooled/micro-averaged convention (as in Cityscapes/PASCAL-VOC-style mIoU):
# TP/FP/FN pixel counts are summed across all 111 test images FIRST, and a
# single IoU/Dice/Precision/Recall is computed from those totals -- rather
# than averaging a separate per-image ratio (the macro-average this block
# used previously). This handles class absence automatically with no
# special-casing: an image with no ground-truth instance of the class
# contributes 0 to TP and FN, and if the model correctly predicts nothing
# there it contributes 0 to FP too, so the image simply drops out of the
# sums; if the model hallucinates the class, that pixel count still lands
# in FP and depresses pooled precision, as it should. No single image's
# zero denominator can make any of the four ratios undefined -- only the
# pooled totals' denominators matter, and those are zero only if the class
# never appears anywhere in ground truth AND is never predicted anywhere.
#
# n is reported separately, as the number of test images whose GROUND TRUTH
# actually contains the class (tp_px + fn_px > 0) -- i.e. how many images
# this row's performance is really "about" -- while the pooled ratios AND
# the bootstrap resampling both use the full 111-image test set (an image
# without the class still correctly contributes zero counts either way).
taxonomy_test <- tax_test %>% rename(image_id = image) %>% filter(!str_detect(taxonomy, ":"))

taxonomy_test_n <- taxonomy_test %>%
  group_by(taxonomy) %>%
  summarise(n = sum((tp_px + fn_px) > 0), .groups = "drop")

iou_dice_test <- taxonomy_test %>%
  group_by(taxonomy) %>%
  group_modify(~ image_bootstrap(.x, pixel_pooled_stats)) %>%
  ungroup() %>%
  left_join(taxonomy_test_n, by = "taxonomy") %>%
  relocate(n, .after = taxonomy)
cat("\n=== Per-taxonomy pixel IoU/Dice/Precision/Recall (pooled TP/FP/FN across all 111 test images,",
    "bootstrap SE over 2000 image resamples; n = images whose ground truth contains the class) ===\n")
print(iou_dice_test, n = Inf)

# ---- NeuralReefer vs. CoralSCOP: LCC IoU/Dice/Precision/Recall table -------
coralscop <- read_csv("data/performance/coralscop_metrics.csv", show_col_types = FALSE) %>%
  filter(taxonomy == "lcc") %>%
  mutate(split = recode(split, train = "In-Sample", test = "Out-of-Sample"))

neuralreefer_lcc <- bind_rows(
  tax_train %>% filter(taxonomy == "lcc") %>% mutate(split = "In-Sample"),
  tax_test  %>% filter(taxonomy == "lcc") %>% mutate(split = "Out-of-Sample")
)

# "Single CNN" ablation (submodel 4 alone, no ensemble) -- same per-image
# schema as tax_train/tax_test, so it slots into the pooled + bootstrap
# convention below the same way the other two methods do.
single_cnn_lcc <- bind_rows(
  read_csv("data/performance/inference/annotations_coco_ablation_submodel4_train_metrics.csv", show_col_types = FALSE) %>%
    filter(taxonomy == "lcc") %>% mutate(split = "In-Sample"),
  read_csv("data/performance/inference/annotations_coco_ablation_submodel4_test_metrics.csv", show_col_types = FALSE) %>%
    filter(taxonomy == "lcc") %>% mutate(split = "Out-of-Sample")
)

comparison_all <- bind_rows(
  neuralreefer_lcc %>% mutate(method = "NeuralReefer"),
  single_cnn_lcc   %>% mutate(method = "Single CNN"),
  coralscop        %>% mutate(method = "CoralSCOP")
)

# Same pooled + bootstrap convention as iou_dice_test above (kept consistent
# with Table 6's own "Live Coral Cover" row, which reports this identical
# NeuralReefer-out-of-sample quantity -- a per-image macro-average here
# would silently disagree with that row's pooled number). n is the number
# of images in that (method, split) group whose ground truth contains any
# coral at all; pooling and the bootstrap resample use the full group.
comparison_n <- comparison_all %>%
  group_by(method, split) %>%
  summarise(n = sum((tp_px + fn_px) > 0), .groups = "drop")

comparison_table <- comparison_all %>%
  group_by(method, split) %>%
  group_modify(~ image_bootstrap(.x, pixel_pooled_stats)) %>%
  ungroup() %>%
  left_join(comparison_n, by = c("method", "split")) %>%
  relocate(n, .after = split)

cat("\n=== NeuralReefer vs. CoralSCOP: LCC IoU/Dice/Precision/Recall (pooled TP/FP/FN,",
    "bootstrap SE over 2000 image resamples; n = images with any ground-truth coral) ===\n")
print(comparison_table, n = Inf)

# ---- Figure 15: IoU exceedance curve, NeuralReefer vs. CoralSCOP, test set -
# P(IoU >= x) for x in [0,1] -- "what fraction of test images does each
# method score at least this well on?" (reviewer comment 9b's quantile ask).
# Per-image IoU here is intentionally recomputed from the raw tp_px/fp_px/
# fn_px counts (with the same "undefined -> NA, not 1.0" zero-denominator
# rule as pixel_pooled_stats/train.py's pixel_confusion_metrics) rather than
# read off the CSV's own `iou` column, so this figure uses the same IoU
# definition as Tables 6-8 instead of silently trusting a column that was
# computed by a since-changed convention (train.py's old "both masks empty
# -> 1.0" default). This is a no-op numerically for LCC specifically -- no
# test image has zero ground-truth-and-predicted coral simultaneously -- but
# keeps the figure correct by construction rather than by coincidence.
per_image_iou <- function(df) {
  total_px <- df$tp_px + df$fp_px + df$fn_px
  ifelse(total_px == 0, NA_real_, df$tp_px / total_px)
}

exceedance_curve <- function(iou_vals, method_name) {
  xs <- seq(0, 1, by = 0.01)
  tibble(method = method_name, threshold = xs,
         pct_exceeding = sapply(xs, function(x) mean(iou_vals >= x, na.rm = TRUE)))
}

nr_test_iou <- per_image_iou(tax_test %>% filter(taxonomy == "lcc"))
cs_test_iou <- per_image_iou(coralscop %>% filter(taxonomy == "lcc", split == "Out-of-Sample"))

exceedance_df <- bind_rows(
  exceedance_curve(nr_test_iou, "Present model"),
  exceedance_curve(cs_test_iou, "CoralSCOP")
)

p_fig15 <- ggplot(exceedance_df, aes(x = threshold, y = pct_exceeding, color = method)) +
  geom_line(linewidth = 1.2) +
  scale_x_continuous(labels = scales::percent) +
  scale_y_continuous(labels = scales::percent) +
  scale_color_manual(values = c("Present model" = PAL2[1], "CoralSCOP" = PAL2[2])) +
  labs(title = "Live Coral Cover IoU Exceedance, Test Set",
       x = "IoU Threshold", y = "% of Test Images At or Above Threshold", color = NULL) +
  nf.theme.single
ggsave(file.path(PAPER_FIG_DIR, "fig15.png"), p_fig15, width = 8, height = 6, dpi = 300)
cat("\nWrote", file.path(PAPER_FIG_DIR, "fig15.png"), "\n")

cat(sprintf("\nExceedance thresholds (prose in Section~5.4): NeuralReefer >= 0.5: %.1f%%, CoralSCOP >= 0.5: %.1f%%, NeuralReefer >= 0.7: %.1f%%, CoralSCOP >= 0.7: %.1f%%\n",
            100 * mean(nr_test_iou >= 0.5, na.rm = TRUE), 100 * mean(cs_test_iou >= 0.5, na.rm = TRUE),
            100 * mean(nr_test_iou >= 0.7, na.rm = TRUE), 100 * mean(cs_test_iou >= 0.7, na.rm = TRUE)))

# ---- Conditional IoU: mean IoU by true-LCC quartile, test set --------------
# Reviewer item 9c: low-LCC images look bad on IoU even when the absolute
# prediction error is tiny (e.g. predicting 4% when the truth is 6% coral
# cover already implies IoU=0.5) -- bucket by true LCC quartile to show this
# directly, rather than only citing the aggregate IoU number.
lcc_test <- lcc_true %>% filter(split == "Test")
lcc_test$lcc_quartile <- ntile(lcc_test$true_frac, 4)

conditional_iou <- lcc_test %>%
  group_by(lcc_quartile) %>%
  summarise(n = n(), mean_true_lcc = mean(true_frac), mean_iou = mean(iou),
            mean_abs_error = mean(abs(pred_frac - true_frac)), .groups = "drop")

cat("\n=== Conditional IoU by true-LCC quartile (test set) ===\n")
print(conditional_iou, n = Inf)
################################################################################

# ---- Spatial-dependence robustness check (current data vintage, Appendix) ---
# Nearby/overlapping train and test photos have correlated coral cover; a test
# image within a small radius of a train image violates the train/test
# independence assumption. Checked at 2m and 5m via the same fixed-radius
# kd-tree search (dbscan::frNN) used above, querying the train-image kd-tree
# with the test-image coordinates so each test image's within-radius train
# neighbors can be read off directly (no O(n_test x n_train) distance matrix).
# lcc_test already carries NorthPhoto_UTM/EastPhoto_UTM from tax_all's earlier
# join against split_meta -- no need to rejoin (a second join on the same
# column names would only suffix them as .x/.y).
all_meta <- read_csv("data/metadata/train_test_split_metadata.csv", show_col_types = FALSE) %>%
  select(filename, split, NorthPhoto_UTM, EastPhoto_UTM) %>%
  distinct(filename, .keep_all = TRUE)

lcc_test_geo <- lcc_test

train_coords <- all_meta %>% filter(split == "train", !is.na(NorthPhoto_UTM), !is.na(EastPhoto_UTM))
test_coords <- lcc_test_geo %>% filter(!is.na(NorthPhoto_UTM), !is.na(EastPhoto_UTM))
n_missing_coords <- nrow(lcc_test_geo) - nrow(test_coords)

train_mat <- as.matrix(train_coords[, c("NorthPhoto_UTM", "EastPhoto_UTM")])
test_mat  <- as.matrix(test_coords[, c("NorthPhoto_UTM", "EastPhoto_UTM")])

nn_2m <- dbscan::frNN(x = train_mat, eps = 2, query = test_mat)
nn_5m <- dbscan::frNN(x = train_mat, eps = 5, query = test_mat)

contam_2m <- test_coords$image_id[lengths(nn_2m$id) > 0]
contam_5m <- test_coords$image_id[lengths(nn_5m$id) > 0]

cat("\n=== Spatial-dependence robustness check ===\n")
cat(n_missing_coords, "test image(s) lack usable GPS/UTM coordinates and could not be checked.\n")
cat(length(contam_2m), "of", nrow(test_coords), "geolocated test images are within 2m of a train image.\n")
cat(length(contam_5m), "of", nrow(test_coords), "geolocated test images are within 5m of a train image.\n")

# Same pooled + bootstrap convention as iou_dice_test/comparison_table above
# (this is the same NeuralReefer-out-of-sample LCC quantity reported there,
# just re-computed on the 2m/5m-excluded subsets); n_gt is the number of
# images in each subset whose ground truth contains any coral, reported
# alongside the subset's total image count.
spatial_subsets <- list(
  "All test images" = lcc_test_geo,
  "2m"              = lcc_test_geo %>% filter(!image_id %in% contam_2m),
  "5m"              = lcc_test_geo %>% filter(!image_id %in% contam_5m)
)

spatial_summary <- map_dfr(names(spatial_subsets), function(subset_name) {
  df <- spatial_subsets[[subset_name]]
  bind_cols(tibble(subset = subset_name, n_total = nrow(df), n_gt = sum((df$tp_px + df$fn_px) > 0)),
            image_bootstrap(df, pixel_pooled_stats_acc) %>% select(-n_boot_images))
})
cat("\nLCC IoU/Dice/Accuracy (pooled TP/FP/FN, bootstrap SE over 2000 image resamples),",
    "with vs. without spatially-contaminated test images excluded",
    "(n_total = images in subset, n_gt = of those, images with any ground-truth coral):\n")
print(spatial_summary, n = Inf)
################################################################################
