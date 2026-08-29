library(tidyverse)
library(readxl)
library(dbscan)
library(igraph)

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

# Report the effect of removing the independence violators, now including
# the derived IoU/Dice/F1 metrics alongside the existing accuracy/bias ones.
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
