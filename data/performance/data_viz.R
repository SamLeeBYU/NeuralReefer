library(tidyverse)
library(readxl)

metadata = readxl::read_xlsx("data/metadata/Day3_Photo_MetaData_sr4.xlsx")

eval.dat = read_csv("data/performance/coral_segmenter_predictions.v.1.0.csv")

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
