################################################################################
# prep
################################################################################
rm(list=ls()) 
# load packages
library(plot0) 
# set output path
res_dir <- paste0(getwd(), '/results/')

################################################################################
# figure 2
################################################################################
# load -------------------------------------------------------------------------
filepath <- paste0(getwd(), '/data')
filename <- 'data_new.csv'
dt_raw <- MyFread(filename, filepath)

default_in_percentile <- 97
additional_screen_percentile <- 55

# plotting parameters
color_scheme <- c("#764885","#ffa600") 
linetype_scheme <- c('twodash', 'solid') 
subtitlename <- ''
groupbycolorname <- 'Race'
xname <- 'Percentile of Algorithm Risk Score'

################################################################################
# bps 
################################################################################
# computt ----------------------------------------------------------------------
dt_raw[, bps_above_139_ind := ifelse(bps_mean_t > 139, 1, 0)]
dt_raw$bps_above_139_ind[is.na(dt_raw$bps_above_139_ind)] <- 0

dt_bps <- MyComputePlotDF(dt_raw, 
                col.to.y = 'bps_above_139_ind',
                col.to.cut = 'risk_score_t',
                col.to.groupby = 'race',
                nquantiles = 5,
                ci.level = 0.95)

dt_bps <- unique(dt_bps, by = c('race', 'percentile', 'quantile'))
dt_bps[, quantile := quantile - 10]

# labels -----------------------------------------------------------------------
titlename <- '(a) Hypertension: Fraction clinic visits with SBP >139 mmHg'
yname <- 'Fraction with uncontrolled blood pressure'
avlocation_threshold <- 0.4

# plot -------------------------------------------------------------------------
ga <- ggplot(data = dt_bps, aes(color = race, linetype = race,
                           group = race)) +
  theme(plot.title = element_text(hjust = 0.5)) + 
  theme_bw() +
  labs(title = titlename, 
       subtitle = subtitlename,
       color = groupbycolorname,
       x = xname,
       y = yname) +
   scale_x_continuous(breaks = seq(0, 100, 10)) +
   scale_color_manual(values = color_scheme, name = groupbycolorname) +
   scale_linetype_manual(values = linetype_scheme, name = groupbycolorname) +
   theme(legend.key.size = grid::unit(5,"lines")) + 
   theme(legend.key.height= grid::unit(2,"lines")) + 
   geom_point(aes(x = percentile, y = col_to_mean_by_percentile_by_race), alpha = 0.4, shape = 4) +
   geom_point(aes(x = quantile, y = col_to_mean_by_quantile_by_race), size = 2) +
   geom_smooth(aes(x = percentile, y = col_to_mean_by_percentile_by_race), se = TRUE, span = 0.99) +
   geom_pointrange(aes(x = quantile, y = col_to_mean_by_quantile_by_race, 
                       ymin = col_to_mean_by_quantile_by_race - 1.96 * ci_se,
                       ymax = col_to_mean_by_quantile_by_race + 1.96 * ci_se)) + 
   geom_vline(aes(xintercept=default_in_percentile), colour="black", linetype="dashed") +
   geom_text(aes(x=default_in_percentile, label="Defaulted into program", y = avlocation_threshold), colour="black", hjust = 1.2, size = 2) +
   geom_vline(aes(xintercept=additional_screen_percentile), colour="dark gray", linetype="dashed") +
   geom_text(aes(x=additional_screen_percentile, label="Referred for screen", y = avlocation_threshold), colour="dark gray", hjust = 1.2, size = 2) 

################################################################################
# Hba1c
################################################################################
# compute ----------------------------------------------------------------------
dt_hba1c <- MyComputePlotDF(dt_raw, 
                col.to.y = 'ghba1c_mean_t',
                col.to.cut = 'risk_score_t',
                col.to.groupby = 'race',
                nquantiles = 5,
                ci.level = 0.95)
dt_hba1c <- unique(dt_hba1c, by = c('race', 'percentile', 'quantile'))
dt_hba1c[, quantile := quantile - 10]

# labels -----------------------------------------------------------------------
titlename <- '(b) Diabetes severity: HbA1c'
yname <- 'Mean HbA1c (%)'
bvlocation_threshold <- quantile(dt_hba1c$col_to_mean_by_percentile_by_race, 0.99, na.rm = T)
bvlocation_threshold <- 7.5

# plot -------------------------------------------------------------------------
gb <- ggplot(data = dt_hba1c, aes(color = race, linetype = race,
                           group = race)) +
  theme(plot.title = element_text(hjust = 0.5)) + 
  theme_bw() +
  labs(title = titlename, 
       subtitle = subtitlename,
       color = groupbycolorname,
       x = xname,
       y = yname) +
   scale_x_continuous(breaks = seq(0, 100, 10)) +
   scale_color_manual(values = color_scheme, name = groupbycolorname) +
   scale_linetype_manual(values = linetype_scheme, name = groupbycolorname) +
   geom_point(aes(x = percentile, y = col_to_mean_by_percentile_by_race), alpha = 0.4, shape = 4) +
   geom_point(aes(x = quantile, y = col_to_mean_by_quantile_by_race), size = 2) +
   geom_smooth(aes(x = percentile, y = col_to_mean_by_percentile_by_race), se = TRUE, span = 0.99) +
   geom_pointrange(aes(x = quantile, y = col_to_mean_by_quantile_by_race, 
                       ymin = col_to_mean_by_quantile_by_race - 1.96 * ci_se,
                       ymax = col_to_mean_by_quantile_by_race + 1.96 * ci_se)) + 
   geom_vline(aes(xintercept=default_in_percentile), colour="black", linetype="dashed") +
   geom_text(aes(x=default_in_percentile, label="Defaulted into program", y = bvlocation_threshold), colour="black", hjust = 1.2, size = 2) +
   geom_vline(aes(xintercept=additional_screen_percentile), colour="dark gray", linetype="dashed") +
   geom_text(aes(x=additional_screen_percentile, label="Referred for screen", y = bvlocation_threshold), colour="dark gray", hjust = 1.2, size = 2) 

################################################################################
# Hematocrit
################################################################################
# compute ----------------------------------------------------------------------
dt_hemo <- MyComputePlotDF(dt_raw, 
                col.to.y = 'hct_mean_t',
                col.to.cut = 'risk_score_t',
                col.to.groupby = 'race',
                nquantiles = 5,
                ci.level = 0.95)
dt_hemo <- unique(dt_hemo, by = c('race', 'percentile', 'quantile'))

# labels -----------------------------------------------------------------------
titlename <- '(e) Anemia severity: hematocrit'
dt_hemo[, quantile := quantile - 10]
yname <- 'Mean Hematocrit (%)'
cvlocation_threshold <- 45 

# plot ---------------------------------------------------------------------------
ge <- ggplot(data = dt_hemo, aes(color = race, linetype = race,
                           group = race)) +
  theme(plot.title = element_text(hjust = 0.5)) + 
  theme_bw() +
  labs(title = titlename, 
       subtitle = subtitlename,
       color = groupbycolorname,
       x = xname,
       y = yname) +
   scale_x_continuous(breaks = seq(0, 100, 10)) +
   scale_color_manual(values = color_scheme, name = groupbycolorname) +
   scale_linetype_manual(values = linetype_scheme, name = groupbycolorname) +
   geom_point(aes(x = percentile, y = col_to_mean_by_percentile_by_race), alpha = 0.4, shape = 4) +
   geom_point(aes(x = quantile, y = col_to_mean_by_quantile_by_race), size = 2) +
   geom_smooth(aes(x = percentile, y = col_to_mean_by_percentile_by_race), se = TRUE, span = 0.99) +
   geom_pointrange(aes(x = quantile, y = col_to_mean_by_quantile_by_race, 
                       ymin = col_to_mean_by_quantile_by_race - 1.96 * ci_se,
                       ymax = col_to_mean_by_quantile_by_race + 1.96 * ci_se)) + 
   geom_vline(aes(xintercept=default_in_percentile), colour="black", linetype="dashed") +
   geom_text(aes(x=default_in_percentile, label="Defaulted into program", y = cvlocation_threshold), colour="black", hjust = 1.2, size = 2) +
   geom_vline(aes(xintercept=additional_screen_percentile), colour="dark gray", linetype="dashed") +
   geom_text(aes(x=additional_screen_percentile, label="Referred for screen", y = cvlocation_threshold), colour="dark gray", hjust = 1.2, size = 2) 

################################################################################
# Creatinine
################################################################################
# compute ----------------------------------------------------------------------
dt_raw[, cre_mean_mgdL_log10 := log10(cre_mean_t)]
dt_raw$cre_mean_mgdL_log10[is.na(dt_raw$re_mean_mgdL_log10)] <- 0
dt_crea <- MyComputePlotDF(dt_raw, 
                col.to.y = 'cre_mean_mgdL_log10',
                col.to.cut = 'risk_score_t',
                col.to.groupby = 'race',
                nquantiles = 5,
                ci.level = 0.95)
dt_crea[, quantile := quantile - 10]
dt_crea <- unique(dt_crea, by = c('race', 'percentile', 'quantile'))

# labels -----------------------------------------------------------------------
titlename <- '(d) Renal failure: creatinine (log)'
yname <- 'Mean creatinine (log mg/dL)'
dvlocation_threshold <- 0.2

# plot ------------------------------------------------------------------------
gd <- ggplot(data = dt_crea, aes(color = race, linetype = race,
                           group = race)) +
  theme(plot.title = element_text(hjust = 0.5)) + 
  theme_bw() +
  labs(title = titlename, 
       subtitle = subtitlename,
       color = groupbycolorname,
       x = xname,
       y = yname) +
   scale_x_continuous(breaks = seq(0, 100, 10)) +
   coord_cartesian(ylim = c(-0.1, 0.2)) + 
   scale_color_manual(values = color_scheme, name = groupbycolorname) +
   scale_linetype_manual(values = linetype_scheme, name = groupbycolorname) +
   geom_point(aes(x = percentile, y = col_to_mean_by_percentile_by_race), alpha = 0.4, shape = 4) +
   geom_point(aes(x = quantile, y = col_to_mean_by_quantile_by_race), size = 2) +
   geom_smooth(aes(x = percentile, y = col_to_mean_by_percentile_by_race), se = TRUE, span = 0.99) +
   geom_pointrange(aes(x = quantile, y = col_to_mean_by_quantile_by_race, 
                       ymin = col_to_mean_by_quantile_by_race - 1.96 * ci_se,
                       ymax = col_to_mean_by_quantile_by_race + 1.96 * ci_se)) + 
   geom_vline(aes(xintercept=default_in_percentile), colour="black", linetype="dashed") +
   geom_text(aes(x=default_in_percentile, label="Defaulted into program", y = dvlocation_threshold), colour="black", hjust = 1.2, size = 2) +
   geom_vline(aes(xintercept=additional_screen_percentile), colour="dark gray", linetype="dashed") +
   geom_text(aes(x=additional_screen_percentile, label="Referred for screen", y = dvlocation_threshold), colour="dark gray", hjust = 1.2, size = 2) 

################################################################################
# LDL
################################################################################
# compute ----------------------------------------------------------------------
dt_ldl <- MyComputePlotDF(dt_raw, 
                col.to.y = 'ldl_mean_t',
                col.to.cut = 'risk_score_t',
                col.to.groupby = 'race',
                nquantiles = 5,
                ci.level = 0.95)
dt_ldl[, percentile := NULL]
dt_ldl[, col_to_mean_by_percentile_by_race:= NULL]
setnames(dt_ldl, 'quantile', 'quartiles')

# add ventiles
MyComputeQuantile(dt_ldl, 20, 'risk_score_t', NULL)
setnames(dt_ldl, 'quantile', 'ventile')

MyComputeMean(dt_ldl, 'ldl_mean_t', c('ventile', 'race'))
setnames(dt_ldl, 'quartiles', 'quantile')
dt_ldl[, quantile := quantile - 10]
dt_ldl <- unique(dt_ldl, by = c('race', 'ventile', 'quantile'))

# labels -----------------------------------------------------------------------
titlename <- '(c) Bad cholesterol: LDL '
yname <- 'Mean LDL (mg/dL)'
evlocation_threshold <- 115

# plot -------------------------------------------------------------------------
gc <- ggplot(data = dt_ldl, aes(color = race, linetype = race,
                           group = race)) +
  theme(plot.title = element_text(hjust = 0.5)) + 
  theme_bw() +
  labs(title = titlename, 
       subtitle = subtitlename,
       color = groupbycolorname,
       x = xname,
       y = yname) +
   scale_x_continuous(breaks = seq(0, 100, 10)) +
   scale_color_manual(values = color_scheme, name = groupbycolorname) +
   scale_linetype_manual(values = linetype_scheme, name = groupbycolorname) +
   geom_point(aes(x = ventile, y = col_to_mean_by_ventile_by_race), alpha = 0.4, shape = 4) +
   geom_point(aes(x = quantile, y = col_to_mean_by_quantile_by_race), size = 2) +
   geom_smooth(aes(x = ventile, y = col_to_mean_by_ventile_by_race), se = TRUE, span = 0.99) +
   geom_pointrange(aes(x = quantile, y = col_to_mean_by_quantile_by_race, 
                       ymin = col_to_mean_by_quantile_by_race - 1.96 * ci_se,
                       ymax = col_to_mean_by_quantile_by_race + 1.96 * ci_se)) + 
   geom_vline(aes(xintercept=default_in_percentile), colour="black", linetype="dashed") +
   geom_text(aes(x=default_in_percentile, label="Defaulted into program", y = evlocation_threshold), colour="black", hjust = 1.2, size = 2) +
   geom_vline(aes(xintercept=additional_screen_percentile), colour="dark gray", linetype="dashed") +
   geom_text(aes(x=additional_screen_percentile, label="Referred for screen", y = evlocation_threshold), colour="dark gray", hjust = 1.2, size = 2) 

################################################################################
# export 
################################################################################
# create a common legend 
glegend <- function(a.gplot){
    tab <- ggplot_gtable(ggplot_build(a.gplot))
    legd <- which(sapply(tab$grobs, function(x) x$name) == "guide-box")
    legend <- tab$grobs[[legd]]
    return(legend)}

commonlegend<-glegend(ga)

fig2 <- grid.arrange(ga + theme(legend.position="none"), 
                        gb + theme(legend.position="none"), 
                        gc + theme(legend.position="none"), 
                        gd + theme(legend.position="none"), 
                        ge + theme(legend.position="none"), 
                        commonlegend, ncol = 2, respect=TRUE)

# png
ggsave(paste0(res_dir, 'figure2.png'), device = 'png', fig2, width = 14, height = 28)
# eps
ggsave(paste0(res_dir, 'figure2.eps'), device = 'eps', fig2, width = 14, height = 28)


# export ----------------------------------------------------------------------
#filename <- 'figure2_a_bps'
#ggsave(ga, device = 'png', file = paste0(res_dir, filename, ".png"),
#       width = 10, height = 10, units = "in", dpi = 250)

# export -----------------------------------------------------------------------
#filename <- 'figure2_b_hba1c'
#ggsave(gb, device = 'png', file = paste0(res_dir, filename, ".png"),
#       width = 10, height = 10, units = "in", dpi = 250)

# export -----------------------------------------------------------------------
#filename <- 'figure2_c_ldl'
#ggsave(gc, device = 'png', file = paste0(res_dir, filename, ".png"),
#       width = 10, height = 10, units = "in", dpi = 250)

# export -----------------------------------------------------------------------
#filename <- 'figure2_d_crea'
#ggsave(gd, device = 'png', file = paste0(res_dir, filename, ".png"),
#       width = 10, height = 10, units = "in", dpi = 250)

# export -----------------------------------------------------------------------
#filename <- 'figure2_e_hemo'
#ggsave(ge, device = 'png', file = paste0(res_dir, filename, ".png"),
#       width = 10, height = 10, units = "in", dpi = 250)

