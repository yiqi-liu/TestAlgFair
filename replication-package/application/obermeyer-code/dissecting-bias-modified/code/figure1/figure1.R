################################################################################
# prep
################################################################################
rm(list=ls()) 
# load packages
library(plot0) 
# set output path
res_dir <- paste0(getwd(), '/results/')

################################################################################
# figure 1a 
################################################################################
# load -------------------------------------------------------------------------
filepath <- paste0(getwd(), '/data')
filename <- 'data_new.csv'
dt <- MyFread(filename, filepath)

# compute ----------------------------------------------------------------------
df <- MyComputePlotDF(dt, 
                col.to.y = 'gagne_sum_t',
                col.to.cut = 'risk_score_t',
                col.to.groupby = 'race',
                nquantiles = 10,
                ci.level = 0.95)
# labels -----------------------------------------------------------------------
titlename <- ''
subtitlename <- ''
groupbycolorname <- 'Race'
xname <- 'Percentile of Algorithm Risk Score'
yname <- 'Number of active chronic conditions'
color_scheme <- c("#764885","#ffa600") 
linetype_scheme <- c('twodash', 'solid') 
group_label = c("Black", "White")

default_in_percentile <- 97
additional_screen_percentile <- 55

# plot -----------------------------------------------------------------------
ga <- ggplot(data = df, aes(color = race, 
                            linetype = race, 
                            group = race)) +
  theme_bw() +
  labs(title = titlename, 
       subtitle = subtitlename,
       color = groupbycolorname,
       x = xname,
       y = yname) +
   scale_x_continuous(breaks = seq(0, 100, 10)) +
   scale_y_continuous(breaks = seq(0, 8, 2)) +
   scale_color_manual(values = color_scheme, labels = group_label, name = groupbycolorname) +
   scale_linetype_manual(values = linetype_scheme, labels = group_label, name = groupbycolorname) +
   theme(legend.key.size = grid::unit(5,"lines")) + 
   theme(legend.key.height= grid::unit(1,"lines")) + 
   theme(legend.position = 'bottom') + 
   theme(aspect.ratio = 1) + 
   geom_point(aes(x = percentile, y = col_to_mean_by_percentile_by_race), alpha = 0.4, shape = 4) +
   geom_point(aes(x = quantile - 5, y = col_to_mean_by_quantile_by_race), size = 2) +
   geom_smooth(aes(x = percentile, y = col_to_mean_by_percentile_by_race), method = "glm", formula = y~x, method.args=list(family = gaussian(link = 'log')), se = FALSE, span = 0.99) +
   geom_pointrange(aes(x = quantile - 5, y = col_to_mean_by_quantile_by_race, 
                       ymin = col_to_mean_by_quantile_by_race - 1.96 * ci_se,
                       ymax = col_to_mean_by_quantile_by_race + 1.96 * ci_se)) + 
   geom_vline(aes(xintercept=default_in_percentile), colour="black", linetype="dashed") +
   geom_text(aes(x=default_in_percentile, label="Defaulted into program", y = 6), colour="black", hjust = 1.2, size = 2) +
   geom_vline(aes(xintercept=additional_screen_percentile), colour="dark gray", linetype="dashed") +
   geom_text(aes(x=additional_screen_percentile, label="Referred for screen", y = 6), colour="dark gray", hjust = 1.2, size = 2) 

################################################################################
# figure 1b
################################################################################
#source(paste0(getwd(), '/code/figure1/figure1b.R'))

# load -------------------------------------------------------------------------
filepath <- paste0(getwd(), '/results')
filename <- 'figure1b.csv'
dt <- MyFread(filename, filepath)

dt_long <- gather(dt, before_or_after, frac, before:after, factor_key=TRUE)
dt_long$before_or_after <- as.factor(dt_long$before_or_after)

# labels -----------------------------------------------------------------------
titlename <- ''
subtitlename <- ''
groupbycolorname <- ''
xname <- 'Percentile of Algorithm Risk Score'
yname <- 'Fraction Black'
color_scheme1 <- c('#b54984', '#ff7547') 
linetype_scheme1 <- c('solid', 'dashed') 
group_label1 <- c("Original", "Simulated")

program_size <- 90

dt_long <- as.data.table(dt_long)
dt_long <- dt_long[percentile >= 55]
# plot ------------------------------------------------------------------------

gb <- ggplot(data = dt_long, 
             aes(x = percentile, y = frac, color = before_or_after, linetype =
                 before_or_after, group = before_or_after)) + 
   theme_bw() + 
   labs(title = titlename, subtitle = subtitlename, color = groupbycolorname, x
        = xname, y = yname) +
   scale_x_continuous(breaks = c(seq(55, 95, 5), 99)) +
   scale_y_continuous(breaks = seq(0, 1, by = 0.05)) +
   scale_color_manual(values = color_scheme1, labels = group_label1, name = groupbycolorname) +
   scale_linetype_manual(values = linetype_scheme1, labels = group_label1, name = groupbycolorname) +
   theme(legend.key.size = grid::unit(5,"lines")) + 
   theme(legend.key.height= grid::unit(1,"lines")) + 
   theme(legend.position = 'bottom') + 
   theme(aspect.ratio = 1) + 
   geom_point(shape = 4) +
   geom_smooth(span = 0.99, se = TRUE, level = 0.95) + 
   geom_vline(aes(xintercept=default_in_percentile), colour="black", linetype="dashed") +
   geom_text(aes(x=default_in_percentile, label="Defaulted into program", y =
                 0.45), colour="black", hjust = 1.2, size = 2) + 
   geom_vline(aes(xintercept=additional_screen_percentile), colour="dark gray",
              linetype="dashed") +
   geom_text(aes(x=additional_screen_percentile, label="Referred for screen", y
                 = 0.45), colour="dark gray", hjust = -0.2, size = 2) 

################################################################################
# export 
################################################################################
# png 
ggsave(paste0(res_dir, 'figure1.png'), grid.arrange(ga, gb, ncol = 2, respect=TRUE), width = 14, height = 7)
# eps 
ggsave(paste0(res_dir, 'figure1.eps'), device="eps", grid.arrange(ga, gb, ncol = 2, respect=TRUE), width = 14, height = 7)
