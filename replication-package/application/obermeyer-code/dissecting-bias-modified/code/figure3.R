################################################################################
# prep
################################################################################
# clear
rm(list=ls())
# load packages
library(plotly)
library(scales)
library(plot0)
# set output path
res_dir <- paste0(getwd(), '/results/')

################################################################################
# figure 3a 
################################################################################
# load -------------------------------------------------------------------------
filepath <- paste0(getwd(), '/data')
filename <- '/data_new.csv'
dt <- MyFread(filename, filepath)

# compute ----------------------------------------------------------------------
df <- MyComputePlotDF(dt, 
                col.to.y = 'cost_t',
                col.to.cut = 'risk_score_t',
                col.to.groupby = 'race',
                nquantiles = 10,
                ci.level = 0.95)
df[, quantile:= quantile - 5]
df <- unique(df, by = c('race', 'percentile', 'quantile'))

df[, lci := log(col_to_mean_by_quantile_by_race - 1.96 * ci_se)]
df[, uci := log(col_to_mean_by_quantile_by_race + 1.96 * ci_se)]
df[, col_to_mean_by_percentile_by_race := log(col_to_mean_by_percentile_by_race + 0.001)]
df[, col_to_mean_by_quantile_by_race := log(col_to_mean_by_quantile_by_race + 0.001)]
df[, ci_se := log(ci_se + 0.001)]

# labels -----------------------------------------------------------------------
titlename <- ''
subtitlename <- ''
groupbycolorname <- 'Race'
xname <- 'Percentile of Algorithm Risk Score'
yname <- 'Mean Total Medical Expenditure'
color_scheme <- c("#764885","#ffa600") 
linetype_scheme <- c('twodash', 'solid') 

default_in_percentile <- 97
additional_screen_percentile <- 55
vlocation_threshold <- log(50000)

brkk <-  log(c(1000, 3000, 8000, 20000, 60000))
labb <- comma_format()(exp(brkk))

# plot -----------------------------------------------------------------------
ga <- ggplot(data = df, aes(color = race, linetype = race,
                           group = race)) +
  theme_bw() +
  labs(title = titlename, 
       subtitle = subtitlename,
       color = groupbycolorname,
       x = xname,
       y = yname) +
   scale_x_continuous(breaks = seq(0, 100, 10)) +
   scale_y_continuous(breaks = brkk, labels = labb, limits = c(7,11))+ 
   scale_color_manual(values = color_scheme, name = groupbycolorname) +
   scale_linetype_manual(values = linetype_scheme, name = groupbycolorname) +
   theme(legend.position="bottom") + 
   theme(legend.key.size = grid::unit(5,"lines")) + 
   theme(legend.key.height= grid::unit(1,"lines")) + 
   geom_point(aes(x = percentile, y = col_to_mean_by_percentile_by_race), shape = 4) +
   geom_point(aes(x = quantile, y = col_to_mean_by_quantile_by_race), size = 2) +
   geom_smooth(aes(x = percentile, y = col_to_mean_by_percentile_by_race), se = F, span = 0.45) +
   geom_pointrange(aes(x = quantile, y = col_to_mean_by_quantile_by_race, 
                       ymin = lci,
                       ymax = uci)) + 
   geom_vline(aes(xintercept=default_in_percentile), colour="black", linetype="dashed") +
   geom_text(aes(x=default_in_percentile, label="Defaulted into program", y = vlocation_threshold), colour="black", hjust = 1.2, size = 2) +
   geom_vline(aes(xintercept=additional_screen_percentile), colour="dark gray", linetype="dashed") +
   geom_text(aes(x=additional_screen_percentile, label="Referred for screen", y = vlocation_threshold), colour="dark gray", hjust = 1.2, size = 2) 

################################################################################
# figure 3b 
################################################################################
# compute ----------------------------------------------------------------------
subset.cols <- c('cost_t', 'gagne_sum_t', 'race')
DF <- copy(dt[, ..subset.cols])
DF[, race := as.factor(race)]
DF[, g_ranks :=  rank(gagne_sum_t, ties.method = 'first')]
DF[, percentile := cut(g_ranks, quantile(g_ranks,probs=0:100/100), include.lowest = TRUE, labels = FALSE)]
#sort(unique(DF$percentile))
DF[, quantile:= cut(g_ranks, quantile(g_ranks,probs=0:10/10), include.lowest = TRUE, labels = FALSE)]
#sort(unique(DF$quantile))
MyComputeMean(DF, 'cost_t', c('percentile', 'race'))
MyComputeMean(DF, 'cost_t', c('quantile', 'race'))
MyComputeSE(DF, 'cost_t', c('quantile', 'race'))

DF <- unique(DF, by = c('race', 'percentile', 'quantile'))
DF[, quantile:= quantile* 10 -5]

DF[, lci := log(col_to_mean_by_quantile_by_race - 1.96 * ci_se)]
DF[, uci := log(col_to_mean_by_quantile_by_race + 1.96 * ci_se)]
DF[, col_to_mean_by_percentile_by_race := log(col_to_mean_by_percentile_by_race + 0.001)]
DF[, col_to_mean_by_quantile_by_race := log(col_to_mean_by_quantile_by_race + 0.001)]
DF[, ci_se := log(ci_se + 0.001)]

# labels -----------------------------------------------------------------------
titlename <- ''
subtitlename <- ''
groupbycolorname <- 'Race'
xname <- 'Chronic Illnesses'
yname <- 'Mean Total Medical Expenditure'

brkk <-  log(c(1000, 3000, 8000, 20000, 60000))
labb <- comma_format()(exp(brkk))

# plot -----------------------------------------------------------------------
gb <- ggplot(data = DF, aes(color = race, group = race, linetype = race))  +
  theme_bw() +
  labs(title = titlename, 
       subtitle = subtitlename,
       color = groupbycolorname,
       x = xname,
       y = yname) +
   scale_x_continuous(breaks = seq(0, 100, 10)) +
   scale_y_continuous(breaks = brkk, labels = labb, limits = c(7,11))+ 
   scale_color_manual(values = color_scheme, name = groupbycolorname) +
   scale_linetype_manual(values = linetype_scheme, name = groupbycolorname) +
   geom_point(aes(x = percentile, y = col_to_mean_by_percentile_by_race), shape = 4) +
   geom_point(aes(x = quantile, y = col_to_mean_by_quantile_by_race), size = 2) +
   geom_smooth(aes(x = percentile, y = col_to_mean_by_percentile_by_race), se = F, span = 0.55)+
   geom_pointrange(aes(x = quantile, y = col_to_mean_by_quantile_by_race, 
                       ymin = lci,
                       ymax = uci)) 

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

fig3 <- grid.arrange(arrangeGrob(ga + theme(legend.position="none"), 
                        gb + theme(legend.position="none"), nrow = 1, respect=TRUE),
                        commonlegend, nrow = 2, heights = c(7,1))

# png
ggsave(paste0(res_dir, 'figure3.png'), device = 'png', fig3, width = 14, height = 7)
# eps
ggsave(paste0(res_dir, 'figure3.eps'), device = 'eps', fig3, width = 14, height = 7)
