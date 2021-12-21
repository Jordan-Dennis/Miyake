rm(list = ls())         # Clearing the global list
library("tidyverse")    # The might of the tidyverse
library("gridExtra")    # For constructing the grid

theme_set(theme_bw())   # Universally setting the theme

normalize <- function(x) {
    # This function preserves the distribution of the data
    # but moves it into the range 0 -> 1
    x <- (x - min(x)) / (max(x) - min(x))
}

colors <- list(# A list to store the color of each dataset
    sakuri = "green",   # The sakuri 20 dataset is coded green
    miyake = "blue",    # The miyake 12 dataset is coded blue
    buntgen = "red"     # The buntegen 18 dataset is coded red
)

sakuri <- read_csv("sakuri20.csv") %>%      # Reading data
    as_tibble() %>%                         # Lazy evaluations
    transmute(
        DC14 = normalize(d14c),             # Normalising d14c
        year,                               # Preserving time data
        set = rep("sakuri", length(year))   # Storing dataset for rbind
    )

sakuri_time_series <- sakuri %>%        # Constructing the time series
    ggplot(aes(x = year, y = DC14)) +
    geom_point(col = colors$sakuri)     # Sakuri color code from color list

miyake <- read_delim("Miyake12.csv") %>%
    as_tibble() %>%
    transmute(
        DC14 = normalize(d14c),             # Normalised carbon 14
        year,                               # Preserving time data
        set = rep("miyake", length(year))   # Adding set for general data
    )

miyake_time_series <- miyake %>%
    ggplot(aes(x = year, y = DC14)) +
    geom_point(col = colors$miyake)

carbon14 <- rbind(miyake, sakuri)

carbon_density <- carbon14 %>%
    ggplot(aes(x = DC14, group = set)) +
    geom_density()

sakuri_grid <- grid.arrange(
    carbon_density,
    sakuri_time_series,
    miyake_time_series
)
