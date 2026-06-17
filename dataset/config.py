"""Shared configuration for dataset plotting/split scripts."""

FASTA_FILES = {
    "DENV1_merged_meta.fasta": "DENV1",
    "DENV2_merged_meta.fasta": "DENV2",
    "DENV3_merged_meta.fasta": "DENV3",
    "DENV4_merged_meta.fasta": "DENV4",
}

SEROTYPES = ["DENV1", "DENV2", "DENV3", "DENV4"]
SEROTYPE_COLORS = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]
SHOW_TITLES = False

# Shared stats-figure style.
# BAR_SLOT_INCHES controls the physical spacing allocated to each x-axis category,
# so stacked bars have comparable visual widths across CD-HIT, continent, and temporal plots.
STATS_BAR_WIDTH = 0.75
STATS_BAR_SLOT_INCHES = 0.95
STATS_BAR_MARGIN_INCHES = 2.3
STATS_BAR_MIN_FIGWIDTH = 6.5
STATS_BAR_HEIGHT = 6.0

# Larger font sizes for paper-style stats figures and the 2x2 panel.
STATS_AXIS_LABEL_FONTSIZE = 13
STATS_TICK_FONTSIZE = 11
STATS_LEGEND_FONTSIZE = 12
STATS_TITLE_FONTSIZE = 14
STATS_PANEL_TITLE_FONTSIZE = 16
STATS_PANEL_LEGEND_FONTSIZE = 13
STATS_PANEL_TICK_FONTSIZE = 13
STATS_PANEL_AXIS_LABEL_FONTSIZE = 15
STATS_PANEL_PIE_TEXT_FONTSIZE = 12
STATS_PANEL_PIE_SMALL_SLICE_Y_SHIFT = 0.09

# Split settings. One fold is generated for each plotted group.
GENERATE_SPLITS = True
VAL_SIZE = 0.10
RANDOM_STATE = 42
SPLIT_SEP = ","
INCLUDE_UNKNOWN_SPLIT = True

# Temporal binning.
CUTOFF_YEAR = 2006
BIN_LENGTH = 5

COUNTRY_ALIASES = {
    "DRC": "Democratic Republic of the Congo",
}
