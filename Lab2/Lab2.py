import sys
sys.path.append('.')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
from visualizer import *

# loading datasets and appending the first to the second.
training_data = pd.read_csv('segmentation.data')
test_data = pd.read_csv('segmentation.test')
segmentation_data = training_data.append(test_data)

# Exploring Dataset
display_attributes_count(segmentation_data)
display_classes_count(segmentation_data)

# Creating Figure Handler Object to handle different Figures
figure_handler = Figures_handler()

segmentation_data = segmentation_data.sort_values(['CLASS'])

# Plot X_data before any normalization
plot_X_data(segmentation_data, figure_handler)

## Visualization

# 1) Pearson's Correlation Matrix
# plot_pearsons_matrix(segmentation_data.loc[:, segmentation_data.columns != 'CLASS'], figure_handler)

# 2) Covariance Matrix
# plot_covariance_matrix(segmentation_data.loc[:, segmentation_data.columns != 'CLASS'], figure_handler)

# 3) Histograms
# bins_values = [5, 10, 12]
# for bins_count in bins_values:
# 	plot_histograms(segmentation_data.loc[:, segmentation_data.columns != 'CLASS'], segmentation_data.loc[:, segmentation_data.columns == 'CLASS'], figure_handler, bins_count)

## Preprocessing

# 1) Normalization)
# segmentation_data = normalize_min_max(segmentation_data)
# plot_X_data(segmentation_data, figure_handler)


segmentation_data = normalize_z_score(segmentation_data)
plot_X_data(segmentation_data, figure_handler)


plt.show()