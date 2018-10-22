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

# Loading datasets and appending the first to the second.
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
# plot_X_data(segmentation_data, figure_handler)

## Visualization

# 1) Pearson's Correlation Matrix
# plot_pearsons_matrix(segmentation_data.loc[:, segmentation_data.columns != 'CLASS'], figure_handler)

# 2) Covariance Matrix
# plot_covariance_matrix(segmentation_data.loc[:, segmentation_data.columns != 'CLASS'], figure_handler)

# 3) Histograms
# bins_values = [5, 10, 12]
# for bins_count in bins_values:
	# plot_histograms(segmentation_data.loc[:, segmentation_data.columns != 'CLASS'], segmentation_data.loc[:, segmentation_data.columns == 'CLASS'], figure_handler, bins_count)

## Preprocessing

# 1) Normalization)
segmentation_data = normalize_min_max(segmentation_data)
# plot_X_data(segmentation_data, figure_handler)


# segmentation_data = normalize_z_score(segmentation_data)
# plot_X_data(segmentation_data, figure_handler)


# 2) Dimensionality Reduction
# PCA
# NOTICE: This algorithm works with z-score normalized data
# n_components = [1, 2, 4, 8, 16, 19]
# variance_ratios = []
# for n_comp in n_components:
# 	principal, var_ratio = project_pca(segmentation_data, n_components=n_comp)
# 	variance_ratios += [sum(var_ratio)]
# 	plot_pearsons_matrix(principal.loc[:, principal.columns != 'CLASS'], figure_handler)

# print('\n---------------------------PCA---------------------------')
# for i in range(len(n_components)):
# 	print("#components: " + str(n_components[i]) + "\t\t" + "variance ratio: "
# 		+ str(variance_ratios[i]))

# Feature Selection
# KBest
# NOTICE: This algorithm works with min-max normalized data
k = [4, 8, 16, 19]
for k_ in k:
	k_best = select_k_best(segmentation_data, k_)
	plot_pearsons_matrix(k_best.loc[:, k_best.columns != 'CLASS'], figure_handler)

plt.show()