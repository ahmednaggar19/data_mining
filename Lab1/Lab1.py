import sys
sys.path.append('.')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import numpy as np
import collections
import visualizer

# import iris data_set
iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
y = iris.target

# Initiate Figures Handler
figure_handler = visualizer.Figures_handler()

# Cosine Similarity
figure_handler.create_figure()
plt.imshow(visualizer.compute_cosine_similarity_martix(X))
plt.title('Cosine Similarity')

# plot histogram
visualizer.plot_X_data(X, y, 4, figure_handler)

# Histograms
visualizer.plot_histogram(X, y, figure_handler)

# Scatter Plot
visualizer.plot_scatter_plot(X, y, figure_handler)

# 3D Scatter Plot
visualizer.plot_3D_scatter_plot(X, y, figure_handler)

# Show plots
plt.show()
