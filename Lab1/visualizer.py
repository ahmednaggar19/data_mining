import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import numpy as np
import collections

class Figures_handler:
    def __init__(self):
    	self.figure_count = 0;

    def create_figure(self):
        self.figure_count = self.figure_count + 1
        return plt.figure(self.figure_count)


def cosine_similarity1(X, Y):
	return cosine(X, Y)

def compute_cosine_similarity_martix(X) :
	return cosine_similarity(X)

def get_classes_X (X, Y):
	class_ids = np.unique(Y)
	counters = collections.Counter(Y)
	class_X = {}
	for class_id in class_ids:
		class_X[class_id] = np.reshape(X[np.where(Y == class_id), :], (counters[class_id], 4))
	return (class_ids, class_X)

def plot_histogram(X, Y, figure_handler) :
	class_ids, class_X = get_classes_X(X, Y)
	figure_handler.create_figure()
	plot_index = 1
	for class_id in class_ids :
		for feature in range(X.shape[1]):
			plt.subplot(3, 4, plot_index)
			plt.hist(x=class_X[class_id][: , feature], bins=10)
			plt.grid(axis='y', alpha=0.75)
			plot_index = plot_index + 1

def plot_X_data(X, Y, no_of_features, figure_handler):
	class_ids = np.unique(Y)
	counters = collections.Counter(Y)
	for class_id in class_ids:
		figure_handler.create_figure()
		plt.boxplot(np.reshape(X[np.where(Y == class_id), :no_of_features], (counters[class_id], no_of_features))) 
		plt.title('boxplots for class ' + str(class_id))

def plot_scatter_plot(X, Y, figure_handler):
	figure_handler.create_figure()
	plot_index = 1
	plt.title("Data ScatterPlot")
	class_ids, class_X = get_classes_X(X, Y)
	for i in range(X.shape[1]):
		for j in range(X.shape[1]):
			plt.subplot(X.shape[1], X.shape[1], plot_index)
			for class_id in class_ids :
				plt.scatter(class_X[class_id][:, i], class_X[class_id][:, j])
			plot_index = plot_index + 1

def plot_3D_scatter_plot(X, Y, figure_handler):
	fig = figure_handler.create_figure()
	plot_index = 1
	plt.title('Data 3D ScatterPlot')
	class_ids, class_X = get_classes_X(X, Y)
	for i in range(X.shape[1]):
		for j in range(i + 1, X.shape[1]):
			for k in range(j + 1, X.shape[1]):
				ax = fig.add_subplot(2, 2, plot_index, projection='3d')
				for class_id in class_ids :
					ax.scatter(class_X[class_id][:, i], class_X[class_id][:, j], class_X[class_id][:, k])
				plot_index = plot_index + 1
