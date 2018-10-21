import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
import collections

class Figures_handler:
    def __init__(self):
    	self.figure_count = 0;

    def create_figure(self):
        self.figure_count = self.figure_count + 1
        return plt.figure(self.figure_count)

def display_attributes_count(dataframe):
	print("Number of Features : " + str(dataframe.columns.shape[0]))

def display_classes_count(dataframe):
	print("Number of Classes : " + str(len(np.unique(dataframe['CLASS']))))

def plot_X_data(dataframe, figure_handler):
	X = dataframe.loc[:, dataframe.columns != 'CLASS']
	figure_handler.create_figure()
	dataframe.boxplot()

def plot_pearsons_matrix(dataframe, figure_handler):
	figure_handler.create_figure()
	plt.imshow(dataframe.corr())

def plot_covariance_matrix(dataframe, figure_handler):
	figure_handler.create_figure()
	plt.imshow(dataframe.cov())


def get_classes_X (X, Y):
	Y = Y.iloc[:, 0].values
	X = X.iloc[:, 1:].values
	class_ids = np.unique(Y)
	counters = collections.Counter(Y)
	class_X = {}
	for class_id in class_ids:
		class_X[class_id] = np.reshape(X[np.where(Y == class_id), :], (counters[class_id], X.shape[1]))
	return (class_ids, class_X, X, Y)

def plot_histograms(X, Y, figure_handler, bins) :
	class_ids, class_X, X, Y = get_classes_X(X, Y)
	figure_handler.create_figure()
	plot_index = 1
	for class_id in class_ids :
			plt.subplot(2, 4, plot_index)
			plt.hist(x=class_X[class_id]	, bins=bins)
			plt.grid(axis='y', alpha=0.75)
			plot_index = plot_index + 1

def normalize_min_max(dataframe):
	X = dataframe.loc[:, dataframe.columns != 'CLASS']
	scaler = preprocessing.MinMaxScaler()
	# scaled = scaler.fit_transform(X.values)
	cols = list(dataframe.columns)
	cols.remove('CLASS')
	for column in cols:
		dataframe[column] = scaler.fit_transform(dataframe[column].values.reshape(-1, 1))
	# dataframe.loc[:, dataframe.columns != 'CLASS'] = (pd.DataFrame(data=scaled))
	return dataframe

def normalize_z_score(dataframe):
	cols = list(dataframe.columns)
	cols.remove('CLASS')
	for column in cols:
		dataframe[column] = (dataframe[column] - dataframe[column].mean()) / dataframe[column].std(ddof=0)
	return dataframe

def project_pca(dataframe, n_components):
	X = dataframe.loc[:, dataframe.columns != 'CLASS']
	Y = dataframe.loc[:, dataframe.columns == 'CLASS']

	pca = PCA(n_components=n_components)
	pca.fit_transform(X.values)

	principal_df = pd.Dataframe(data=dataframe,
		columns=['PC' + str(i) for i in range(n_components)])
	return pd.concat([principalDf, Y], axis=1)
