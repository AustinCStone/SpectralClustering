import numpy as np
import matplotlib.pyplot as plt
import math as m

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA


def parse(filename):
	output = []
	lines = open(filename, 'r').read().splitlines()
	for line in lines:
		output.append(map(lambda x: float(x), line.split()))
	return np.asarray(output)


def dist(p1, p2):
	return np.sum(np.sqrt((p1-p2)**2.))


def construct_affinity(data, std_dev):
	mat = np.zeros((len(data), len(data)))
	for i, p1 in enumerate(data):
		for j, p2 in enumerate(data):
			if i == j:
				mat[i, j] = 0.
			else:
				mat[i, j] = m.exp((-dist(p1, p2)**2.)/std_dev)
	return mat


def construct_N(affinity_mat):
	(rows, columns) = affinity_mat.shape
	N = np.zeros((rows, columns))
	for row in range(rows):
		for column in range(columns):
			N[row, column] = affinity_mat[row, column] / \
				m.sqrt(np.sum(affinity_mat[row, :]) * np.sum(affinity_mat[:, column]))
	return N


def construct_Y(N, n_clusters):
	w, v = LA.eig(N)
	v_sorted = np.asarray([v for (w,v) in sorted(zip(w,v.transpose()), \
		reverse=True)][0:n_clusters]).transpose()
	row_sums = LA.norm(v_sorted, axis=1)
	return v_sorted / row_sums[:, np.newaxis]


def run_spectral_clustering(data, n_clusters, std_dev=.05):
	affinity_mat = construct_affinity(data, std_dev)
	N = construct_N(affinity_mat)
	Y = construct_Y(N, n_clusters)
	plot_kmeans(Y, n_clusters, plot_data=data)


def plot_kmeans(clustering_data, n_clusters, plot_data=None):
	result = KMeans(n_clusters=n_clusters).fit(clustering_data)
	c = []	
	for val in result.labels_:
		if val == 1:
			c.append([1., 1.,1.])
		else:
			c.append([0., 0., 0.])

	data = plot_data if plot_data is not None else clustering_data

	if len(data[0]) == 3:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(data[:,0], data[:,1], zs=data[:,2], c=c, depthshade=False)
		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')
		plt.show()
	elif len(data[0]) == 2:
		plt.scatter(data[:,0], data[:,1], c=c)
	else:
		raise ValueError('Unsupported number of dimensions')
	plt.show()

data = parse('concentric.txt')
run_spectral_clustering(data, 2)


