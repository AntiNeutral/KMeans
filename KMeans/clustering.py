import numpy as np
from numpy import random
import torch as pt
from typing import Any


class KCluster:
    def __init__(self, data: np.ndarray, cluster_num: Any = range(2, 5), gpu: bool = False) -> None:
        """
        :param cluster_num: An iterable of numbers of clusters.
        :param gpu: Computations are to be done on gpu only if gpu is set to True. It is False by default
        """
        self.cluster_num = cluster_num
        self.dimension = data.shape
        self.results = None
        self.best = None
        self.devices = pt.device('cuda') if gpu else pt.device('cpu')
        self.data = pt.from_numpy(data).to(self.devices).float()

    def centroids(self, labels: pt.Tensor, cluster_num: int) -> pt.Tensor:
        # TODO:
        """
        Compute the centroids given the labels
        :param labels: array of labels
        :param cluster_num: number of clusters
        :return: centroids as row vectors stacked vertically
        """
        return self._compute_centers(self._recover(labels, cluster_num))

    def _recover(self, groups: np.ndarray | pt.Tensor, count):
        """
        convert a 1d label array to 2d 1-of-k encoding label matrix
        :param count: number of clusters
        :param groups: arrary / Tensor that records the group labels
        :return: n by count label matrix of 1 and 0s
        """
        row_number = self.dimension[0]
        zeros = pt.zeros(row_number, count, device=self.devices)
        zeros[range(row_number), list(groups)] = 1
        return zeros

    def _initial_assignment(self, clusters: int) -> pt.Tensor:
        initial_assignment = random.randint(0, clusters, self.dimension[0])
        return self._recover(initial_assignment, clusters)

    def _compute_centers(self, label_matrix: pt.Tensor) -> pt.Tensor:
        pass

    def _assign_centers(self, centers) -> np.ndarray | pt.Tensor:
        pass

    def single_fit(self, initial_groups: np.ndarray | pt.Tensor):
        """
        Perform clustering with cosine similarity
        :param initial_groups: label matrix
        :return:
        """
        centers = self._compute_centers(initial_groups)
        previous_groups = initial_groups
        for i in range(1000):
            current_groups = self._assign_centers(centers)
            if pt.equal(current_groups, previous_groups):
                return pt.nonzero(previous_groups, as_tuple=True)[1], self._wss(previous_groups, centers)
            centers = self._compute_centers(current_groups)
            previous_groups = current_groups
        return pt.nonzero(previous_groups, as_tuple=True)[1], self._wss(previous_groups, centers)

    def distance_matrix(self) -> np.ndarray | pt.Tensor:
        pass

    def silhouette(self, groups: np.ndarray | pt.Tensor, count):
        label_matrix = self._recover(groups, count)
        group_card = pt.sum(label_matrix, dim=0) - label_matrix
        distances = (self.distance_matrix() @ label_matrix) / group_card
        within_distances = pt.sum(pt.multiply(distances, label_matrix), dim=1)
        value, index = pt.topk(pt.multiply(distances, 1 - label_matrix), k=2, largest=False, dim=1)
        between_distance = value[:, -1]
        return pt.mean((between_distance - within_distances) / pt.maximum(between_distance, within_distances))

    def _ss_distance_to_center(self, centers: np.ndarray | pt.Tensor) -> np.ndarray | pt.Tensor:
        pass

    def _wss(self, groups: np.ndarray | pt.Tensor, centers: np.ndarray | pt.Tensor) -> np.ndarray | pt.Tensor:
        """
        Compute the within sum of square
        :param centers: centers as row vecters
        :param groups: n by t label matrix
        :return: a scalar, sum of squares
        """
        row_centers = groups @ centers
        return pt.sum(self._ss_distance_to_center(row_centers))

    def fit(self, num_init: int = 1) -> None:
        """
        Perform clustering using algorithm analogue to the tranditional K means clustering
        :return:
        """
        first = self.cluster_num[0]
        results = {}
        initial_group = self._initial_assignment(first)
        result = self.single_fit(initial_group)
        results[first] = result
        for n in self.cluster_num:
            initial_group = self._initial_assignment(n)
            result = self.single_fit(initial_group)
            for i in range(num_init - 1):
                initial_group = self._initial_assignment(n)
                current = self.single_fit(initial_group)
                result = current if current[1] < result[1] else result
            results[n] = result + (self.silhouette(result[0], n), ) if len(result[0]) > 0 else result + (-1, )
        self.results = results
        if len(self.cluster_num) > 3:
            s_scores = {i: results[i][1].cpu() for i in results}
            diffs = {i: s_scores[i + 1] + s_scores[i - 1] - 2 * s_scores[i] for i in self.cluster_num[1:-1]}
            turn_point = [key for key, values in diffs.items() if values == max(diffs.values())]
            self.best = (turn_point[0], ) + results[turn_point[0]]


class KCosine(KCluster):
    def __int__(self, data: np.ndarray, cluster_num: range = range(2, 5), gpu: bool = False) -> None:
        super().__init__(data, cluster_num, gpu)

    def _compute_centers(self, label_matrix) -> np.ndarray | pt.Tensor:
        """
        compute the cosine between the target and each row of data
        :param label_matrix: n by t matrix with 1-of-k encoding
        :return: row vectors of centers
        """
        means = label_matrix.T @ self.data
        return means / pt.norm(means, dim=1, p=2).view(label_matrix.shape[1], -1)

    def _assign_centers(self, centers: np.ndarray | pt.Tensor) -> np.ndarray | pt.Tensor:
        """
        assign centers to each row
        :param centers: centers as rows stacked vertically
        :return: a 1d array indicating group number of each row
        """
        distances = self.data @ centers.T
        return (distances == pt.max(distances, dim=1)[0].view(-1, 1)).float()

    def distance_matrix(self) -> np.ndarray | pt.Tensor:
        return pt.sqrt(pt.abs(1 - self.data @ self.data.T))

    def _ss_distance_to_center(self, centers: np.ndarray | pt.Tensor) -> np.ndarray | pt.Tensor:
        return 1 - pt.matmul(self.data.view(self.dimension[0], 1, -1), centers.view(self.dimension[0], -1, 1)).squeeze()

    @staticmethod
    def predict(centroids: pt.Tensor, data: pt.Tensor):
        distances = data @ centroids.T
        label_matrix = (distances == pt.max(distances, dim=1)[0].view(-1, 1)).float()
        return pt.nonzero(label_matrix, as_tuple=True)[1]


class KMean(KCluster):
    def __int__(self, data: np.ndarray, cluster_num: range = range(2, 5), gpu: bool = False) -> None:
        super().__init__(data, cluster_num, gpu)

    def _compute_centers(self, label_matrix) -> np.ndarray | pt.Tensor:
        """
        compute the cosine between the target and each row of data
        :param label_matrix: n by t matrix with 1-of-k encoding
        :return: row vectors of centers
        """
        means = label_matrix.T @ self.data
        return means / pt.sum(label_matrix, dim=0).unsqueeze(1)

    def _assign_centers(self, centers: np.ndarray | pt.Tensor) -> np.ndarray | pt.Tensor:
        """
        assign centers to each row
        :param centers: centers as rows stacked vertically
        :return: a 1d array indicating group number of each row
        """
        diff = self.data.view(self.dimension[0], 1, 1, -1) - centers.reshape(1, centers.shape[0], 1, -1)
        distances = pt.matmul(diff, diff.transpose(2, 3)).squeeze()
        return (distances == pt.min(distances, dim=1)[0].view(-1, 1)).float()

    def _ss_distance_to_center(self, centers: np.ndarray | pt.Tensor) -> np.ndarray | pt.Tensor:
        diff = self.data - centers
        row_num = self.dimension[0]
        return pt.matmul(diff.view(row_num, 1, -1), diff.view(row_num, -1, 1)).squeeze()

    def distance_matrix(self) -> np.ndarray | pt.Tensor:
        diff = self.data.view(self.dimension[0], 1, 1, -1) - self.data.view(1, self.dimension[0], 1, -1)
        return pt.sqrt(pt.matmul(diff, diff.transpose(2, 3)).squeeze())

    @staticmethod
    def predict(centroids: pt.Tensor, data: pt.Tensor):
        diff = data.view(data.shape[0], 1, 1, -1) - centroids.reshape(1, centroids.shape[0], 1, -1)
        distances = pt.matmul(diff, diff.transpose(2, 3)).squeeze()
        label_matrix = (distances == pt.min(distances, dim=1)[0].view(-1, 1)).float()
        return pt.nonzero(label_matrix, as_tuple=True)[1]
