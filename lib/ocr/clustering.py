from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

Bounding_Box = tuple[int, int, int, int]


class Clustering:

    def get_box_center(self, box: Bounding_Box) -> tuple[int, int]:
        delta_x = box[2] - box[0]
        delta_y = box[3] - box[1]
        x = box[0] + round(delta_x / 2)
        y = box[1] + round(delta_y / 2)
        return x, y

    def predict_groups(self, bounding_boxes: list[Bounding_Box]):
        # if there are only a few boxes, treat them all as one text block, if there are more use k-means
        if len(bounding_boxes) <= 3:
            return bounding_boxes

        groups = []
        coordinates = np.array([self.get_box_center(box) for box in bounding_boxes])

        if len(coordinates) <= 5:
            min_k_value = 2
            max_k_value = 3
        else:
            min_k_value = 3
            max_k_value = min(len(coordinates), 12)  # don't allow more than 12 text groups

        k_values = range(min_k_value, max_k_value)
        silhouette_scores = []

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(coordinates)
            silhouette_scores.append(silhouette_score(coordinates, kmeans.labels_))

        optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]

        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')  # TODO: test different n_init values
        kmeans.fit(coordinates)

        labels, centers = kmeans.labels_, kmeans.cluster_centers_

        for i in range(optimal_k):
            boxes = [box for box, label in zip(bounding_boxes, labels) if label == i]
            groups.append(boxes)

        return groups
