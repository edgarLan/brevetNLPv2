

from scipy.sparse import csr_matrix
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

class Distance():

    @staticmethod
    def jaccard_dist(set1, set2):
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return 1-len(intersection) / len(union)
    
    @staticmethod

    def jacc_dist_matrix(list1, list2):
        # Combine the lists for binarization
        combined_data = list1 + list2

        # Initialize and fit MultiLabelBinarizer on the combined data
        mlb = MultiLabelBinarizer()
        mlb.fit(combined_data)

        # Transform each dataset individually into binary matrices
        list1_matrix = mlb.transform(list1)
        list2_matrix = mlb.transform(list2)

        # Convert binary matrices into sparse format
        list1_sparse = csr_matrix(list1_matrix)
        list2_sparse = csr_matrix(list2_matrix)

        # Calculate the intersection as a sparse dot product
        intersection_sparse = list1_sparse.dot(list2_sparse.T)

        # Calculate the union using sparse sum operations
        list1_sum = list1_sparse.sum(axis=1).A1  # This will be a sparse matrix
        list2_sum = list2_sparse.sum(axis=1).A1  # This will be a sparse matrix

        # Convert sums to a dense format for union calculation (sparsity needed here)
        union_sparse = list1_sum[:, np.newaxis] + list2_sum - intersection_sparse.toarray()

        # Jaccard similarity and distance
        jaccard_similarity_sparse = intersection_sparse.toarray() / union_sparse  # Element-wise division
        jaccard_matrix_sparse = 1 - jaccard_similarity_sparse

        return jaccard_matrix_sparse
