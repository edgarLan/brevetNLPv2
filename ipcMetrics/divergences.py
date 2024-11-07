from scipy.sparse import csr_matrix
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from itertools import combinations
from collections import Counter
import tqdm
import math

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
    
    @staticmethod
    def pmi(list_ipc):
        # Count occurrences of each IPC code
        ipc_counts = Counter(ipc for sublist in list_ipc for ipc in sublist)
        
        # Count co-occurrences of each IPC pair
        pair_counts = Counter()
        for sublist in list_ipc:
            for pair in combinations(sublist, 2):  # Generate all pairs in the sublist
                sorted_pair = tuple(sorted(pair))
                pair_counts[sorted_pair] += 1
        
        # Total patents considered
        total_patents = len(list_ipc)
        
        # Calculate PMI for each pair
        pmi_values_dict = {}
        for (ipc1, ipc2), pair_count in pair_counts.items():
            p_x = ipc_counts[ipc1] / total_patents
            p_y = ipc_counts[ipc2] / total_patents
            p_xy = pair_count / total_patents
            
            # Calculate PMI if none of the probabilities are zero
            if p_x > 0 and p_y > 0 and p_xy > 0:
                pmi_values_dict[(ipc1, ipc2)] = math.log(p_xy / (p_x * p_y))
    
        return pmi_values_dict

