import ast
from collections import Counter
import re
import ast

import json
from tqdm import tqdm
from math import log
import heapq

from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.tokenize import word_tokenize
import nltk
from collections import OrderedDict

import divergences
from divergences import Jensen_Shannon
import numpy as np
import random
import importlib
importlib.reload(divergences)
from divergences import Jensen_Shannon
from scipy.sparse import csr_matrix
from scipy.special import rel_entr
from heapq import heappush, heappop
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from multiprocessing import Pool, cpu_count
import os
from tqdm import tqdm
import math


class Newness():
    """
    Estimate the ratio of new terms in the distribution and the ratio of disappearing terms
    known_P is the known distribution (Knowledge or Expectation Base) and new_Q should be the novel distribrution (Determine if the document is new or not)
    The two option are mathematicallyequivalent if you set equivalent threshold -- to choose based on your ease of interpretation
    """
    def __init__(self, known_P, new_Q, lambda_=0.8):

        self.known_P = known_P
        self.new_Q = new_Q
        self.lambda_ = lambda_
        
        JS = Jensen_Shannon()
        self.JSD_vector = JS.linear_JSD(known_P, new_Q)
        self.nb_elements = len(self.JSD_vector)

    def divergent_terms(self, thr_div=0.1, thr_new=0.5):

        """
        JSD == 0 if and only if pi = qi, but we want to make sure the distribution gap between this two are large enough
        To interpret as if the new term make the divergence greater than threshold, then it is a significant cointributing, we just need to know in appearing or disappearing
        """
        count_appear = 0
        count_disappear = 0
        for i in range(self.nb_elements):
            if self.JSD_vector[i] > thr_div:
                if self.new_Q[i] > self.known_P[i]:
                    count_appear += 1
                if self.known_P[i] > self.new_Q[i]:
                    count_disappear += 1

        appear_ratio = count_appear / self.nb_elements
        disappear_ratio = count_disappear / self.nb_elements
        newness = self.lambda_ * appear_ratio + (1-self.lambda_) * disappear_ratio
        novelty = 0
        if newness > thr_new:
            novelty = 1
        
        return newness, novelty

    def probable_terms(self, thr_prob=2, cte = 1e-10, thr_new=0.5):
        """
        To interpret as if the new term has a probability of appearing thr_new times greater than old doc, it is a new significant term. Reverse is true for disappearing
        """
        count_appear = 0
        count_disappear = 0
        for i in range(self.nb_elements):
            if self.JSD_vector[i] != 0:
                # To interpret as if the new term has a probability of appearing thr_new times greater than old doc, it is a new significant term. Reverse is true for disappearing
                if self.new_Q[i] / (self.known_P[i]+cte) > thr_prob:  
                    count_appear += 1
                if self.known_P[i] / (self.new_Q[i]+cte) > thr_prob:
                    count_disappear += 1
        
        appear_ratio = count_appear / self.nb_elements
        disappear_ratio = count_disappear / self.nb_elements
        newness = self.lambda_ * appear_ratio + (1 - self.lambda_) * disappear_ratio
        novelty = 0
        if newness > thr_new:
            novelty = 1
        
        return newness, novelty


class Uniqueness():
    """
        We estimate the distance between an new distribution and the overall generall distribution
    """
    def __init__(self, known_P):
        self.known_P = known_P
        #self.new_Q = new_Q
        self.JS = Jensen_Shannon()
        
    def dist_to_proto(self, new_Q, thr_uniq=0.05):
        
        novel_uniq = 0
        uniqueness_ = self.JS.JSDiv(self.known_P, new_Q)
        if uniqueness_ > thr_uniq:
            novel_uniq = 1
            
        return uniqueness_, novel_uniq

    def proto_dist_shift(self, new_P, thr_uniqp=0.05):
        
        #new_P = self.known_P + self.new_Q
        uniqueness = self.JS.JSDiv(self.known_P, new_P)
        novel_uniq = 0
        if uniqueness > thr_uniqp:
            novel_uniq = 1

        return uniqueness, novel_uniq

    
class Difference():
    """
        We estimate the ratio of point that are in close vicinity of the point. 
        list_know_P : represent the list of all distribution vectors for each individual documents
    """
    def __init__(self, list_know_P, new_Q, N=5):

        self.list_know_P = list_know_P.tocsr()
        self.new_Q = new_Q
        self.JS = Jensen_Shannon()
        self.N = N
       

    def dist_estimate(self, sample=True, sample_size=1000, do_sample_P=True):
        """
        Computes the average Jensen-Shannon (JS) distance between a point and its 
        `N` closest neighbors, based on a subset of the data for computational efficiency.

        Parameters:
        ----------
        sample : bool, optional, default=True
            Whether to take a random sample of points from `list_know_P` 
            for which to compute the closest neighbors. If `False`, all points are used.
            
        sample_size : int, optional, default=1000
            The number of points to sample for computing distances when `sample` is `True`.
            Ignored if `sample` is `False`.
            
        do_sample_P : bool, optional, default=True
            Whether to take a smaller random subset (of size 10,000) of `list_know_P` 
            to use as the set of candidate neighbors for distance calculations. 
            If `False`, all points in `list_know_P` are used as candidates.

        """
        num_points = self.list_know_P.shape[0]
        
        # Ensure the matrix is in CSR format for efficient row access
        self.list_know_P = self.list_know_P.tocsr()

        # Randomly sample `sample_size` indices
        if sample:
            sampled_indices = random.sample(range(num_points), min(sample_size, num_points))
        else: 
            sampled_indices = num_points

        # Sample of list_know_P
        if do_sample_P:
            sample_P = random.sample(range(num_points), min(10000, num_points))
        else:
            sample_P = num_points

        avg_dists = []  # Store average distances
        all_dists_per_point = []  # Store all distances for each point

        def compute_distance(j):
                P_j = self.list_know_P[j].toarray().flatten()
                return Jensen_Shannon().JSDiv(P_i, P_j)

        for i in tqdm(sampled_indices):
            P_i = self.list_know_P[i].toarray().flatten()  # Convert sparse row to dense array

            # Compute distances to all other points in parallel
            all_dists = Parallel(n_jobs=-1, batch_size=int(num_points / os.cpu_count()))(
                delayed(compute_distance)(j) for j in sample_P if j != i  
            )
            # Take the smallest N distances if there are enough distances
            if len(all_dists) > self.N:
                all_dists = heapq.nsmallest(self.N, all_dists)

            avg_dist_i = sum(all_dists) / len(all_dists)
            avg_dists.append(avg_dist_i)
            all_dists_per_point.append(all_dists)

        # Compute final average distance
        avg_final = sum(avg_dists) / len(avg_dists)
        return avg_final


    def ratio_to_all(self, neighbor_dist, thr_diff=0.95):
        """
        Computes the ratio of points where the distance to `new_Q` exceeds `neighbor_dist`.
        """
        count_diff = 0
        num_known_P = self.list_know_P.shape[0]
        dists =[]
        for i in range(num_known_P):
            P_i = self.list_know_P[i].toarray().flatten()
            distance = self.JS.JSDiv(P_i, self.new_Q)
            dists.append(distance)
            if distance > neighbor_dist:
                count_diff += 1

        # Compute the proportion of points with distances exceeding the threshold
        difference = count_diff / num_known_P
        novel_diff = int(difference > thr_diff)

        return difference, novel_diff, dists

    def ratio_to_neighbors_fC(self, neighbor_dist, thr_diff=0.85):
        count_diff = 0
        #We compute all distances to identify the closest neighbors
        all_dists = []
        for P_i in tqdm(self.list_know_P):
            distance = self.JS.JSDiv_fC(P_i, self.new_Q)
            all_dists.append(distance)
        closests = heapq.nsmallest(self.N, all_dists)
        #We check the proportion of neighbors that are closer that it should be on average
        for dist in closests: 
            if dist >= neighbor_dist:
                count_diff += 1

        #Proportion of neighbor points where the distance is superior to the average distance to normal neighbors -- the higher the more different
        difference = count_diff / len(closests)
        novel_diff = 0
        if difference > thr_diff:
            novel_diff = 1
        mean100 = np.mean(closests)
        return difference, novel_diff, mean100


    def ratio_to_neighbors(self, neighbor_dist, thr_diff=0.85):
        """
        Computes the ratio of nearest neighbors where the distance to new_Q exceeds neighbor_dist.
        """
        count_diff = 0
        num_known_P = self.list_know_P.shape[0]
        all_dists = []
        list_know_P = self.list_know_P.tocsr()  # Ensure CSR format for efficiency
        new_Q = self.new_Q

        # Compute distances to all points
        for i in tqdm(range(num_known_P)):
            P_i = list_know_P[i].toarray().flatten()
            all_dists.append(Jensen_Shannon().JSDiv(P_i, new_Q))

        # Identify the closest N neighbors
        closest_dists = heapq.nsmallest(self.N, all_dists)

        # Count neighbors with distances exceeding the threshold
        count_diff = sum(1 for dist in closest_dists if dist > neighbor_dist)

        # Compute the proportion of neighbors exceeding the threshold
        difference = count_diff / len(closest_dists)
        novel_diff = int(difference > thr_diff)
        mean100 = np.mean(closest_dists)

        return difference, novel_diff, mean100
  


    def ratio_to_neighbors_joblib(self, neighbor_dist, thr_diff=0.85):
        """
        Computes the ratio of nearest neighbors where the distance to new_Q exceeds neighbor_dist.
        """
        new_Q = self.new_Q
        list_know_P = self.list_know_P
        def compute_distance(i):
            P_i = list_know_P[i].toarray().flatten()
            return Jensen_Shannon().JSDiv(P_i, new_Q)
        
        # Compute distances to all points in parallel
        all_dists = Parallel(n_jobs=-1, batch_size=int(list_know_P.shape[0]/os.cpu_count()))(delayed(compute_distance)(i) for i in (range(list_know_P.shape[0])))
        
        # Identify the closest N neighbors
        closests = heapq.nsmallest(self.N, all_dists)
        count_diff = sum(1 for dist in closests if dist > neighbor_dist)
        
        # Compute the proportion of neighbors exceeding the threshold
        difference = count_diff / len(closests)
        novel_diff = int(difference > thr_diff)
        mean100 = np.mean(closests)

        return difference, novel_diff, closests, mean100
    

class ClusterKS(Difference):
    def __init__(self, list_know_P, new_Q, N, nbPtsPerCluster):
        super().__init__(list_know_P, new_Q, N)

        self.nbPtsPerCluster=nbPtsPerCluster

    def clusterKS(self):
        # Reduce dimensionality for faster clustering (optional)

        # Perform clustering
        n_clusters = int(self.list_know_P.shape[0]/self.nbPtsPerCluster)  # Set based on dataset size and structure
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        # print(n_clusters)
        labels = kmeans.fit_predict(self.list_know_P)

        # Cluster assignments
        clusters = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(labels):
            clusters[label].append(idx)
        
        self.clusters = clusters
        # self.pca = pca
        self.kmeans = kmeans
        return clusters, kmeans
    

    def ratio_to_neighbors_kmeans(self, variation_dist, neighbor_dist=0, thr_diff=0.85, nb_clusters=4):
        """
        Compute the ratio of neighbors' distributions within the closest k-means clusters 
        based on Jensen-Shannon divergence.

        Parameters:
        - variation_dist (array-like): The target distribution for comparison.
        - nb_clusters (int): The number of closest clusters to consider.

        Returns:
        - kmean_closest (list): The smallest N Jensen-Shannon divergences.
        - mean100 (float): The mean of the smallest N Jensen-Shannon divergences.
        """
        # Compute distances from the k-means cluster centers to the target distribution
        cluster_dists = np.linalg.norm(self.kmeans.cluster_centers_ - variation_dist, axis=1)
        closest_clusters = np.argsort(cluster_dists)[:nb_clusters]

        # Gather indices of points in the closest clusters
        closest_indices = []
        for cluster_idx in closest_clusters:
            closest_indices.extend(self.clusters[cluster_idx])

        # Subset the known probability distributions to those in the closest clusters
        closest_distributions = self.list_know_P[closest_indices]

        def compute_jsd(i):
            """
            Compute the Jensen-Shannon divergence between the target distribution 
            and a specific distribution from the closest clusters.
            """
            P = closest_distributions[i].toarray().flatten()  # Convert sparse row to dense
            return Jensen_Shannon().JSDiv(P=P, Q=variation_dist)

        # Parallelize computation of Jensen-Shannon divergences
        batch_size = max(1, int(closest_distributions.shape[0] / os.cpu_count()))
        js_divergences = Parallel(n_jobs=-1, batch_size=batch_size)(
            delayed(compute_jsd)(i) for i in range(closest_distributions.shape[0])
        )

        
        # Find the smallest N Jensen-Shannon divergences
        kmean_closest = heapq.nsmallest(self.N, js_divergences)
        print(kmean_closest)
        count_diff = sum(1 for dist in kmean_closest if dist > neighbor_dist)

        dif_score = count_diff / len(kmean_closest)
        dif_bin = int(dif_score > thr_diff)
        mean100 = np.mean(kmean_closest)

        return dif_score, dif_bin, mean100 #, kmean_closest, mean100
    


    
    def dist_estimate_clusters(self, iterations, nb_clusters):
        self.nb_clusters = nb_clusters
        def process_random_point(random_point):
            cluster_dists = np.linalg.norm(self.kmeans.cluster_centers_ - random_point, axis=1)
            closest_clusters = np.argsort(cluster_dists)[:self.nb_clusters]

            closest_indices = []
            for cluster_idx in closest_clusters:
                closest_indices.extend(self.clusters[cluster_idx])
            closest_distributions = self.list_know_P[closest_indices]

            P = closest_distributions  
            js_divergences = np.array([Jensen_Shannon().JSDiv(P[i].toarray().flatten(), random_point) for i in range(P.shape[0])])
            closest_divergences = heapq.nsmallest(self.N, js_divergences)
            return np.mean(closest_divergences)

        num_points = self.list_know_P.shape[0]
        random_indices = np.random.choice(num_points, size=iterations, replace=False)  # Select random points upfront
        random_points = [self.list_know_P[idx].toarray().flatten() for idx in random_indices]

        # Parallelize across random points
        results = Parallel(n_jobs=-1)(
            delayed(process_random_point)(random_point) for random_point in random_points
        )

        return np.mean(results)






