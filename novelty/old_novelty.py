

    # def ratio_to_neighbors_kmeans_2(self, variation_dist):
    #     # Precompute cluster distances and get the 3 closest clusters
    #     cluster_dists = np.linalg.norm(self.kmeans.cluster_centers_ - variation_dist, axis=1)
    #     closest_clusters = np.argsort(cluster_dists)[:3]

    #     # Get indices of points in the 3 closest clusters
    #     closest_indices = np.concatenate([self.clusters[cluster_idx] for cluster_idx in closest_clusters])
    #     closest_distributions = self.list_know_P[closest_indices]

    #     # Pre-convert variation_dist to a dense array if needed
    #     # variation_dist_dense = variation_dist.toarray().flatten() if hasattr(variation_dist, "toarray") else variation_dist

    #     # Pre-convert closest distributions to dense arrays (if sparse)
    #     if hasattr(closest_distributions, "toarray"):
    #         closest_distributions = closest_distributions.toarray()

    #     # Function to compute JSD
    #     def compute_jsd(i):
    #         return Jensen_Shannon().JSDiv(P=closest_distributions[i], Q=variation_dist)

    #     # Parallelize JSD computation
    #     js_divergences = Parallel(n_jobs=-1, backend="loky")(delayed(compute_jsd)(i) for i in range(closest_distributions.shape[0]))

    #     # Get the N smallest distances and their mean
    #     kmean_closest = heapq.nsmallest(self.N, js_divergences)
    #     mean100 = np.mean(kmean_closest)

    #     return kmean_closest, mean100
    
    # def mean_js_divergence_across_clusters(self, num_closest=100):
    #     """
    #     Computes the mean JS divergence across clusters.

    #     Parameters:
    #         cluster_dict (dict): A dictionary mapping cluster IDs to lists of point indices.
    #         num_closest (int): Number of closest points to consider for mean calculation (default is 100).
        
    #     Returns:
    #         float: The average of the means of JS divergences within clusters.
    #     """
    #     def process_cluster(cluster_id, indices):
    #         if len(indices) < num_closest:
    #             print(f"Cluster {cluster_id} has fewer than {num_closest} points. Skipping.")
    #             return None
            
    #         # Select a random point from the cluster
    #         random_point_idx = random.choice(indices)
    #         random_point = self.list_know_P[random_point_idx].toarray().flatten()
            
    #         # Compute JS divergence of the random point to all other points in the cluster
    #         divergences = [
    #             Jensen_Shannon().JSDiv_csr(random_point, self.list_know_P[idx].toarray().flatten())
    #             for idx in indices if idx != random_point_idx
    #         ]
            
    #         # Get the num_closest smallest divergences
    #         closest_divergences = heapq.nsmallest(num_closest, divergences)
            
    #         # Calculate the mean of these closest divergences
    #         mean_divergence = np.mean(closest_divergences)
    #         return mean_divergence

    #     # Process each cluster in parallel
    #     results = Parallel(n_jobs=-1, batch_size='auto')(
    #         delayed(process_cluster)(cluster_id, indices) for cluster_id, indices in tqdm(self.clusters.items())
    #     )

    #     # Filter out None results (clusters that were skipped)
    #     means_per_cluster = [result for result in results if result is not None]

    #     # Calculate the average of the means across all clusters
    #     average_mean_divergence = np.mean(means_per_cluster)
    #     return average_mean_divergence


    # def ratio_to_neighbors_kmeans(self, variation_dist, nb_clusters=3):
    #     self.nb_clusters=nb_clusters
    #     # Compute cluster distances
    #     cluster_dists = np.linalg.norm(self.kmeans.cluster_centers_ - variation_dist, axis=1)
    #     closest_clusters = np.argsort(cluster_dists)[:nb_clusters]
    #     closest_clusters

    #     # Search within the closest cluster
    #     closest_indices = []
    #     for cluster_idx in closest_clusters:
    #         closest_indices.extend(self.clusters[cluster_idx])
    #     closest_distributions = self.list_know_P[closest_indices]

    #     def compute_jsd(i):
    #         return Jensen_Shannon().JSDiv(P=closest_distributions[i].toarray().flatten(), Q=variation_dist)

    #     # Parallelize the JSD computation
    #     js_divergences = Parallel(n_jobs=-1, batch_size=int(closest_distributions.shape[0]/os.cpu_count()))(delayed(compute_jsd)(i) for i in range(closest_distributions.shape[0]))
    #     kmean_closest = heapq.nsmallest(self.N, js_divergences)
    #     mean100 = np.mean(kmean_closest)

    #     return kmean_closest, mean100

  
    # def ratio_to_neighbors(self, neighbor_dist, thr_diff=0.85):
    #     """
    #     Computes the ratio of nearest neighbors where the distance to new_Q exceeds neighbor_dist.
    #     """
    #     count_diff = 0
    #     num_known_P = self.list_know_P.shape[0]
    #     all_dists = []
    #     list_know_P=self.list_know_P
    #     new_Q = self.new_Q

    #     # Compute distances to all points
    #     for i in tqdm(range(num_known_P)):
    #         P_i = list_know_P[i].toarray().flatten()
    #         distance = Jensen_Shannon().JSDiv(P_i, new_Q)
    #         all_dists.append(distance)

    #     # Identify the closest N neighbors
    #     closests = heapq.nsmallest(self.N, all_dists)

    #     # Count neighbors with distances exceeding the threshold
    #     for dist in closests:
    #         if dist > neighbor_dist:
    #             count_diff += 1

    #     # Compute the proportion of neighbors exceeding the threshold
    #     difference = count_diff / len(closests)
    #     novel_diff = int(difference > thr_diff)

    #     return difference, novel_diff
    # def ratio_to_neighbors(self, neighbor_dist, thr_diff=0.85):
    #     """
    #     Computes the ratio of nearest neighbors where the distance to new_Q exceeds neighbor_dist.
    #     """
    #     count_diff = 0
    #     num_known_P = self.list_know_P.shape[0]
    #     all_dists = []
    #     list_know_P=self.list_know_P.tocsr()
    #     new_Q = self.new_Q

    #     # Compute distances to all points
    #     for i in tqdm(range(num_known_P)):
    #         P_i = list_know_P[i].toarray().flatten()
    #         all_dists.append(Jensen_Shannon().JSDiv(P_i, new_Q))
        
    #         # Identify the closest N neighbors
    #     closests = heapq.nsmallest(self.N, all_dists)

    #         # Count neighbors with distances exceeding the threshold
    #     for dist in closests:
    #         if dist > neighbor_dist:
    #             count_diff += 1

    #     # Compute the proportion of neighbors exceeding the threshold
    #     difference = count_diff / len(closests)
    #     novel_diff = int(difference > thr_diff)

    #     return difference, novel_diff
    
# class Difference():
#     """
#         We estimate the ratio of point that are in close vicinity of the point. 
#         list_know_P : represent the list of all distribution vectors for each individual documents
#     """
#     def __init__(self, list_know_P, new_Q, N=5):

#         self.list_know_P = list_know_P.tocsr()
#         self.new_Q = new_Q
#         self.JS = Jensen_Shannon()
#         self.N = N
#         #self.neighbor_dist = self.dist_estimate()
        
#     def dist_estimate(self):
#         """
#         Here we take the N closest neighbours of each points and we estimate the average distance to each points to its closests neighbors.
#         Then we return the average for the whole dataset to know what is the average distance a point is close to its neighbors

#             Stop at a sample of points -- prevent the code here to run forever???
#         """
#         avg_dists = []
#         for i in tqdm(range(self.list_know_P.shape[0])):
#             P_i = self.list_know_P[i]
#             #list_execpt = self.list_know_P[:i] + self.list_know_P[i+1:] ## We compare the dist to all elements except himself
#             list_execpt = np.delete(self.list_know_P, i, axis=0)
            
#             all_dists = []
#             for P_j in list_execpt:
#                 all_dists.append(self.JS.JSDiv(P_i, P_j))
            
#             if len(all_dists) > self.N:
#                 all_dists = heapq.nsmallest(self.N, all_dists)
            
#             avg_dist_i = sum(all_dists) / len(all_dists)
#             avg_dists.append(avg_dist_i)
            
#         avg_final = sum(avg_dists) / len(avg_dists)
    
#         return avg_final
    
#     def ratio_to_all(self, neighbor_dist, thr_diff=0.95):
#         count_diff = 0
#         for P_i in self.list_know_P:
#             distance = self.JS.JSDiv(P_i, self.new_Q)
#             if distance > neighbor_dist:
#                 count_diff += 1
        
#         #Proportion of points where the distance is superior to the average distance to normal neighbor -- the higher the more different
#         difference = count_diff / self.list_know_P.shape[0]
#         novel_diff = 0
#         if difference > thr_diff:
#             novel_diff = 1

#         return difference, novel_diff

#     def ratio_to_neighbors(self, neighbor_dist, thr_diff=0.85):
#         count_diff = 0
#         #We compute all distances to identify the closest neighbors
#         all_dists = []
#         for P_i in self.list_know_P:
#             distance = self.JS.JSDiv(P_i, self.new_Q)
#             all_dists.append(distance)
#         closests = heapq.nsmallest(self.N, all_dists)
#         #We check the proportion of neighbors that are closer that it should be on average
#         for dist in closests: 
#             if dist > neighbor_dist:
#                 count_diff += 1

#         #Proportion of neighbor points where the distance is superior to the average distance to normal neighbors -- the higher the more different
#         difference = count_diff / len(closests)
#         novel_diff = 0
#         if difference > thr_diff:
#             novel_diff = 1

#         return difference, novel_diff


    # def mean_js_divergence_across_clusters(self, iterations, nb_clusters=3, num_closest=100):
    #     """
    #     Computes the mean JS divergence across the closest clusters, starting from a random point.

    #     Parameters:
    #         iterations (int): Number of times to repeat the process.
    #         nb_clusters (int): Number of nearest clusters to consider (default is 3).
    #         num_closest (int): Number of closest points to consider for mean calculation (default is 100).
        
    #     Returns:
    #         float: The average of the means of JS divergences across the specified number of iterations.
    #     """
    #     def process_random_point():
    #         num_points = self.list_know_P.shape[0]

    #         # Select a random point from the entire dataset
    #         random_point_idx = random.randint(0, num_points - 1)
    #         random_point = self.list_know_P[random_point_idx].toarray().flatten()

    #         # Compute cluster distances from the random point
    #         cluster_dists = np.linalg.norm(self.kmeans.cluster_centers_ - random_point, axis=1)
    #         closest_clusters = np.argsort(cluster_dists)[:nb_clusters]

    #         # Search within the closest clusters
    #         closest_indices = []
    #         for cluster_idx in closest_clusters:
    #             closest_indices.extend(self.clusters[cluster_idx])
    #         closest_distributions = self.list_know_P[closest_indices]

    #         def compute_jsd(i):
    #             return Jensen_Shannon().JSDiv(closest_distributions[i].toarray().flatten(), random_point)

    #         # Compute JS divergence for all points in the closest clusters
    #         js_divergences = Parallel(n_jobs=-1, batch_size='auto')(
    #             delayed(compute_jsd)(i) for i in range(closest_distributions.shape[0])
    #         )

    #         # Get the num_closest smallest divergences
    #         closest_divergences = heapq.nsmallest(num_closest, js_divergences)

    #         # Calculate the mean of these closest divergences
    #         mean_divergence = np.mean(closest_divergences)
    #         return mean_divergence

    #     # Run the process for the specified number of iterations
    #     results = []
    #     for _ in tqdm(range(iterations)):
    #         results.append(process_random_point())

    #     # Calculate the average of the means
    #     average_mean_divergence = np.mean(results)
    #     return average_mean_divergence

    # def mean_js_divergence_across_clusters(self, iterations, num_closest=100):
    #     """
    #     Computes the mean Jensen-Shannon (JS) divergence across the closest clusters, starting from a random point.

    #     Parameters:
    #         iterations (int): Number of times to repeat the process.
    #         nb_clusters (int): Number of nearest clusters to consider (default is 3).
    #         num_closest (int): Number of closest points to consider for mean calculation (default is 100).

    #     Returns:
    #         float: The average of the means of JS divergences across the specified number of iterations.
    #     """
    #     def process_random_point():
    #         """
    #         Processes a single random point to compute the mean JS divergence
    #         across the specified number of closest clusters.
    #         """
    #         num_points = self.list_know_P.shape[0]

    #         # Select a random point from the entire dataset
    #         random_point_idx = random.randint(0, num_points - 1)
    #         random_point = self.list_know_P[random_point_idx].toarray().flatten()

    #         # Compute distances from the random point to cluster centers
    #         cluster_dists = np.linalg.norm(self.kmeans.cluster_centers_ - random_point, axis=1)
    #         closest_clusters = np.argsort(cluster_dists)[:self.nb_clusters]

    #         # Gather indices of distributions in the closest clusters
    #         closest_indices = []
    #         for cluster_idx in closest_clusters:
    #             closest_indices.extend(self.clusters[cluster_idx])
    #         closest_distributions = self.list_know_P[closest_indices]

    #         # Compute JS divergence for all distributions in the closest clusters
    #         def compute_jsd(i):
    #             P = closest_distributions[i].toarray().flatten()
    #             return Jensen_Shannon().JSDiv(P=P, Q=random_point)

    #         js_divergences = Parallel(n_jobs=-1, batch_size='auto')(
    #             delayed(compute_jsd)(i) for i in range(closest_distributions.shape[0])
    #         )

    #         # Find the smallest `num_closest` divergences and calculate their mean
    #         closest_divergences = heapq.nsmallest(num_closest, js_divergences)
    #         return np.mean(closest_divergences)

    #     # Run the process for the specified number of iterations
    #     results = []
    #     for _ in tqdm(range(iterations), desc="Computing mean JS divergences"):
    #         results.append(process_random_point())

    #     # Return the average mean divergence
    #     return np.mean(results)
    
    
# class Difference():
#     """
#         We estimate the ratio of point that are in close vicinity of the point. 
#         list_know_P : represent the list of all distribution vectors for each individual documents
#     """
#     def __init__(self, list_know_P, new_Q, N=5):

#         self.list_know_P = list_know_P.tocsr()
#         self.new_Q = new_Q
#         self.JS = Jensen_Shannon()
#         self.N = N
#         #self.neighbor_dist = self.dist_estimate()
        
#     def dist_estimate(self):
#         """
#         Here we take the N closest neighbours of each points and we estimate the average distance to each points to its closests neighbors.
#         Then we return the average for the whole dataset to know what is the average distance a point is close to its neighbors

#             Stop at a sample of points -- prevent the code here to run forever???
#         """
#         avg_dists = []
#         for i in tqdm(range(self.list_know_P.shape[0])):
#             P_i = self.list_know_P[i]
#             #list_execpt = self.list_know_P[:i] + self.list_know_P[i+1:] ## We compare the dist to all elements except himself
#             list_execpt = np.delete(self.list_know_P, i, axis=0)
            
#             all_dists = []
#             for P_j in list_execpt:
#                 all_dists.append(self.JS.JSDiv(P_i, P_j))
            
#             if len(all_dists) > self.N:
#                 all_dists = heapq.nsmallest(self.N, all_dists)
            
#             avg_dist_i = sum(all_dists) / len(all_dists)
#             avg_dists.append(avg_dist_i)
            
#         avg_final = sum(avg_dists) / len(avg_dists)
    
#         return avg_final
    
#     def ratio_to_all(self, neighbor_dist, thr_diff=0.95):
#         count_diff = 0
#         for P_i in self.list_know_P:
#             distance = self.JS.JSDiv(P_i, self.new_Q)
#             if distance > neighbor_dist:
#                 count_diff += 1
        
#         #Proportion of points where the distance is superior to the average distance to normal neighbor -- the higher the more different
#         difference = count_diff / self.list_know_P.shape[0]
#         novel_diff = 0
#         if difference > thr_diff:
#             novel_diff = 1

#         return difference, novel_diff

#     def ratio_to_neighbors(self, neighbor_dist, thr_diff=0.85):
#         count_diff = 0
#         #We compute all distances to identify the closest neighbors
#         all_dists = []
#         for P_i in self.list_know_P:
#             distance = self.JS.JSDiv(P_i, self.new_Q)
#             all_dists.append(distance)
#         closests = heapq.nsmallest(self.N, all_dists)
#         #We check the proportion of neighbors that are closer that it should be on average
#         for dist in closests: 
#             if dist > neighbor_dist:
#                 count_diff += 1

#         #Proportion of neighbor points where the distance is superior to the average distance to normal neighbors -- the higher the more different
#         difference = count_diff / len(closests)
#         novel_diff = 0
#         if difference > thr_diff:
#             novel_diff = 1

#         return difference, novel_diff