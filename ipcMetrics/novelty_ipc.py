
import heapq
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
from divergences import Distance
from collections import Counter


class Newness():
    """
    Estimate the ratio of new IPC codes on known IPC codes in novel patents, list_new_Q (list of lists of IPCs).
    known_ipc is a list of all known unique IPC codes (Knowledge or Expectation Base).
    Keeping main IPC code in the computation because new main IPC code probably means novelty...
    """
    def __init__(self, known_ipc, list_new_Q):
        self.known_ipc = known_ipc
        self.list_new_Q = list_new_Q
    
    # def novelty_score_norm(self):
    #     score = []
    #     for patent in tqdm(range(0, len(self.list_new_Q))):
    #         novelty = 0
    #         for i in range(0, len(self.list_new_Q[patent])):
    #             if self.list_new_Q[patent][i] not in self.known_ipc:
    #                 novelty += 1
    #         if len(self.list_new_Q[patent])==0:
    #             score.append(novelty)
    #         else: score.append(novelty/len(self.list_new_Q[patent]))
    #     prop = len([x for x in score if x>0])/ len(score)
    #     return score, prop

    def novelty_score_norm(self):
        scores = []  # Use a more descriptive name for the result
        total_patents = len(self.list_new_Q)

        for patent in tqdm(range(total_patents)):
            novelty = sum(1 for i in self.list_new_Q[patent] if self.list_new_Q[patent][i] not in self.known_ipc)

            # Normalize the novelty score, handling the division by zero case
            normalized_novelty = novelty / len(self.list_new_Q[patent]) if self.list_new_Q[patent] else novelty
            scores.append(normalized_novelty)
        prop = len([x for x in scores if x>0])/ len(scores)
        return scores, prop
    

class Uniqueness():

    def __init__(self, list_known_P, list_new_Q):
        self.list_known_P = list_known_P
        self.list_new_Q = list_new_Q

        
    def dist_toProto(self, thr_uniqp=0.05):
        set_new_Q = [set(sublist) for sublist in self.list_new_Q]
        proto = list(Counter([tuple(sublist) for sublist in self.list_known_P]).most_common(1)[0][0])
        score = []
        for patent in set_new_Q:
            uniqueness = Distance.jaccard_dist(patent, proto)
            score.append(uniqueness)            
        return score

    # def proto_dist_shift(self, thr_uniqp=0.005):
        
    #     new_P = self.known_P + self.new_Q
    #     uniqueness = self.JS.JSDiv(self.known_P, new_P)
    #     if uniqueness >= thr_uniqp:
    #         novel_uniq = 1

    #     return uniqueness, novel_uniq



class Difference():
    """
        We estimate the ratio of point that are in close vicinity of the novel point. 
        list_know_P : List of listss of IPCs per patent (in knowledge base)
        list_new_Q : List of lists of new patents (in toEval)
    """
    def __init__(self, list_known_P, list_new_Q, N=10, nbKS=1000):

        self.list_known_P = list_known_P
        self.list_new_Q = list_new_Q
        self.N = N
        self.nbKS = nbKS
        self.dist_fct = Distance()      
        self.neighbor_dist = self.dist_estimate() # seuil


    def dist_estimate(self):
        """
        Here we take the N closest neighbours of each point and we estimate the average distance of each point to its closests neighbors.
        Then we return the average for the nbKS points from the dataset to know what is the average distance a point is close to its neighbors
        """
        mean_closest = []
        list_rand = (pd.Series(self.list_known_P).sample(n=self.nbKS, random_state=41)) ################# take out random state=41
        list_rand = [set(sublist) for sublist in list_rand]
        jaccard_dist_matrix = self.dist_fct.jacc_dist_matrix(list_rand, self.list_known_P)
        for i in range(self.nbKS):
            # Get the row and exclude the diagonal (self-comparison) - no need since not comparing with tiself
            row = jaccard_dist_matrix[i]
            closest_indices = np.argsort(row)[:self.N]   # Garder les n plus proche voisin 
            mean_value = np.mean(row[closest_indices]) # faire la moyenne des distances ces n pts
            mean_closest.append(mean_value)
        seuil = np.mean(mean_closest)
    
        return seuil
    
    
    def ratio_toAll(self, n=500, thr_diff=0.95):
        """
        For each new patent in list_new_Q, returns proportion of patents in list_known_P that are different different 
        (greater then neighbor_dist).
        n is the number of patents taken in the list_new_Q (to shorten time). If n=0, then all list_new_Q is taken.
        """
        if n==0:
            n=len(self.list_new_Q)
        ratios=[]
        set_new_Q = [set(sublist) for sublist in self.list_new_Q]
        set_known_P = [set(sublist) for sublist in self.list_known_P]
        random.seed(42)
        for Q_i in tqdm(random.sample(set_new_Q, n)):
            count_diff = 0
            for P_i in set_known_P:
                distance = self.dist_fct.jaccard_dist(P_i, Q_i)
                if distance >= self.neighbor_dist:
                    count_diff += 1
            ratios.append(count_diff/len(self.list_known_P))

        # #Proportion of points where the distance is superior to the average distance to normal neighbor -- the higher the more different
        #     difference = count_diff / len(self.list_known_P)
        #     novel_diff = 0
        #     if difference > thr_diff:
        #         novel_diff = 1

        return ratios


    def ratio_toAll_matrix(self, n=500, thr_diff=0.95):
        """
        For each new patent in list_new_Q, returns proportion of patents in list_known_P that are different different 
        (greater then neighbor_dist).
        n is the number of patents taken in the list_new_Q (to shorten time). If n=0, then all list_new_Q is taken, might have RAM problems with too big matrices.
        """
        if n==0:
            n=len(self.list_new_Q)
        ratios=[]
        random.seed(42)
        jaccard_dist_matrix = self.dist_fct.jacc_dist_matrix(random.sample(self.list_new_Q, n), self.list_known_P)
        for i in tqdm(range(n)):
            #Get the row and exclude the diagonal (self-comparison) - no need since not comparing with tiself
            row = jaccard_dist_matrix[i]
            length = len([value for value in row if value >= self.neighbor_dist])
            ratios.append(length/len(row))

        return ratios

    # def ratio_to_neighbors(self, thr_diff=0.85):
    #     count_diff = 0
    #     #We compute all distances to identify the closest neighbors
    #     all_dists = []
    #     for P_i in self.list_know_P:
    #         distance = self.JS.JSD(P_i, new_Q)
    #         all_dists.append(distance)
    #     closests = heapq.nlargest(self.N, all_dists)
        
    #     #We check the proportion of neighbors that are closer that it should be on average
    #     for dist in closests: 
    #         if dist >= neighbor_dist:
    #             count_diff += 1

    #     #Proportion of neighbor points where the distance is superior to the average distance to normal neighbors -- the higher the more different
    #     difference = count_diff / len(closests)
    #     novel_diff = 0
    #     if difference > thr_diff:
    #         novel_diff = 1

    #     return difference, novel_diff

        
        


