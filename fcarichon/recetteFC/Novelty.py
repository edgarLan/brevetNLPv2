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

from Divergences import Jensen_Shannon

class Newness():
    """
    Estimate the ratio of new terms in the distribution and the ratio of disappearing terms
    known_P is the known distribution (Knowledge or Expectation Base) and new_Q should be the novel distribrution (Determine if the document is new or not)
    The two option are mathematically equivalent if you set equivalent threshold -- to choose based on your ease of interpretation
    """
    def __init__(self, known_P, new_Q, lambda_=0.8):

        self.known_P = known_P
        self.new_Q = new_Q
        self.lambda_ = lambda_
        
        JS = Jensen_Shannon()
        self.JSD_vector = JS.linear_JSD(known_P, new_Q)
        self.nb_elements = len(self.JSD_vector)

    def divergent_terms(self, thr_div=0.1):

        """
        JSD == 0 if and only if pi = qi, but we want to make sure the distribution gap between this two are large enough
        To interpret as if the new term make the divergence greater than threshold, then it is a significant cointributing, we just need to know in appearing or disappearing
        """
        for i in range(self.nb_elements):
            count_appear = 0
            count_disappear = 0
            if self.JSD_vector[i] > thr_div:
                if self.new_Q[i] > self.known_P[i]:
                    count_appear += 1
                if self.known_P[i] > self.new_Q[i]:
                    count_disappear += 1

        appear_ratio = count_appear / self.nb_elements
        disappear_ratio = count_disappear / self.nb_elements
        newness = self.lambda_ * appear_ratio + (1-self.lambda_) * disappear_ratio
        
        return newness, appear_ratio, disappear_ratio

    def probable_terms(self, thr_new=2):
        """
        To interpret as if the new term has a probability of appearing thr_new times greater than old doc, it is a new significant term. Reverse is true for disappearing
        """
        for i in range(self.nb_elements):
            count_appear = 0
            count_disappear = 0
            if self.JSD_vector[i] != 0:
                # To interpret as if the new term has a probability of appearing thr_new times greater than old doc, it is a new significant term. Reverse is true for disappearing
                if self.new_Q[i] / self.known_P[i] > thr_new:  
                    count_appear += 1
                if self.known_P[i] / self.new_Q[i]:
                    count_disappear += 1
        
        appear_ratio = count_appear / self.nb_elements
        disappear_ratio = count_disappear / self.nb_elements
        newness = self.lambda_ * appear_ratio + (1 - self.lambda_) * disappear_ratio
        
        return newness, appear_ratio, disappear_ratio


class Uniqueness():

    def __init__(self, known_P, new_Q):
        self.known_P = known_P
        self.new_Q = new_Q
        self.JS = Jensen_Shannon()
        
    def dist_to_proto(self, thr_uniqp=0.05):
        
        novel_uniq = 0
        uniqueness = self.JS.JSDiv(self.known_P, self.new_Q)
        if self.uniqueness >= thr_uniq:
            novel_uniq = 1
            
        return uniqueness, novel_uniq

    def proto_dist_shift(self, thr_uniqp=0.005):
        
        new_P = self.known_P + self.new_Q
        uniqueness = self.JS.JSDiv(self.known_P, new_P)
        if uniqueness >= thr_uniqp:
            novel_uniq = 1

        return uniqueness, novel_uniq

class Difference():
    """
        We estimate the ratio of point that are in close vicinity of the point. 
        list_know_P : represent the list of all distribution vectors for each individual documents
    """
    def __init__(self, list_know_P, new_Q, N=10):

        self.list_know_P = list_know_P
        self.new_Q = new_Q
        self.neighbor_dist = self.dist_estimate()
        self.N = N
        self.JS = Jensen_Shannon()

    def dist_estimate(self):
        """
        Here we take the N closest neighbours of each points and we estimate the average distance to each points to its closests neighbors.
        Then we return the average for the whole dataset to know what is the average distance a point is close to its neighbors
        """
        avg_dists = []
        for i in range(len(self.list_know_P)):
            P_i = self.list_know_P[i]
            list_execpt = self.list_know_P[:i] + self.list_know_P[i+1:] ## We compare the dist to all elements except himself
            
            all_dists = []
            for P_j in list_execpt:
                all_dists.append(self.JS.JSDiv(P_i, Pj))
            
            if len(all_dists) > self.N:
                all_dists = heapq.nlargest(self.N, all_dists)
            
            avg_dist_i = sum(all_dists) / len(all_dists)
            avg_dists.append(avg_dist_i)
            
        avg_final = sum(avg_dists) / len(avg_dists)
    
        return avg_final
    
    def ratio_to_all(self, thr_diff=0.95):
        count_diff = 0
        for P_i in self.list_know_P:
            distance = self.JS.JSD(P_i, new_Q)
            if distance >= neighbor_dist:
                count_diff += 1
        
        #Proportion of points where the distance is superior to the average distance to normal neighbor -- the higher the more different
        difference = count_diff / len(self.list_know_P)
        novel_diff = 0
        if difference > thr_diff:
            novel_diff = 1

        return difference, novel_diff

    def ratio_to_neighbors(self, thr_diff=0.85):
        count_diff = 0
        #We compute all distances to identify the closest neighbors
        all_dists = []
        for P_i in self.list_know_P:
            distance = self.JS.JSD(P_i, new_Q)
            all_dists.append(distance)
        closests = heapq.nlargest(self.N, all_dists)
        
        #We check the proportion of neighbors that are closer that it should be on average
        for dist in closests: 
            if dist >= neighbor_dist:
                count_diff += 1

        #Proportion of neighbor points where the distance is superior to the average distance to normal neighbors -- the higher the more different
        difference = count_diff / len(closests)
        novel_diff = 0
        if difference > thr_diff:
            novel_diff = 1

        return difference, novel_diff

        
        