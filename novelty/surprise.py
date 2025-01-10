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
from collections import OrderedDict, defaultdict
import divergences
from divergences import Jensen_Shannon
from utils import pmi_to_dict_adj_dict
import importlib
importlib.reload(divergences)
from divergences import Jensen_Shannon
import numpy as np

class Surprise():

    def __init__(self, pmi_new):
        self.pmi_new = pmi_new
        self.JS = Jensen_Shannon()

    # def get_common_vectors(self, dict_old, dict_new, epsilon):
    #     """ Input : nested dictionaries for each PMI collocations
    #     Ouput : list of tuples vectors for each words """
    #     variables_1 = dict_old['variables']
    #     variables_2 = dict_new['variables']
    #     inter_list = list(set(variables_1 + variables_2))
    #     # print("inter_list: ", inter_list)
        
    #     vectors = {}
    #     for entry in inter_list:
    #         # print("entry: ", entry)
    #         vec_1 = [dict_old.get(entry, {}).get(key, epsilon) for key in inter_list]
    #         # print(f"    {vec_1},    {dict_old.get(entry, {})}")
    #         vec_2 = [dict_new.get(entry, {}).get(key, epsilon) for key in inter_list]
    #         # print(f"    {vec_2},    {dict_new.get(entry, {})}")
    #         vectors[entry] = (vec_1, vec_2)
    
    #     return vectors
    
    def get_common_vectors_adj(self, dict_old, dict_new, epsilon):
        """ Input : nested dictionaries for each PMI collocations
        Ouput : list of tuples vectors for each words """
        w1_old = dict_old['w1']
        w1_new = dict_new['w1']
        w2_old = dict_old['w2']
        w2_new = dict_new['w2']
        # interList_w1 = list(set(w1_old + w1_new))
        # interList_w2 = list(set(w2_old + w2_new))

        # interList_w1 = [word for word in interList_w1 if word not in ['w1', 'w2']]
        # interList_w2 = [word for word in interList_w2 if word not in ['w1', 'w2']]

        interList_w1 = list(set(w1_old) & set(w1_new) - {'w1', 'w2'})
        interList_w2 = list(set(w2_old) & set(w2_new) - {'w1', 'w2'})

        vectors = {}
        # for entry in tqdm(interList_w1):
        #     vec_1 = [dict_old.get(entry, {}).get(key, epsilon) for key in interList_w2]
        #     vec_2 = [dict_new.get(entry, {}).get(key, epsilon) for key in interList_w2]
        #     vectors[entry] = (vec_1, vec_2)

        for entry in (interList_w1):
            # Fetch the vectors from dict_old and dict_new for the current entry
            vec_1_dict = dict_old.get(entry, {})
            vec_2_dict = dict_new.get(entry, {})

            # Use list comprehension to extract the values for interList_w2 keys, 
            # defaulting to epsilon if the key is not found
            vec_1 = [vec_1_dict.get(key, epsilon) for key in interList_w2]
            vec_2 = [vec_2_dict.get(key, epsilon) for key in interList_w2]

            vectors[entry] = (vec_1, vec_2)

        return vectors
    
    def new_surprise(self, pmi_known, thr_surp=0.0104):
        """ On compare la distribution avec en sans new_Q -- on compare l'apparition de nouvelles collocations -- PMI augmente drastiquement selon un threshold
        By setting threshold to 0, as long as there is a new tuple we will consider it """
        
        # Find tuples in list_1 but not in list_2 and exceed the threshold
        # temp_known = [t[0] for t in tqdm(pmi_known)] ##  Useful to have only list of tuple not associated with their probbilities

        # unique_tuples = [t for t in tqdm(self.pmi_new) if t[0] not in temp_known and t[1] > 0.]
        n_unique_tuples = len(self.pmi_new) - len(pmi_known)
        # print(len(unique_tuples))
        # print(n_unique_tuples)
        # print(unique_tuples)
        # count_unique = len(unique_tuples)
        surprise_rate = n_unique_tuples / len(self.pmi_new)
        
        new_suprise = 0
        if surprise_rate > thr_surp:
            new_suprise = 1
        
        return surprise_rate, new_suprise

    # def uniq_surprise(self, dict_known, eps= 0.000001, thr_surp=0.):
    #     """ On compare la distribution avec en sans new_Q -- on compare la divergence JSD moyenne de ces deux distributions"""
    #     dict_new = pmi_to_dict_adj(self.pmi_new)
    #     vecotr_tuples = self.get_common_vectors(dict_known, dict_new, epsilon = eps)
        
    #     key_list = vecotr_tuples.keys()
    #     surprise_dists = []
    #     print("HERE")
    #     for entry in tqdm(key_list):
    #         tuple_known = vecotr_tuples[entry][0]
    #         tuple_new = vecotr_tuples[entry][1]
            
    #         #### We want only positive PMI score -- no negative values for not going nan or inf values
    #         tuple_known = [max(0., val) for val in tuple_known]
    #         tuple_new = [max(0., val) for val in tuple_new]   
    #         if sum(tuple_known) != 0 and sum(tuple_new) != 0:   #### We can't compare to non existing vectors neither
    #             surprise_dists.append(self.JS.JSDiv(tuple_known, tuple_new))
    
    #     surprise_score = sum(surprise_dists) / len(surprise_dists)
    #     dist_surprise = 0
    #     if surprise_score > thr_surp:
    #         dist_surprise = 1
            
    #     return surprise_score, dist_surprise
    
    def uniq_surprise_adj(self, dict_known, eps= 0.000001, thr_surp=0.):
        """ On compare la distribution avec en sans new_Q -- on compare la divergence JSD moyenne de ces deux distributions"""
        dict_new = pmi_to_dict_adj_dict(self.pmi_new)
        print("get_common_vectors")
        vecotr_tuples = self.get_common_vectors_adj(dict_known, dict_new, epsilon = eps)
        
        key_list = vecotr_tuples.keys()
        surprise_dists = []
        # print(key_list)
        print("JSDiv")
        #i=0
        for entry in (key_list):
            tuple_known, tuple_new = vecotr_tuples[entry]
            
            #### We want only positive PMI score -- no negative values for not going nan or inf values  
            tuple_known = np.maximum(0., np.array(tuple_known))
            # if i==4:
            #     print(list(tuple_known))
            #     print(sum(tuple_known))
            #     print("FINITO")
            #     print(tuple_new)
            #     print(sum(tuple_new))
            # i+=1
            tuple_new = np.maximum(0., np.array(tuple_new))
            mask = (np.array(tuple_new) != 0).astype(int)
            # Apply the mask to tuple_known
            tuple_known = tuple_known * mask
    
            if tuple_known.sum() and tuple_new.sum():
                surprise_dists.append(Jensen_Shannon().JSDiv(tuple_known, tuple_new))
            else: surprise_dists.append(0)

        surprise_score = sum(surprise_dists) / len(surprise_dists)
        # print(sum(surprise_dists))
        dist_surprise = 0
        if surprise_score > thr_surp:
            dist_surprise = 1
            
        return surprise_score, dist_surprise
    
    def unique_surp_courte(self, newpmi_PMI, known_pmi, base_bigram_set,  eps= 0, thr_surp=0.):
        update_bigram_set = set(newpmi_PMI.keys())
        common_bigram_set = update_bigram_set & base_bigram_set

        dict_new = pmi_to_dict_adj_dict({key: newpmi_PMI[key] for key in common_bigram_set})
        dict_known = pmi_to_dict_adj_dict({key: known_pmi[key] for key in common_bigram_set})
        vecotr_tuples = self.get_common_vectors_adj(dict_known, dict_new, epsilon = eps)

        key_list = vecotr_tuples.keys()
        surprise_dists = []
        for entry in (key_list):
            tuple_known, tuple_new = vecotr_tuples[entry]
            
            #### We want only positive PMI score -- no negative values for not going nan or inf values  
            mask = (np.array(tuple_known) != 0).astype(int)
            tuple_known = np.maximum(0., np.array(tuple_known))
            # Apply the mask to tuple_known
            tuple_new = tuple_new * mask

            mask = (np.array(tuple_new) != 0).astype(int)
            tuple_new = np.maximum(0., np.array(tuple_new))
            # Apply the mask to tuple_known
            tuple_known = tuple_known * mask

            if tuple_known.sum() and tuple_new.sum():
                surprise_dists.append(Jensen_Shannon().JSDiv(tuple_known, tuple_new))
            else: surprise_dists.append(0)
        surprise_score = sum(surprise_dists) / len(surprise_dists)
        if surprise_score > thr_surp:
            dist_surprise = 1
            
        return surprise_score, dist_surprise