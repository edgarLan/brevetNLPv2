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

class Surprise():

    def __init__(self, pmi_known, pmi_new):
        self.pmi_known = pmi_known
        self.pmi_new = pmi_new
        self.JS = Jensen_Shannon()

    def pmi_to_dict(pmi_list):
        """ Take a PMI list of tuples in nltk format [((w1,w2),value)] and output a nested dictionary """
        nested_dict = {}
        column_names = list(OrderedDict.fromkeys(item[0][1] for item in pmi_list))
        
        for (key1, key2), value in pmi_list:
            if key1 not in nested_dict:
                nested_dict[key1] = {}
            nested_dict[key1][key2] = value
        nested_dict['variables'] = column_names
        
        return nested_dict

    def get_common_vectors(dict_old, dict_new, epsilon):
        """ Input : nested dictionaries for each PMI collocations
        Ouput : list of tuples vectors for each words """
        dict_old['variables']
        dict_new['variables']
        inter_list = list(set(variables_1 + variables_2))

        vectors = {}
        for entry in inter_list:
            vec_1 = [dict_old.get(entry, {}).get(key, epsilon) for key in inter_list]
            vec_2 = [dict_new.get(entry, {}).get(key, epsilon) for key in inter_list]
            vectors[entry] = (vec_1, vec_2)
    
        return vectors
    
    def new_surprise(self, thr_surp=0.):
        """ On compare la distribution avec en sans new_Q -- on compare l'apparition de nouvelles collocations -- PMI augmente drastiquement selon un threshold
        By setting threshold to 0, as long as there is a new tuple we will consider it """
        
        # Find tuples in list_1 but not in list_2 and exceed the threshold
        unique_tuples = [t for t in self.pmi_new if t not in self.pmi_known and t[1] > thr_surp]
        count_unique = len(unique_tuples)
        surprise_rate = count_unique / len(self.pmi_new)
        
        new_suprise = 0
        if count_unique > 0:
            new_suprise = 1
        
        return surprise_rate, new_suprise

    def uniq_surprise(self, eps= 0.000001, thr_surp=0.):
        """ On compare la distribution avec en sans new_Q -- on compare la divergence JSD moyenne de ces deux distributions"""
        dict_known = self.pmi_to_dict(self.pmi_known)
        dict_new = self.pmi_to_dict(self.pmi_new)
    
        vecotr_tuples = self.get_common_vectors(dict_known, dict_new, epsilon = eps)
        key_list = vecotr_tuples.keys()
        surprise_dists = []
        for entry in key_list:
            tuple_known = vecotr_tuples[entry][0]
            tuple_new = vecotr_tuples[entry][1]
            surprise_dists.append(self.JS.JSD(tuple_known, tuple_new))
    
        surprise = sum(surprise_dists) / len(surprise_dists)
        dist_surprise = 0
        if surprise_score > thr_surp:
            dist_surprise = 1
            
        return surprise, dist_surprise