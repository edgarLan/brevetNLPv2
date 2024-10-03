import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
from collections import Counter, defaultdict
import math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from tqdm import tqdm

class knowledge():
    
    def __init__(self, df, mode=0):
        
        self.nlp = spacy.load("en_core_web_sm")
        self.df_knowledge = df
        self.knowledge_claims = list(df['claims'])
        self.knowledge_back = list(df['background'])
        if mode == 0:
            self.knowledge_texts = self.concat_texts(self.knowledge_claims, self.knowledge_back)
        elif mode == 1:
            self.knowledge_texts = self.knowledge_claims
        elif mode == 2:
            self.knowledge_texts = self.knowledge_back
        
        del df
        
        self.tfidf_model = TfidfVectorizer()
        self.knowledge_matrix = self.tfidf_model.fit_transform(self.knowledge_texts)
        #print(np.mean(self.knowledge_matrix, axis=0))
        #self.dense_know_mat = self.knowledge_matrix.todense()
        #test = self.knowledge_matrix[0]
        #print(cosine_similarity(self.knowledge_matrix, test).shape, 1-cosine_similarity(self.knowledge_matrix, test))
        
    def concat_texts(self, text_list1, text_list2):
        
        assert len(text_list1) == len(text_list2)
        concat_list = []
        for i in range(len(text_list1)):
            concat_list.append(str(text_list1[i]) + str(text_list2[i]))
        return concat_list
    
    def new_tfidf(self, word_freq):
        
        #When we don't have a word in the list, we estimate the idf as New_IDF = log_e(# of documents +1 / 1)
        idf = math.log(len(self.knowledge_texts)+1/1)
        new_tfidf = word_freq * idf
        
        return new_tfidf
    
    def newness(self, text):
        """
        Input : One text to eval against all the knowledge space
        """
        
        #Getting the dictionary with the words already know in the matrx
        feature_names = self.tfidf_model.get_feature_names_out()
        tfidf_text_to_eval =  self.tfidf_model.transform([text]).todense()
        feature_index = tfidf_text_to_eval[0,:].nonzero()[1]
        tfidf_scores = zip([feature_names[i] for i in feature_index], [tfidf_text_to_eval[0, x] for x in feature_index])
        existing_words = dict(tfidf_scores)
        
        #Calculating an estimated score for words not present in the dictionnary:
        word_list = [token.text for token in self.nlp(text)]
        word_freq = Counter(word_list)
        
        #Calcuating the score for the sentence
        sent_score = 0.
        for word in word_list:
            if word in existing_words:
                sent_score += existing_words[word]
            else:
                sent_score += self.new_tfidf(word_freq[word])
                
        #Normalizing the score by the sentence lenght
        sent_score = sent_score / len(word_list)
        
        return sent_score
    
    def divergence(self, text):
        """
        Here we should mainly look at dist_min and dist_avg -- should have a huge average distance to everyone else + should not have a very close document -- dist_min
        """
        tfidf_text_to_eval = self.tfidf_model.transform([text])
        dist_score = 1 - abs(cosine_similarity(self.knowledge_matrix, tfidf_text_to_eval))
        
        dist_avg = np.mean(dist_score)
        dist_max = np.max(dist_score)
        dist_min = np.min(dist_score)
        
        return dist_avg, dist_max, dist_min
    
    def uniqueness(self, text):
        
        """
        Here we seek for the maximum distance of the point from the centroïd -- the bigger text_dist the better novelty + Idealy a positive delta or close to 0 which means extending or being close to
        the maximum existing distance.
        """
        
        tfidf_text_to_eval = np.asarray(self.tfidf_model.transform([text]).todense())
        dist_list = []
        avg_rep = np.asarray(np.mean(self.knowledge_matrix, axis=0))
        dist_score = 1 - abs(cosine_similarity(self.knowledge_matrix, avg_rep))
        
        #GEtting the document with the maximum distance from centroïd - estimating if the document "increase" the knowledge space or not
        dist_max = np.max(dist_score)
        text_dist = 1 - abs(cosine_similarity(tfidf_text_to_eval, avg_rep))
        delta = text_dist - dist_max
        
        return text_dist[0][0], delta[0][0]