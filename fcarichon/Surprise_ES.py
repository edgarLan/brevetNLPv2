import spacy
from scipy.spatial import distance
from scipy.special import softmax
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import math
from collections import Counter, defaultdict
import pandas as pd
from tqdm import tqdm
import dask
import dask.dataframe as dd

#https://stackoverflow.com/questions/58701337/how-to-construct-ppmi-matrix-from-a-text-corpus
class expectation():
    
    def __init__(self, df, window_size=2, mode=0):

        self.window_size = window_size
        self.df_expectation = df
        self.expectation_claims = list(df['claims'])
        self.expectation_back = list(df['background'])
        if mode == 0:
            self.expectation_texts = self.concat_texts(self.expectation_claims, self.expectation_back)
        elif mode == 1:
            self.expectation_texts = self.expectation_claims
        elif mode == 2:
            self.expectation_texts = self.expectation_back
        
        del df
        del self.df_expectation
        
        self.nlp = spacy.load("en_core_web_sm")
        self.log2 = math.log(2)
    
    def concat_texts(self, text_list1, text_list2):
        
        assert len(text_list1) == len(text_list2)
        concat_list = []
        for i in range(len(text_list1)):
            concat_list.append(str(text_list1[i]) + str(text_list2[i]))
        return concat_list
    
    def co_occurrence(self, save=False, path_name = './pmi_matrix.csv'):
        
        """This function takes a list of sentences and returns a pandas.DataFrame object representing the co-occurrence matrix and a window_size number:"""
        sentences = self.expectation_texts
        d = defaultdict(int)
        vocab = set()
        for i in tqdm(range(len(sentences))):
            # preprocessing (use tokenizer instead)
            text = sentences[i].lower().split()
            # iterate over sentences
            for i in range(len(text)):
                token = text[i]
                vocab.add(token)  # add to vocab
                next_token = text[i+1 : i+1+self.window_size]
                for t in next_token:
                    key = tuple(sorted([t, token]) )
                    d[key] += 1

        # formulate the dictionary into dataframe
        del sentences
        vocab = sorted(vocab) # sort vocab | np.int16
        print('prout')
        df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.uint8), index=vocab, columns=vocab)
        del vocab
        
        print('prout2')
        for key, value in d.items():
            df.at[key[0], key[1]] = value
            df.at[key[1], key[0]] = value
        print('prout3')
        del d
        #if save:
        print(df.info(verbose = False))
        df.to_csv(path_name, index=False)
       # return df
    
    def pmi(self, df, positive=True):
        
        col_totals = df.sum(axis=0)
        total = col_totals.sum()
        row_totals = df.sum(axis=1)
        expected = np.outer(row_totals, col_totals) / total
        df = df / expected
        # Silence distracting warnings about log(0):
        with np.errstate(divide='ignore'):
            df = np.log(df)
        df[np.isinf(df)] = 0.0  # log(0) = 0
        if positive:
            df[df < 0] = 0.0
        return df
    
    def saving_pmi(self, path_name = './pmi_matrix.csv'):
        print(len(self.expectation_texts))
        df_coocc = self.co_occurrence()
        print('done1')
        df_pmi = self.pmi(df_coocc)
        print('done2')
        df_pmi.to_csv(path_name, index=False)
    
    def surprise(self, text, loading_path='./pmi_matrix.csv', unk_score=0.75):
        
        #Loading the PMI for the expectation state:
        try:
            df_pmi_expect = pd.read_csv(loading_path)
        except:
            print("Error : Please enter correct path to PMI matrix or run saving_pmi first")
            
        #Estimating PMI for current text:
        df_coocc_current = self.co_occurrence([text])
        df_pmi_current = self.pmi(df_coocc_current)
        
        #Calculating the distance between PMI vectors:
        divergence = 0.
        distance_ = 0.
        expect_tokens = set(list((df_pmi_expect.columns)))
        current_tokens = set(list((df_pmi_current.columns)))
        common_tokens = set.intersection(expect_tokens, current_tokens)

        #Dealing with words that were not in the Expectation State
        new_tokens = current_tokens - common_tokens
        distance_ += unk_score * len(new_tokens)
        divergence += unk_score * len(new_tokens)
        
        #Incrementing divergence score for known tokens
        for token in common_tokens:
            context_tokens = common_tokens - set(list(token))
            #Select the columns with only common words to efficiently measure the divergence of existing collocations
            df_pmi_current_token = df_pmi_current[list(context_tokens)]
            df_pmi_expect_token = df_pmi_expect[list(context_tokens)]
            
            # Since matrix is symetric, we get the row_index by having the same colum indexed (row are not indexed compared to column in our matrix)
            current_row_index = df_pmi_current.columns.get_loc(token)
            expect_row_index = df_pmi_expect.columns.get_loc(token)
            
            #Getting the vector associated with row of the concern token
            current_token_vec = df_pmi_current_token.iloc[current_row_index].values.reshape(1, -1)
            expect_token_vec = df_pmi_expect_token.iloc[expect_row_index].values.reshape(1,-1)
            dist_score = euclidean_distances(current_token_vec, expect_token_vec) / math.sqrt(current_token_vec.shape[1])
            distance_ += dist_score
            
            current_dist = softmax(current_token_vec)
            expect_dist = softmax(expect_token_vec)
            print(current_dist.shape, expect_dist.shape)
            div_score = distance.jensenshannon(current_dist, expect_dist, axis=1)
            divergence += div_score
            
        #Normalize by length of documents
        weighted_dist = distance_ / len([token.text for token in self.nlp(text)])
        weighted_div = divergence / len([token.text for token in self.nlp(text)])
        
        return weighted_div, weighted_dist
    
    
    
    
class dask_pmi():
    
    def __init__(self, ddf, window_size=2, mode=0, partitions=6):

        self.window_size = window_size
        self.mode=mode
        self.expectation_claims = list(ddf['claims'].compute())
        self.expectation_back = list(ddf['background'].compute())
        self.partitions = partitions
        if mode == 0:
            self.expectation_texts = self.concat_texts(self.expectation_claims, self.expectation_back)
        elif mode == 1:
            self.expectation_texts = self.expectation_claims
        elif mode == 2:
            self.expectation_texts = self.expectation_back
        del ddf
        self.log2 = math.log(2)
    
    def concat_texts(self, text_list1, text_list2):
        
        assert len(text_list1) == len(text_list2)
        concat_list = []
        for i in range(len(text_list1)):
            concat_list.append(str(text_list1[i]) + str(text_list2[i]))
        return concat_list
    
    def dask_co_occurrence(self, save=False, path_name = './pmi_matrix.csv'):
        """This function takes a list of sentences and returns a pandas.DataFrame object representing the co-occurrence matrix and a window_size number:"""

        d = defaultdict(int)
        vocab = set()
        for i in tqdm(range(len(self.expectation_texts))):
            # preprocessing (use tokenizer instead)
            text = self.expectation_texts[i].lower().split()
            # iterate over sentences
            for i in range(len(text)):
                token = text[i]
                vocab.add(token)  # add to vocab
                next_token = text[i+1 : i+1+self.window_size]
                for t in next_token:
                    key = tuple(sorted([t, token]) )
                    d[key] += 1

        # formulate the dictionary into dataframe
        #del sentences
        vocab = sorted(vocab) # sort vocab | np.int16
        print('prout')
        df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.uint8), index=vocab, columns=vocab)
        del vocab
        
        print('prout2')
        for key, value in d.items():
            df.at[key[0], key[1]] = value
            df.at[key[1], key[0]] = value
        print('prout3')
        del d
        ddf_parts = dd.from_pandas(df, npartitions=self.partitions)
        del df
        ddf_parts.to_csv(path_name, index=False)
       # return df
    
    def pmi(self, ddf, positive=True, path_name = './pmi_matrix.csv'):
        
        """
        Input : Dask dataframe
        """
        #TEST 1 -- passing all to pandas and retransfering to Dask for saving:
        df = ddf.compute()
        col_totals = df.sum(axis=0)
        total = col_totals.sum()#.compute()
        row_totals = df.sum(axis=1)
        expected = np.outer(row_totals, col_totals) / total
        df = df / expected
        # Silence distracting warnings about log(0):
        with np.errstate(divide='ignore'):
            df = np.log(df)
        df[np.isinf(df)] = 0.0  # log(0) = 0
        if positive:
            df[df < 0] = 0.0
        
        df.to_csv(path_name, index=False)
        
        #Test 2 - computing with pandas but partitioning with dask
        #ddf_parts = dd.from_pandas(df, npartitions=self.partitions)
        #del df
        #ddf_parts.to_csv(path_name, index=False)
        
        #Test 3 - computing everything with dask : 
        #col_totals = ddf.sum(axis=0)
        #total = col_totals.sum().compute()
        #row_totals = ddf.sum(axis=1)
        #expected = np.outer(row_totals, col_totals) / total
        #ddf = ddf / expected
        #with np.errstate(divide='ignore'):
       #     ddf = np.log(ddf)
       # ddf[np.isinf(ddf)] = 0.0  # log(0) = 0
       # if positive:
        #    ddf[ddf < 0] = 0.0
        
        #ddf.to_csv(path_name, index=False)
        
        
        
class dask_pmi_2():
    
    def __init__(self, window_size=2, mode=0, partitions=6):

        self.window_size = window_size
        self.mode=mode
        self.log2 = math.log(2)
        self.partitions = partitions
        
    def concat_texts(self, text_list1, text_list2):
        
        assert len(text_list1) == len(text_list2)
        concat_list = []
        for i in range(len(text_list1)):
            concat_list.append(str(text_list1[i]) + str(text_list2[i]))
        return concat_list
    
    def dask_co_occurrence(self, ddf, path_name = './cooccurence_matrix.csv'):
        """This function takes a list of sentences and returns a pandas.DataFrame object representing the co-occurrence matrix and a window_size number:"""
        
        
        expectation_claims = list(ddf['claims'].compute())
        expectation_back = list(ddf['background'].compute())
        
        if self.mode == 0:
            expectation_texts = self.concat_texts(expectation_claims, expectation_back)
        elif self.mode == 1:
            expectation_texts = expectation_claims
        elif self.mode == 2:
            expectation_texts = expectation_back
        del ddf
        
        
        d = defaultdict(int)
        vocab = set()
        for i in tqdm(range(len(expectation_texts))):
            # preprocessing (use tokenizer instead)
            text = expectation_texts[i].lower().split()
            # iterate over sentences
            for i in range(len(text)):
                token = text[i]
                vocab.add(token)  # add to vocab
                next_token = text[i+1 : i+1+self.window_size]
                for t in next_token:
                    key = tuple(sorted([t, token]) )
                    d[key] += 1

        # formulate the dictionary into dataframe
        #del sentences
        vocab = sorted(vocab) # sort vocab | np.int16
        df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.uint8), index=vocab, columns=vocab)
        del vocab

        for key, value in d.items():
            df.at[key[0], key[1]] = value
            df.at[key[1], key[0]] = value

        del d
        ddf_parts = dd.from_pandas(df, npartitions=self.partitions)
        del df
        ddf_parts.to_csv(path_name, index=False)
    
    def pmi(self, ddf, positive=True, path_name = './pmi_matrix.csv'):
        
        """
        Input : Dask dataframe
        """
        #TEST 1 -- passing all to pandas and retransfering to Dask for saving:
        df = ddf.compute()
        col_totals = df.sum(axis=0)
        total = col_totals.sum()#.compute()
        row_totals = df.sum(axis=1)
        expected = np.outer(row_totals, col_totals) / total
        df = df / expected
        # Silence distracting warnings about log(0):
        with np.errstate(divide='ignore'):
            df = np.log(df)
        df[np.isinf(df)] = 0.0  # log(0) = 0
        if positive:
            df[df < 0] = 0.0
        
        df.to_csv(path_name, index=False)
        
        #Test 2 - computing with pandas but partitioning with dask
        #ddf_parts = dd.from_pandas(df, npartitions=self.partitions)
        #del df
        #ddf_parts.to_csv(path_name, index=False)
        
        #Test 3 - computing everything with dask : 
        #col_totals = ddf.sum(axis=0)
        #total = col_totals.sum().compute()
        #row_totals = ddf.sum(axis=1)
        #expected = np.outer(row_totals, col_totals) / total
        #ddf = ddf / expected
        #with np.errstate(divide='ignore'):
       #     ddf = np.log(ddf)
       # ddf[np.isinf(ddf)] = 0.0  # log(0) = 0
       # if positive:
        #    ddf[ddf < 0] = 0.0
        
        #ddf.to_csv(path_name, index=False)