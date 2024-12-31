from collections import Counter
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.tokenize import word_tokenize
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict, defaultdict
import pandas as pd
import numpy as np
from nltk.probability import FreqDist
from tqdm import tqdm
import copy

def pmi(input_, w_size=3):
    """ Input : list of ORDERED feature variables (words for example) -- if feature order does not matter set window_size to inf. """
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(input_, window_size= w_size)
    return finder.score_ngrams(bigram_measures.pmi) #, finder.word_fd, finder.ngram_fd, total_words_temp




from collections import Counter
from nltk.util import ngrams
import math
from tqdm import tqdm

class OptimizedIncrementalPMI:
    def __init__(self, window_size=3, word_counts=Counter(), bigram_counts=Counter(), total_words=0, initial = True):
        if initial:
            self.window_size = window_size
            self.word_counts = Counter()  # Word frequency counts
            self.bigram_counts = Counter()  # Bigram counts
            self.total_words = 0  # Total number of words seen so far`
        else:
            self.window_size = window_size
            self.word_counts = word_counts  
            self.bigram_counts = bigram_counts
            self.total_words = total_words

    def update(self, input_):
        """
        Update bigram counts and word counts with new input data.

        Parameters:
        - input_: List of words (new text to process)
        """
        # Update word counts
        self.word_counts.update(input_)
        self.total_words += len(input_)

        # Generate bigrams for each window size
        bigrams = (
            (input_[i], input_[j])
            for i in range(len(input_))
            for j in range(i + 1, min(i + self.window_size, len(input_)))
        )
        self.bigram_counts.update(bigrams)

    def reset(self, input_):
        """
        Reset bigram counts and word counts with an input data, effectively removing its influence.

        Parameters:
        - input_: List of words (text to remove from the counts)
        """
        # Decrement word counts
        for word in input_:
            if self.word_counts[word] > 0:
                self.word_counts[word] -= 1
                self.total_words -= 1
                if self.word_counts[word] == 0:
                    del self.word_counts[word]  # Remove word if its count reaches zero
        
        # Decrement bigram counts
        bigrams = (
            (input_[i], input_[j])
            for i in range(len(input_))
            for j in range(i + 1, min(i + self.window_size, len(input_)))
        )
        for bigram in bigrams:
            if self.bigram_counts[bigram] > 0:
                self.bigram_counts[bigram] -= 1
                if self.bigram_counts[bigram] == 0:
                    del self.bigram_counts[bigram]  # Remove bigram if its count reaches zero

    def compute_pmi(self):
        """
        Compute PMI for bigrams with counts >= min_count.
        """
        pmi_scores = {}

        for (word1, word2), count in tqdm(self.bigram_counts.items()):
            pmi = math.log2(count/(self.word_counts[word1]*self.word_counts[word2])) + math.log2(self.total_words*1/(self.window_size - 1))
            pmi_scores[(word1, word2)] = [pmi, self.word_counts[word1], self.word_counts[word2], count]
    
        # pmi_scores = []
        # for (word1, word2), count in tqdm(self.bigram_counts.items()):
        #     pmi = math.log2(count / (self.word_counts[word1] * self.word_counts[word2])) + math.log2(self.total_words * 1 / (self.window_size - 1))
        #     pmi_scores.append(((word1, word2), pmi))
        word_counts_temp = Counter(self.word_counts)
        bigram_counts_temp = Counter(self.bigram_counts)
        total_words_temp = self.total_words
        return pmi_scores, word_counts_temp, bigram_counts_temp, total_words_temp



# class IncrementalPMI:
#     def __init__(self, window_size=3):
#         self.window_size = window_size
#         self.word_counts = defaultdict(int)  # Word frequency counts
#         self.bigram_counts = defaultdict(int)  # Bigram counts
#         self.total_words = 0  # Total number of words seen so far
#         self.bigram_measures = BigramAssocMeasures()  # PMI scoring
#         self.finder = None  # Placeholder for BigramCollocationFinder

#     def update(self, input_):
#         """
#         Update bigram counts and word counts with new input data.
        
#         Parameters:
#         - input_: List of words (new text to process)
#         """
#         # Update word counts and bigram counts with new input, considering the window size
#         for i in tqdm(range(len(input_))):
#             # Update word frequency count
#             self.word_counts[input_[i]] += 1
#             self.total_words += 1
            
#             # Create bigrams considering the window size
#             for j in range(i + 1, min(i + self.window_size, len(input_))):
#                 word1, word2 = input_[i], input_[j]
#                 self.bigram_counts[(word1, word2)] += 1
#         print("totalWords = ", self.total_words)
#         print(len(self.word_counts))
#         print(len(self.bigram_counts))


#     def compute_pmi(self):
#         """
#         Compute PMI for the bigrams using the accumulated counts.
        
#         Returns:
#         - List of PMI scores for the bigrams.
#         """
#         # Create a BigramCollocationFinder from word counts
#         self.finder = BigramCollocationFinder(FreqDist(self.word_counts), FreqDist(self.bigram_counts), window_size=self.window_size)  # Create finder from word list

#         # Score the bigrams with PMI
#         scored_bigrams = self.finder.score_ngrams(self.bigram_measures.pmi)
        
#         return scored_bigrams
    

def pmi_to_dict_adj(pmi_list):
        """ Take a PMI list of tuples in nltk format [((w1,w2),value)] and output a nested dictionary """
        nested_dict = {}
        w1 = list(OrderedDict.fromkeys(item[0][0] for item in pmi_list))
        w2 = list(OrderedDict.fromkeys(item[0][1] for item in pmi_list))
        
        for (key1, key2), value in pmi_list:
            if key1 not in nested_dict:
                nested_dict[key1] = {}
            nested_dict[key1][key2] = value
        nested_dict['w1'] = w1
        nested_dict['w2'] = w2
        
        return nested_dict

def pmi_to_dict_adj_dict(pmi_dict):
    """ Take a PMI dictionary in the format {('w1', 'w2'): value} and output a nested dictionary """
    nested_dict = {}
    w1 = list(OrderedDict.fromkeys(key[0] for key in pmi_dict.keys()))
    w2 = list(OrderedDict.fromkeys(key[1] for key in pmi_dict.keys()))
    
    for (key1, key2), value in pmi_dict.items():
        if key1 not in nested_dict:
            nested_dict[key1] = {}
        nested_dict[key1][key2] = value[0]
    nested_dict['w1'] = w1
    nested_dict['w2'] = w2
    
    return nested_dict

def pmi_to_dict_adj(pmi_dict):
    """ Take a PMI dictionary in the format {('w1', 'w2'): value} and output a nested dictionary """
    nested_dict = {}

    # Use set comprehension to get unique keys for w1 and w2
    w1 = {key[0] for key in pmi_dict.keys()}
    w2 = {key[1] for key in pmi_dict.keys()}

    for (key1, key2), value in pmi_dict.items():
        if key1 not in nested_dict:
            nested_dict[key1] = {}
        nested_dict[key1][key2] = value[0]

    nested_dict['w1'] = list(w1)
    nested_dict['w2'] = list(w2)

    return nested_dict



    
# def data_analysis(data_dict, ref=True, col_name='Train_Variations'):
    
#     recettes = []
#     if ref: 
#         Base_infos = data_dict["Reference_Base"]
#         indexes = [item for item in Base_infos.keys() if item != 'AllIngredients']
#     else:
#         Base_infos = data_dict[col_name]
#         indexes = list(Base_infos.keys())

#     for index in indexes:
#         recette = Base_infos[index]['recipe_clean']
#         recette = re.sub(r'\\u00b0', ' degree', recette)  # Use re.sub to replace the Unicode degree symbol with the word "degree"
#         recette = re.sub(r'(\d+)\\', r'\1 inch', recette) ## Replace any sequnce of Number// by Numberinch
#         recettes.append(recette)
    
#     return recettes, indexes

def docs_distribution(baseSpace, tE):
    """
    Computes probability distributions for documents and the entire corpus.
    
    Parameters:
    - baseSpace: DataFrame with multiple text columns (e.g., 'claims' and others).
    - tE: Object with `claims` attribute containing potential variations (Text Examples).
    
    Returns:
    - Prob_KB_matrix: Term probability distributions for documents in the Knowledge Base.
    - Corpus_dist: Overall term probability distribution in the Knowledge Base.
    - Count_matrix: Term count matrix for all documents (Knowledge Base + Text Examples).
    """
    # Combine baseSpace text with tE claims and handle NaN values
    KS_corpus = pd.concat([baseSpace, tE], axis=0).fillna("")

    # Vectorize the text to create the term-document matrix
    vectorizer = CountVectorizer()
    Count_matrix = vectorizer.fit_transform(KS_corpus)

    # Split term-document matrix into Knowledge Base and tE
    Old_matrix = Count_matrix[:len(baseSpace), :]  # Sparse matrix slicing

    # Compute term probability distributions for Knowledge Base documents
    row_sums = np.array(Old_matrix.sum(axis=1)).flatten()  # Ensure 1D array
    row_sums[row_sums == 0] = np.finfo(float).eps         # Replace zeros

    # Reshape row_sums to align with the sparse matrix's row structure
    row_sums_reshaped = row_sums[:, np.newaxis]  # Convert to column vector

    # Perform element-wise division safely
    Prob_KB_matrix = Old_matrix.multiply(1 / row_sums_reshaped)
    Prob_KB_matrix = Prob_KB_matrix.tocsr()  # Ensure CSR format

    # Compute overall term distribution in the Knowledge Base
    Count_overall = Old_matrix.sum(axis=0).A1  # Sum over all documents, convert to 1D array
    Corpus_dist = Count_overall / Count_overall.sum()

    return Prob_KB_matrix, Corpus_dist, Count_matrix


def new_distribution(Count_matrix, select_variation):
    """
    Computes updated corpus distribution and term distribution for the base + 1 tE patent
    
    Parameters:
    - Count_matrix: Sparse term-document count matrix (scipy.sparse.csr_matrix).
    - select_variation: Indices of rows to include in the new calculation.
    
    Returns:
    - updated_Corpus_dist: Overall term probability distribution (KB + 1 patent of tE).
    - Varations_dist: Term probability distribution for the last selected row (tE row).
    """
    # Extract the rows corresponding to the selected variations
    New_Count_matrix = Count_matrix[select_variation, :]
    # Compute row sums (1D array)
    row_sums = New_Count_matrix.sum(axis=1).A1  # `.A1` converts sparse matrix result to 1D numpy array
    # Calculate row-wise probabilities (element-wise multiplication for sparse matrices)
    Variation_matrix = New_Count_matrix.multiply(1 / row_sums[:, None])

    Varations_dist = Variation_matrix.getrow(-1).toarray().flatten()  # Keep as sparse
    # Compute overall term counts (sum along columns)
    New_Count_overall = New_Count_matrix.sum(axis=0).A1  # `.A1` for 1D array
    # Compute overall term distribution
    updated_Corpus_dist = New_Count_overall / New_Count_overall.sum()

    return updated_Corpus_dist, Varations_dist
    

def combine_columns(data, selected_columns):
    """
    Combines text from the selected columns into a single column.

    Parameters:
    - data: DataFrame containing text columns.
    - selected_columns: List of column names to combine. If None, all columns are combined.

    Returns:
    - Series: A single column with concatenated text from the selected columns.
    """
    if selected_columns is None:
        # Use all columns if none are specified
        selected_columns = data.columns
    
    # Check if all selected columns exist in the DataFrame
    for col in selected_columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")
    
    # Combine selected columns into a single column, row-wise
    combined_column = data[selected_columns].fillna('').agg(' '.join, axis=1)
    
    return combined_column


