from collections import Counter
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.tokenize import word_tokenize
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict

def pmi(input_, w_size=3):
    """ Input : list of ORDERED feature variables (words for example) -- if feature order does not matter set window_size to inf. """
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(input_, window_size= w_size)
    return finder.score_ngrams(bigram_measures.pmi)

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
    
def data_analysis(data_dict, ref=True, col_name='Train_Variations'):
    
    recettes = []
    if ref: 
        Base_infos = data_dict["Reference_Base"]
        indexes = [item for item in Base_infos.keys() if item != 'AllIngredients']
    else:
        Base_infos = data_dict[col_name]
        indexes = list(Base_infos.keys())

    for index in indexes:
        recette = Base_infos[index]['recipe_clean']
        recette = re.sub(r'\\u00b0', ' degree', recette)  # Use re.sub to replace the Unicode degree symbol with the word "degree"
        recette = re.sub(r'(\d+)\\', r'\1 inch', recette) ## Replace any sequnce of Number// by Numberinch
        recettes.append(recette)
    
    return recettes, indexes

def docs_distribution(KB_recettes, variations):
    """Input : Cleaned text corpus - With the KB and ALL the potential variations
       Output : Probability distribution for each document in corpus and for the all corpus
    """
    text_corpus_with_new = KB_recettes + variations
    vectorizer = CountVectorizer()
    Count_KB = vectorizer.fit_transform(text_corpus_with_new)
    Count_matrix = Count_KB.toarray()
    Old_matrix = Count_matrix[:-len(variations), :]
    
    #Getting the term distribution for all documents in the all KB
    Prob_KB_matrix = Old_matrix/Old_matrix.sum(axis=1, keepdims=True)
    #Getting the overall term distribution in the all KB
    Count_overall = Old_matrix.sum(axis=0)
    Corpus_dist = Count_overall / Count_overall.sum()
    
    # Getting same info for updated matrix and the new document
    #Variation_matrix = Count_matrix/Count_matrix.sum(axis=1, keepdims=True)
    #Varations_dist = Variation_matrix[-1, :]
    #New_Count_overall = Count_matrix.sum(axis=0)
    #updated_Corpus_dist = New_Count_overall / New_Count_overall.sum()
    
    return Prob_KB_matrix, Corpus_dist, Count_matrix

def new_distribution(Count_matrix, select_variation):

    # Getting same info for updated matrix and the new document
    New_Count_matrix = Count_matrix[select_variation, :]
    Variation_matrix = New_Count_matrix/New_Count_matrix.sum(axis=1, keepdims=True)
    Varations_dist = Variation_matrix[-1, :]
    New_Count_overall = New_Count_matrix.sum(axis=0)
    updated_Corpus_dist = New_Count_overall / New_Count_overall.sum()

    return updated_Corpus_dist, Varations_dist
    

