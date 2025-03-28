import spacy
nlp = spacy.load("en_core_web_sm")
import pandas as pd
from tqdm import tqdm
import nltk
import re
from nltk.corpus import stopwords
from collections import Counter
nltk.download('stopwords')
import glob
import os
import random
import ast
tqdm.pandas(miniters=100000, maxinterval=1000)

class Vocab():
    """
    Class to create Vocab
    Inputs:
        technet
        stopwords
        df_lemm : lemmatized technet
    """
    
    def __init__(self, technet, df_lemm, stopwords):
        self.technet = technet
        self.set_vocab = self.setVocab()
        self.stopwords = stopwords
        self.tn_lemm_filtered = self.filterSW(df_lemm)
        self.lemm_dict = pd.Series(self.tn_lemm_filtered['lemmatized'].values, index=self.tn_lemm_filtered['technet_vocab']).to_dict()
        

    def setVocab(self):
        # Creates vocab set with technet vocab.
        keep_vocab = list(self.technet['technet_vocab'])
        # keep_vocab = list(filter(lambda a: a != '-', keep_vocab))  # retirer  -
        # keep_vocab = list(filter(lambda a: a not in self.stopwords, keep_vocab)) # retirer mot dans nltk stopEN
        set_keep = set(keep_vocab)
        return set_keep
    
    def clean_tokens(self, text):
        # cleans the text, keeping only words in set_vocab
        doc_token = []
        for tok in nlp.tokenizer(str(text)):
            token_lower = tok.text.lower()
            if token_lower in self.set_vocab:
                doc_token.append(token_lower)
        filtered_text = ' '.join(doc_token)
        return filtered_text
    
    def cleanDF(self, df_text, type="all"):
        # applies cleaning to all texts in patents.
        df_clean = pd.DataFrame()
        df_clean["application_number"] = df_text["application_number"]
        df_clean["label"] = df_text["label"]
        print("Clean")
        if type=="all":
            df_clean['background'] = df_text['background'].progress_apply(self.clean_tokens)
            df_clean['claims'] = df_text['claims'].progress_apply(self.clean_tokens)
            df_clean['abstract'] = df_text['abstract'].progress_apply(self.clean_tokens)
            df_clean['summary'] = df_text['summary'].progress_apply(self.clean_tokens)
        elif type=="claims":
            df_clean['claims'] = df_text['claims'].progress_apply(self.clean_tokens)
        elif type=="others":
            df_clean['background'] = df_text['background'].progress_apply(self.clean_tokens)
            df_clean['abstract'] = df_text['abstract'].progress_apply(self.clean_tokens)
            df_clean['summary'] = df_text['summary'].progress_apply(self.clean_tokens)


        return df_clean
    
    def lemmatize_with_dict(self, text):
        # lemmatize technet
        words = text.split()
        lemmatized_text = []

        for word in words:
            # Check if the word is in the dictionary and add it only if found
            lemmatized_word = self.lemm_dict.get(word)

            if isinstance(lemmatized_word, str):
                lemmatized_text.append(lemmatized_word)

        return ' '.join(lemmatized_text)
    
    def lemmDF(self, df_clean):
        # Lemmatize the technet cleaned df
        print("Lemmatize")
        lemm_df = df_clean
        for column in df_clean.columns:
            if column not in ["application_number", "label"]:
                lemm_df[column] = df_clean[column].progress_apply(lambda x: self.lemmatize_with_dict(str(x)))
        return lemm_df
    
    def filterSW(self, df_lemm):
        # filters out stopwords from the lemmatized technet dataframe
        tn_lemm = df_lemm[~df_lemm['lemmatized'].isin(self.stopwords)]
        return(tn_lemm)


def get_file_names(pathCSV):
    # Get the list of files in the directory
    all_files = [f for f in os.listdir(pathCSV) if os.path.isfile(os.path.join(pathCSV, f))]
    # Filter the files for those ending with '.csv'
    csv_files = [f for f in all_files if f.endswith('.csv')]
    return csv_files

def extract_year_ipc(filename):
    # Regular expression to match both formats
    match = re.match(r'(\d{4})(?:_\d{4})?_(\w{4})(?:_.*)?\.csv', filename)
    
    if match:
        year = match.group(1)
        ipc = match.group(2)
        return year, ipc
    else:
        print(f"Filename format unexpected: {filename}")
        return None, None

def extract_year_ipc_vs(filename):
    # Regular expression to capture year, ipc, and the part between ipc and Metrics
    match = re.match(r'(\d{4})_(\w{4})_([^_]+_.*)_Metrics\.csv', filename)
    
    if match:
        year = match.group(1)
        ipc = match.group(2)
        vs = match.group(3)
        return year, ipc, vs
    else:
        print(f"Filename format unexpected: {filename}")
        return None, None, None

def parse_stopwords(file_path):
    # Creates a dictionnary of lists of stopwords from .txt file

    # Open and read the file content
    with open(file_path, 'r') as file:
        content = file.read()

    # Define a regex pattern to match each list block
    pattern = r"(\w[\w\s]+):\s*(\[.*?\])"
    matches = re.findall(pattern, content, re.DOTALL)

    # Create a dictionary to store the lists
    stopword_dict = {}
    for label, list_str in matches:
        # Use ast.literal_eval to safely evaluate the string as a Python list
        stopword_dict[label.strip()] = ast.literal_eval(list_str)
    
    return stopword_dict

    

# On peut enlever
def count_word_frequency_patent(text):
    words = text.split()
    word_count = Counter(words)
    return word_count

def count_word_frequency_total(df_clean):
    # Initialize a Counter to hold the total word frequency across the entire column
    total_word_count = Counter()

    # Iterate through the rows of the selected columns and update the total word count
    for column in list(df_clean.columns):
        for text in df_clean[column]:
            total_word_count += count_word_frequency_patent(text)
    return total_word_count