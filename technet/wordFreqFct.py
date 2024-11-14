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

class Vocab():
    """
    Class to create Vocab
    Inputs:
        technet
        stopwords
    """
    
    def __init__(self, technet, df_lemm):
        self.technet = technet
        self.set_vocab = self.setVocab()
        self.dictionary = pd.Series(df_lemm['lemmatized'].values, index=df_lemm['technet_vocab']).to_dict()
        

    def setVocab(self):
        keep_vocab = list(self.technet['technet_vocab'])
        # keep_vocab = list(filter(lambda a: a != '-', keep_vocab))  # retirer  -
        # keep_vocab = list(filter(lambda a: a not in self.stopwords, keep_vocab)) # retirer mot dans nltk stopEN
        set_keep = set(keep_vocab)
        return set_keep
    
    def clean_tokens(self, text):
        doc_token = []
        for tok in nlp.tokenizer(str(text)):
            token_lower = tok.text.lower()
            if token_lower in self.set_vocab:
                doc_token.append(token_lower)
        filtered_text = ' '.join(doc_token)
        return filtered_text
    
    def cleanDF(self, df_text, type="all"):
        df_clean = pd.DataFrame()
        if type=="all":
            df_clean['background'] = df_text['background'].apply(self.clean_tokens)
            df_clean['claims'] = df_text['claims'].apply(self.clean_tokens)
            df_clean['abstract'] = df_text['abstract'].apply(self.clean_tokens)
            df_clean['summary'] = df_text['summary'].apply(self.clean_tokens)
        elif type=="claims":
            df_clean['claims'] = df_text['claims'].apply(self.clean_tokens)
        elif type=="others":
            df_clean['background'] = df_text['background'].apply(self.clean_tokens)
            df_clean['abstract'] = df_text['abstract'].apply(self.clean_tokens)
            df_clean['summary'] = df_text['summary'].apply(self.clean_tokens)


        return df_clean
    
    def lemmatize_with_dict(self, text):
        words = text.split()
        lemmatized_text = []

        for word in words:
            # Check if the word is in the dictionary and add it only if found
            lemmatized_word = self.dictionary.get(word)

            if isinstance(lemmatized_word, str):
                lemmatized_text.append(lemmatized_word)

        return ' '.join(lemmatized_text)
    
    def lemmDF(self, df_clean):
        lemm_df = pd.DataFrame()
        for column in df_clean.columns:
            lemm_df[column] = df_clean[column].apply(lambda x: self.lemmatize_with_dict(str(x)))
        return lemm_df

    
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

    
def df2dict(pathCSV, n=50):

    data_by_year_ipc = {}

    num_files = len([f for f in os.listdir(pathCSV) if os.path.isfile(os.path.join(pathCSV, f))])
    if n > num_files:
        n=num_files

    # Filter for files that contain "KS" in the filename
    ks_files = [f for f in os.listdir(pathCSV) if os.path.isfile(os.path.join(pathCSV, f)) and f.endswith('.csv')]

    # Get the count of available KS files
    num_files = len(ks_files)

    # Adjust n if it exceeds the number of available KS files
    if n > num_files:
        n = num_files

    # Randomly sample n files from the KS files
    sampled_files = random.sample(ks_files, n)

    # Get the full paths of the sampled files
    csv_files = [os.path.join(pathCSV, f) for f in sampled_files]

    # Loop through each file, extract year and IPC, and store DataFrame in the dictionary
    for file in csv_files:
        # Get the base name of the file (without directory)
        filename = os.path.basename(file)
        
        # Extract year and IPC code from the filename
        try:
            year, _, ipc = filename.split("_")[:3]
            
            # Load the CSV file as a DataFrame
            df = pd.read_csv(file)
            
            # Initialize the year dictionary if not already present
            if year not in data_by_year_ipc:
                data_by_year_ipc[year] = {}
            
            # Store the DataFrame in the appropriate year and IPC code
            data_by_year_ipc[year][ipc] = df

        except ValueError:
            print(f"Filename format unexpected: {filename}")
    return(data_by_year_ipc)

