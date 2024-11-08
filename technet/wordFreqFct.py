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

class Vocab():
    """
    Class to create Vocab
    Inputs:
        technet
        stopwords
    """
    
    def __init__(self, technet, stopwords):
        self.technet = technet
        self.stopwords = stopwords
        self.set_vocab = self.setVocab()
        

    def setVocab(self):
        keep_vocab = list(self.technet['technet_vocab'])
        keep_vocab = list(filter(lambda a: a != '-', keep_vocab))  # retirer  -
        keep_vocab = list(filter(lambda a: a not in self.stopwords, keep_vocab)) # retirer mot dans nltk stopEN
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
    
    def cleanDF(self, df_full):
        df_clean = pd.DataFrame()
        df_clean['background'] = df_full['background'].apply(self.clean_tokens)
        df_clean['claims'] = df_full['claims'].apply(self.clean_tokens)
        df_clean['abstract'] = df_full['abstract'].apply(self.clean_tokens)
        df_clean['summary'] = df_full['summary'].apply(self.clean_tokens)
        return df_clean
    
    def count_word_frequency_patent(self, text):
        words = text.split()
        word_count = Counter(words)
        return word_count

    def count_word_frequency_total(self, df_clean):
        # Initialize a Counter to hold the total word frequency across the entire column
        total_word_count = Counter()

        # Iterate through the rows of the selected columns and update the total word count
        for column in list(df_clean.columns):
            for text in df_clean[column]:
                total_word_count += self.count_word_frequency_patent(text)
        return total_word_count

    
def df2dict(pathCSV):
    # Directory containing the CSV files
    directory = pathCSV
    data_by_year_ipc = {}
    # Get all CSV file paths

    csv_files = glob.glob(os.path.join(directory, "*.csv"))

    # Loop through each file, extract year and IPC, and store DataFrame in the dictionary
    for file in csv_files:
        # Get the base name of the file (without directory)
        filename = os.path.basename(file)
        
        # Extract year and IPC code from the filename
        try:
            year, ipc, _ = filename.split("_")[:3]
            
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