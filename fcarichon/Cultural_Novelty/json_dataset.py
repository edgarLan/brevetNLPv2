import json
from tqdm import tqdm
import pandas as pd
from collections import Counter
import re
import ast
import spacy

class dataset_generation():

    def __init__(self, df_names_full, df_train, df_valid, df_test, lemma="True", authrorized_pos= ['PROPN', 'PRON', 'ADJ', 'ADV', 'NOUN', 'NUM', 'VERB']):
        self.nlp = spacy.load("en_core_web_sm")
        self.lemma = lemma
        self.authrorized_pos = authrorized_pos

        ##Datasets
        self.df_names_full = df_names_full
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test


        self.recipe_KB, self.ingr_KB = [], []
        self.recipe_train, self.country_train, self.ingr_train = [], [], []
        self.recipe_valid, self.country_valid, self.ingr_valid = [], [], []
        self.recipe_test, self.country_test, self.ingr_test = [], [], []
        self.recipe_KB_clean, self.recipe_train_clean, self.recipe_valid_clean, self.recipe_test_clean = [],[],[],[]
    
    def getting_lists(self, string_list):

        list_ = []
        for elem in string_list:
            list_.append(ast.literal_eval(elem))
    
        assert len(list_) == len(string_list)
        
        return list_

    def text_cleaning(self, recipe_list):

        clean_list = []
        for i in range(len(recipe_list)):
            recipe_doc = self.nlp(str(recipe_list[i]))
            temp_list = []
            for token in recipe_doc:
                if token.pos_ in self.authrorized_pos:
                    if not token.is_stop:
                        if self.lemma:
                            temp_list.append(token.lemma_)
                        else:
                            temp_list.append(token)
            clean_text = ' '.join(temp_list)
            clean_list.append(clean_text)
        return clean_list

    def recipe_gathering(self, id_list, df_, mode="var"):
        """
        Function to get the text from id_list
        """
        if mode == "KB": 
            recipe_list, ingr_list = [], []
            for id in id_list:
                df_id = df_[df_['Unnamed: 0'] == id]
                recipe_list.append(df_id['directions'].values[0])
                temp_list = list(ast.literal_eval(df_id['NER'].values[0]))
                ingr_list.extend(temp_list)
    
            ingr_list_f = list(set(ingr_list)) ## We just need to keep the unique list of all ingredient ever encountered for the KB
            
            return recipe_list, ingr_list_f
        else: 
            recipe_list, country_list, ingr_list = [], [], []
            for id in id_list:
                df_id = df_[df_['Unnamed: 0'] == id]
                recipe_list.append(df_id['directions'].values[0])
                country_list.append(df_id['countries'].values[0])
                ingr_list.append(df_id['NER'].values[0])  ## We append string here for now
    
            return recipe_list, country_list, ingr_list

    def get_ids(self):
        """
        Get list of lists of ids
        """
        string_KB = list(self.df_names_full['ids_LongList'])
        self.list_KB = self.getting_lists(string_KB)
        string_train_var = list(self.df_names_full['train_variations'])
        self.list_train_var = self.getting_lists(string_train_var)
        string_valid_var = list(self.df_names_full['valid_variations'])
        self.list_valid_var = self.getting_lists(string_valid_var)
        string_test_var = list(self.df_names_full['test_variations'])
        self.list_test_var = self.getting_lists(string_test_var)
    
    def get_elems(self):

        #we take the list of infos from the each Recette

        for i in tqdm(range(len(self.list_KB))):
            
            recipe_KB_, ingr_KB_ = self.recipe_gathering(self.list_KB[i], self.df_train, mode="KB")  ## The KB is formed with the trained dataset only
            self.recipe_KB.append(recipe_KB_)
            self.ingr_KB.append(ingr_KB_)
            recipe_train_, country_train_, ingr_train_ = self.recipe_gathering(self.list_train_var[i], self.df_train)
            self.recipe_train.append(recipe_train_)
            self.country_train.append(country_train_)
            self.ingr_train.append(ingr_train_)
            recipe_valid_, country_valid_, ingr_valid_ = self.recipe_gathering(self.list_valid_var[i], self.df_valid)
            self.recipe_valid.append(recipe_valid_)
            self.country_valid.append(country_valid_)
            self.ingr_valid.append(ingr_valid_)
            recipe_test_, country_test_, ingr_test_ = self.recipe_gathering(self.list_test_var[i], self.df_test)
            self.recipe_test.append(recipe_test_)
            self.country_test.append(country_test_)
            self.ingr_test.append(ingr_test_)

    def get_clean_recipe(self):


        """
        Input : list0 of N recipes. Each elem0 is a list1 composed of X matching recipes. Each elem 1 is a literal string with the description of the recipe
        """
        #assert len(self.recipe_KB) == len(self.list_KB)
        
        for i in tqdm(range(len(self.list_KB))):
            
            self.recipe_KB_clean.append(self.text_cleaning(self.recipe_KB[i]))
            self.recipe_train_clean.append(self.text_cleaning(self.recipe_train[i]))
            self.recipe_valid_clean.append(self.text_cleaning(self.recipe_valid[i]))
            self.recipe_test_clean.append(self.text_cleaning(self.recipe_test[i]))
        
    def json_create(self, file_path='.'):

        recipes_names = list(self.df_names_full['Recette'])
        recipes_country = list(self.df_names_full['Pays'])
        
        print('===========Starting Data Collection============')
        self.get_ids()
        print('=====IDs collected=====')
        self.get_elems()
        print('=====Recipes, countries and ingredients collected=====')
        self.get_clean_recipe()
        print('=====cleaning recipes done=====')
        count = 0
        print('============Starting saving procedure============')
        for i in tqdm(range(len(recipes_names))):
            name_save = recipes_names[i].split(' | ')[0]
            
            #Instanciating the dictionnary to save the information regarding this recipe
            dict_ = {}
            dict_['Recipe_Name'] = name_save
            dict_['Name variations'] = recipes_names[i]
            dict_['Country'] = recipes_country[i]
            dict_['Reference_Base'] = {}
            dict_['Reference_Base']['AllIngredients'] = self.ingr_KB[i]
            for j, id_ in enumerate(self.list_KB[i]):
                dict_['Reference_Base'][str(id_)]= {}
                dict_['Reference_Base'][str(id_)]['recipe_raw'] = self.recipe_KB[i][j]
                dict_['Reference_Base'][str(id_)]['recipe_clean'] = self.recipe_KB_clean[i][j]
            
            dict_['Train_Variations'] = {}
            for j, id_ in enumerate(self.list_train_var[i]):
                dict_['Train_Variations'][str(id_)]= {}
                dict_['Train_Variations'][str(id_)]['recipe_raw'] = self.recipe_train[i][j]
                dict_['Train_Variations'][str(id_)]['recipe_clean'] = self.recipe_train_clean[i][j]
                dict_['Train_Variations'][str(id_)]['country'] = self.country_train[i][j]
                dict_['Train_Variations'][str(id_)]['ingredient_list'] = self.ingr_train[i][j]
    
            dict_['Valid_Variations'] = {}
            for j, id_ in enumerate(self.list_valid_var[i]):
                dict_['Valid_Variations'][str(id_)]= {}
                dict_['Valid_Variations'][str(id_)]['recipe_raw'] = self.recipe_valid[i][j]
                dict_['Valid_Variations'][str(id_)]['recipe_clean'] = self.recipe_valid_clean[i][j]
                dict_['Valid_Variations'][str(id_)]['country'] = self.country_valid[i][j]
                dict_['Valid_Variations'][str(id_)]['ingredient_list'] = self.ingr_valid[i][j]
    
            dict_['Test_Variations'] = {}
            for j, id_ in enumerate(self.list_test_var[i]):
                dict_['Test_Variations'][str(id_)]= {}
                dict_['Test_Variations'][str(id_)]['recipe_raw'] = self.recipe_test[i][j]
                dict_['Test_Variations'][str(id_)]['recipe_clean'] = self.recipe_test_clean[i][j]
                dict_['Test_Variations'][str(id_)]['country'] = self.country_test[i][j]
                dict_['Test_Variations'][str(id_)]['ingredient_list'] = self.ingr_test[i][j]
    
            ## Saving the JSON for each recipe :
            file_name = file_path + name_save +'_recipe_{}.json'.format(i)
            with open(file_name, "w") as outfile:
                json.dump(dict_, outfile)
    
            del dict_
            count += 1

        print('============ALL {} FILES SAVED============'.format(count))