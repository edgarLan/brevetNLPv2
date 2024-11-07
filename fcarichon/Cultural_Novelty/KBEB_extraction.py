import pandas as pd
import ast
from collections import Counter
import re
import ast

class ID_linkages():
    """
    ### Knowledge Base 1 -- set of authentic recipe from the same country || Expectation Base 1 -- Combination of features
    ### Knowledge Base 2 -- set of recipe from the same countryc || Expectation Base 2 -- Combination of features
    """
    def __init__(self, df_names_): 

        self.df_names_ = df_names_
        self.list_recipes = list(df_names_['Recette'])
        self.list_countries = list(df_names_['Pays'])
        #self.df_KnowCountries = 

        ################ IMPORTANT NOTES
        # Niger is Nigeria | Netherlands Antille is Netherlands | Jewish is not Israel | Åland Islands is Sweden
    
    def ID_list_shortKB(self, df_KnowCountries_):
        """This function takes recipes -- i.e Spinash lasagna = lasagna or Spaghetti Carbonara = Carbonara and identify all items in authentic recipes from their country to constitutes the DB"""
        
        df_names_2 = self.df_names_
        df_names_2["ids_ShortList"] = ""
        for i, name in enumerate(self.list_recipes):
            name_list = name.split(' | ')
            country = self.list_countries[i]
            df_samecountry = df_KnowCountries_[df_KnowCountries_['countries'] == country]
            df_authrecipes = df_samecountry[df_samecountry['authenticity'] == "authentic"]
            list_title = list(df_authrecipes['title'])

            temp_ids = []
            for j, title in enumerate(list_title):
                title = title.lower()
                for name in name_list:
                    name = name.lower()
                    if name in title:
                        temp_ids.append(j)
            
            list_of_ids = list(df_authrecipes.iloc[temp_ids]["Unnamed: 0"])
            df_names_2.at[i, 'ids_shortlist'] = list_of_ids
            
        return df_names_2

    def ID_list_interKB(self, df_Allauth_):
        
        """
        Same function but with all authentic names recipes not just the ones from the same country
            Here a recipe from two different countries have the same base -- hypothesis is to consider that people are not able to make the differentiation because all recipes influence our perception
        """
        
        df_names_2 = self.df_names_
        df_names_2["ids_InterList"] = ""
        list_title = list(df_Allauth_['title'])
        
        for i, name in enumerate(self.list_recipes):
            name_list = name.split(' | ')
            
            temp_ids = []
            for j, title in enumerate(list_title):
                title = title.lower()
                for name in name_list:
                    name = name.lower()
                    if name in title:
                        temp_ids.append(j)
                        
            list_of_ids = list(df_Allauth_.iloc[temp_ids]["Unnamed: 0"])
            df_names_2.at[i, 'ids_interlist'] = list_of_ids
            
        return df_names_2

    def ID_list_longKB(self, df_KnowCountries_):
        """Same function but with all recipe of the same country with same name """
        df_names_2 = self.df_names_
        df_names_2["ids_LongList"] = ""

        for i, name in enumerate(self.list_recipes):
            name_list = name.split(' | ')
            country = self.list_countries[i]
            df_samecountry = df_KnowCountries_[df_KnowCountries_['countries'] == country]
            list_title = list(df_samecountry['title'])
            
            temp_ids = []
            for j, title in enumerate(list_title):
                title = title.lower()
                for name in name_list:
                    name = name.lower()
                    if name in title:
                        temp_ids.append(j)

            list_of_ids = list(df_samecountry.iloc[temp_ids]["Unnamed: 0"])
            df_names_2.at[i, 'ids_LongList'] = list_of_ids
        
        return df_names_2

    def train_variations(self, df_names_, df_train_):
    
        """
        Here we pick the variations froom the training data with a different country | We take the one from another country since all names in the same countries are the KB/EB
            -- just in case we need to complete
        """
        df_names_["train_variations"] = ""
        for i, name in enumerate(self.list_recipes):
            name_list = name.split(' | ')
            country = self.list_countries[i]

            #Here we study non local variqtions -- we pick all instances from different countries
            df_variations = df_train_[df_train_['countries'] != country]
            list_title = list(df_variations['title'])
            
            temp_ids = []
            for j, title in enumerate(list_title):
                title = title.lower()
                for name in name_list:
                    name = name.lower()
                    if name in title:
                        temp_ids.append(j)
                        
            list_of_ids = list(df_variations.iloc[temp_ids]["Unnamed: 0"])
            #Saving the final list of IDs
            df_names_.at[i, 'train_variations'] = list_of_ids

        return df_names_

    def test_variations(self, df_names_, df_variations, col_name='test_variations'):
    
        """
        Here we pick the variations froom the test or validation data -- we don't care about the country of origin -- here we want all variations since we know they are not in the KB
        """
        df_names_[col_name] = ""
        
        #Taking the list of all titles from the valid/test file
        
        list_title = list(df_variations['title'])
        for i, name in enumerate(self.list_recipes):
            name_list = name.split(' | ')
    
            temp_ids = []
            for j, title in enumerate(list_title):
                title = title.lower()
                for name in name_list:
                    name = name.lower()
                    if name in title:
                        temp_ids.append(j)
                        
            list_of_ids = list(df_variations.iloc[temp_ids]["Unnamed: 0"])
            #Saving the final list of IDs
            df_names_.at[i, col_name] = list_of_ids

        return df_names_