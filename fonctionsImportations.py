import os
import pandas as pd
from os import listdir
from os.path import isfile, join
import json
import collections
import re
from tqdm import tqdm
import tarfile 
import time

def tar_gz2json(yearBeginning, yearEnd, basePath, destPath):
    for year in range(yearBeginning, yearEnd+1):  # 2019 is excluded
        # Construct the filename
        filename = f'{basePath}{year}.tar.gz'
        
        # Start time before extraction
        start_time = time.time()

        # Open the tar.gz file
        with tarfile.open(filename, 'r:gz') as file:
            # Extract all files
            file.extractall(destPath) #'C:/Users/edgar/OneDrive/Bureau/Ecole/HEC/A24/BrevetNLP/data'

        # Calculate the time taken for this iteration
        elapsed_time = time.time() - start_time

        print(f'Extracted: {filename} in {elapsed_time:.2f} seconds')


# Function returning T if years are valid and present in dataset
def checkYears(year, yearsNeeded, pathData):
    ret = False
    folder_names = [folder for folder in os.listdir(pathData) if os.path.isdir(os.path.join(pathData, folder))]
    folder_years = [int(name) for name in folder_names] 
    required_years = set(range(year - yearsNeeded, year))
    missing_years = required_years - set(folder_years)

    # Check for missing 2004 and 2005
    if {2004}.issubset(required_years | {year}) or {2005}.issubset(required_years | {year}):
        print(f"For eval year {year}: Missing IPC - 2004 and/or 2005")
    if {2017}.issubset(required_years | {year}) or {2018}.issubset(required_years | {year}):
        print(f"For eval year {year}: Incomplete data - 2017 and/or 2018")

    # General check for missing years
    if not missing_years:
        print(f"All {yearsNeeded} reference years for eval year {year} present")
    else:
        print(f"Missing years: {missing_years}")
    if not missing_years and not({2004}.issubset(required_years | {year}) or {2005}.issubset(required_years | {year})) and not({2017}.issubset(required_years | {year}) or {2018}.issubset(required_years | {year})):
        ret = True                                                                                                                      
    return ret

# Function taking year, IPC class, path to data, and path to output and writing CSV file toEval for current year and IPC, and also writing secondary IPC in /text/_.txt
# Needs a directory /test in output path
def json2toEval(year, listIPC, pathData, pathOutput):
    print(f"Create toEval, iterate through all patents of current year {year}")

    pathYear = pathData+ f"/{year}/"                                # Updates with variable year
    jsonNamesYear = [f for f in listdir(pathYear) if isfile(join(pathYear, f))] 

    # Initialize list for each IPC class
    patent_listIPC_dict = {ipc: [] for ipc in listIPC}  # A list to store patents for each IPC

    # Creates list of patents for each IPC class
    for i in tqdm(range(len(jsonNamesYear))):
        patent_path = pathYear + jsonNamesYear[i]
        with open(patent_path) as f:
            d = json.load(f)  # Load json in d
            # No need to explicitly close the file with "with open" syntax
            
        class_mainIPC = d['main_ipcr_label']
        
        # Check if the class matches any IPC in the listIPC
        for ipc in listIPC:
            if re.match(f'^{ipc}', class_mainIPC):
                patent_listIPC_dict[ipc].append(jsonNamesYear[i])
    
    final_patents_dict = {ipc: [] for ipc in listIPC}  # Create a final list for each IPC

    # Create list excluding all other than accepted and rejected
    for ipc in listIPC:
        # Access the patents for each IPC class
        patent_listIPC = patent_listIPC_dict[ipc]  
    
        for i in tqdm(range(len(patent_listIPC))):
            patent_path = pathYear + patent_listIPC[i]
            with open(patent_path) as f:
                d = json.load(f)
            
            # Check if the decision is either ACCEPTED or REJECTED
            if d['decision'] in ['ACCEPTED', 'REJECTED']: 
                final_patents_dict[ipc].append(patent_listIPC[i])  # Add to the corresponding IPC list

    good_expectations_classes_dict = {ipc: [] for ipc in listIPC}  # To store good expectations for each IPC

    # Load needed data for patents by IPC class
    data_by_ipc = {ipc: {
        'patent_number': [], 'titles': [], 'backgrounds': [], 'claims': [],
        'summary': [], 'abstract': [], 'main_ipc': [], 'sec_ipc': [], 'labels': []
    } for ipc in listIPC}

    for ipc in listIPC:
        non_main_ipc = []  
        for i in tqdm(range(len(final_patents_dict[ipc]))):
            patent_path = pathYear + final_patents_dict[ipc][i]
            with open(patent_path) as f:
                d = json.load(f)

            # Creating the lists for the other information
            data_by_ipc[ipc]['patent_number'].append(d['application_number'])
            data_by_ipc[ipc]['titles'].append(d['title'])
            data_by_ipc[ipc]['backgrounds'].append(d['background'])
            data_by_ipc[ipc]['claims'].append(d['claims'])
            data_by_ipc[ipc]['summary'].append(d['summary'])
            data_by_ipc[ipc]['abstract'].append(d['abstract'])
            data_by_ipc[ipc]['main_ipc'].append(d['main_ipcr_label'])
            data_by_ipc[ipc]['sec_ipc'].append(d['ipcr_labels'])

            # Getting labels based on decision
            label = 0
            if d['decision'] == 'ACCEPTED':
                label = 1
            data_by_ipc[ipc]['labels'].append(label)

            # Collect secondary IPC classes and filter out the main IPC class
            non_main_ipc.extend([ipcr for ipcr in d['ipcr_labels'] if ipcr != d['main_ipcr_label']])

        # Keep only unique secondary IPC classes
        expectations_classes = list(set(non_main_ipc))  # Unique secondary IPC classes
        good_expectations_classes_dict[ipc] = [ipcr for ipcr in expectations_classes if ipcr[:4] != ipc]  # Exclude main IPC classes

    # Prepare DataFrames and save to CSV files for each IPC class
    for ipc in listIPC:
        df = pd.DataFrame({
            'application_number': data_by_ipc[ipc]['patent_number'],
            'title': data_by_ipc[ipc]['titles'],
            'abstract': data_by_ipc[ipc]['abstract'],
            'claims': data_by_ipc[ipc]['claims'],
            'background': data_by_ipc[ipc]['backgrounds'],
            'summary': data_by_ipc[ipc]['summary'],
            'ipc': ipc,
            'sec_ipc': data_by_ipc[ipc]['sec_ipc'],
            'label': data_by_ipc[ipc]['labels']
        })
        df.to_csv(pathOutput + f'/toEval/{year}_{ipc}_patents_toEval.csv', index=False)

        # Save IPC in text format for each IPC class
        with open(pathOutput + f'/ES/text/{year}_{ipc}_expectation_IPC_class.txt', 'w') as fp:
            for item in good_expectations_classes_dict[ipc]:
                fp.write("%s\n" % item)
            print(f'text/{year}_{ipc} Done')
        print(f"Nb secondary IPC for {ipc}: ", len(good_expectations_classes_dict[ipc]))


# Function taking ips, year studied, year in which we are searching, and path for input and output, and outputs dfs for ES et KS for the year_yearRef
def json2_KS_ES(year, yearRef, ipc, pathData, pathOutput):

    pathYear = pathData+ f"/{yearRef}/"                                # Updates with varialbe year
    jsonNamesYear = [f for f in listdir(pathYear) if isfile(join(pathYear, f))]

    # import secondary ipc classes of year toEval
    expect_classes_ipc_yearRef = []
    with open(pathOutput + f'/ES/text/{year}_{ipc}_expectation_IPC_class.txt', 'r') as fp:
        for line in fp:
            x = line[:-1]
            expect_classes_ipc_yearRef.append(x)#[0:4])  ######

    
    # Initialize KS and ES for this IPC class (for toEval yearRef)
    KS_ipc = []
    ES_ipc = []
    print(f"Create KS, iterate through patents of IPC {ipc}, of reference year {yearRef} for evalYear {year}")
    for i in tqdm(range(len(jsonNamesYear))):
        patent_path = pathYear + jsonNamesYear[i]
        with open(patent_path) as f:
            d = json.load(f)
            f.close()
        
        class_mainIPC = d['main_ipcr_label']#[0:4] #######
        class_main = class_mainIPC[0:4]

        #We are collecting all documents related to the main class - we distinguish them later by date
        if class_main == ipc:
            KS_ipc.append(jsonNamesYear[i])
        
        #For the expectations states - we have one for each year since the class are not similar ???
        if class_mainIPC in expect_classes_ipc_yearRef:
            ES_ipc.append(jsonNamesYear[i])

    current_date = int(f"{year}"+"0101")

    #Create knowledge space per year in df
    patent_number, titles, backgrounds, claims, summary, abstract, main_ipc, labels, sec_ipc, yearRefVec = [], [], [], [], [], [], [], [], [], []

    print(f"Create ES, iterate through patents of secondary IPC {ipc}, of reference year {yearRef} for evalYear {year}")
    for i in tqdm(range(len(KS_ipc))):
        patent_path = pathYear + KS_ipc[i]
        with open(patent_path) as f:
            d = json.load(f)
            f.close()
        #Not taking patents that are not published yet
        if int(d['date_published']) < current_date:
            
            #Creating the lists for the other information
            patent_number.append(d['application_number'])
            titles.append(d['title'])
            backgrounds.append(d['background'])
            claims.append(d['claims'])
            summary.append(d['summary'])
            abstract.append(d['abstract'])
            main_ipc.append(d['main_ipcr_label'])
            labels.append(d['decision'])
            sec_ipc.append(d['ipcr_labels'])
            yearRefVec.append(yearRef)
        else:
            #If the date is superior, we still take accepted or rejected into account
            if d['decision'] == 'ACCEPTED' or d['decision'] == 'REJECTED':
                #Creating the lists for the other information
                patent_number.append(d['application_number'])
                titles.append(d['title'])
                backgrounds.append(d['background'])
                claims.append(d['claims'])
                summary.append(d['summary'])
                abstract.append(d['abstract'])
                main_ipc.append(d['main_ipcr_label'])
                labels.append(d['decision'])
                sec_ipc.append(d['ipcr_labels'])
                yearRefVec.append(yearRef)

    df_KS = pd.DataFrame({'application_number': patent_number, 'title': titles, 'abstract':abstract,
                        'claims':claims, 'background': backgrounds, 'summary':summary, 'ipc':main_ipc, 
                        'sec_ipc': sec_ipc, 'label': labels, 'yearRef': yearRefVec})
    
    
    #Create expectations space per year in df
    patent_number, titles, backgrounds, claims, summary, abstract, main_ipc, labels, sec_ipc, yearRefVec = [], [], [], [], [], [], [], [], [], []
    for i in tqdm(range(len(ES_ipc))):
        patent_path = pathYear + ES_ipc[i]
        with open(patent_path) as f:
            d = json.load(f)
            f.close()
        
        #Not taking patnts that are not published yet
        if int(d['date_published']) < current_date:
            
            #Creating the lists for the other information
            patent_number.append(d['application_number'])
            titles.append(d['title'])
            backgrounds.append(d['background'])
            claims.append(d['claims'])
            summary.append(d['summary'])
            abstract.append(d['abstract'])
            main_ipc.append(d['main_ipcr_label'])
            labels.append(d['decision'])
            sec_ipc.append(d['ipcr_labels'])
            yearRefVec.append(yearRef)
        else:
            #If the date is superior to 2016, we still take accepted or rejected into account ???
            if d['decision'] == 'ACCEPTED' or d['decision'] == 'REJECTED':
                #Creating the lists for the other information
                patent_number.append(d['application_number'])
                titles.append(d['title'])
                backgrounds.append(d['background'])
                claims.append(d['claims'])
                summary.append(d['summary'])
                abstract.append(d['abstract'])
                main_ipc.append(d['main_ipcr_label'])
                labels.append(d['decision'])
                sec_ipc.append(d['ipcr_labels'])
                yearRefVec.append(yearRef)

    df_ES = pd.DataFrame({'application_number': patent_number, 'title': titles, 'abstract':abstract,
                        'claims':claims, 'background': backgrounds, 'summary':summary, 'ipc': main_ipc, 
                        "sec_ipc": sec_ipc, 'label': labels, 'yearRef': yearRefVec})
    
    return (df_KS, df_ES)

# Function that simply loops over json2_KS_ES yearsNeeded times, and binds dataframes together. Writes a CSV for KS and ES.
def loop_KS_ES(year, yearsNeeded, ipc, pathData, pathOutput):

    required_years = set(range(year - yearsNeeded, year))
    df_KS = pd.DataFrame()
    df_ES = pd.DataFrame()
    for i in required_years:
        dfs_temp = json2_KS_ES(year, i, ipc, pathData, pathOutput)
        df_KS_temp = dfs_temp[0]
        df_ES_temp = dfs_temp[1]
        df_KS = pd.concat([df_KS, df_KS_temp], axis=0, ignore_index=True)
        df_ES = pd.concat([df_ES, df_ES_temp], axis=0, ignore_index=True)
        print("ES and KS done for: " + f"{year}_{i}_{ipc}")
    
    df_KS.to_csv(pathOutput + f'/KS/{year}_{str(list(required_years)[0])[2:4]}{str(list(required_years)[-1])[2:4]}_{ipc}_KS_raw.csv', index=False)
    df_ES.to_csv(pathOutput + f'/ES/{year}_{str(list(required_years)[0])[2:4]}{str(list(required_years)[-1])[2:4]}_{ipc}_ES_raw.csv', index=False)
    print("ES and KS done for: " + f"{year}_{ipc}")
    print("df_KS shape: ", df_KS.shape)
    print("df_ES shape: ", df_ES.shape)

# Function taking years to be evaluated, number of years as reference and a list of IPC classes.
def loopFinal(listIPC, listYearsEval, nbYearsRef, pathData, pathOutput):
    # check if valid years
    for year in listYearsEval:   
        cY = checkYears(year, nbYearsRef, pathData)
        if not cY:
            return
    # Loop through each ipc
    for ipc in tqdm(listIPC):
        # Loop through each year
        for year in tqdm(listYearsEval):
            json2toEval(year, ipc, pathData, pathOutput)
            loop_KS_ES(year, nbYearsRef, ipc, pathData, pathOutput)