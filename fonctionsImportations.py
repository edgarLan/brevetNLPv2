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
# Needs a directory /test in output ES path
def json2toEval(year, listIPC, pathData, pathOutput, batch_size=1):
    print(f"Create toEval, iterate through all patents of current year {year}")

    pathYear = pathData + f"/{year}/"  # Updates with variable year
    jsonNamesYear = [f for f in listdir(pathYear) if isfile(join(pathYear, f))] 

    # Initialize list for each IPC class
    good_expectations_classes_dict = {ipc: [] for ipc in listIPC}  # To store good expectations for each IPC

    # Load needed data for patents by IPC class
    data_by_ipc = {ipc: {
        'patent_number': [], 'titles': [], 'backgrounds': [], 'claims': [],
        'summary': [], 'abstract': [], 'main_ipc': [], 'sec_ipc': [], 'labels': []
    } for ipc in listIPC}
    non_main_ipc = {ipc: [] for ipc in listIPC}

    # Total number of JSON files
    total_files = len(jsonNamesYear)

    # Creates a list of patents for each IPC class with batch-size tqdm
    with tqdm(total=total_files, desc=f"toEval - Processing patents - {year}") as pbar:
        for i in range(0, total_files, batch_size):
            for j in range(i, min(i + batch_size, total_files)):  # Process in batches
                patent_path = pathYear + jsonNamesYear[j]
                with open(patent_path) as f:
                    d = json.load(f)  # Load JSON in d

                class_mainIPC = d['main_ipcr_label']

                # Check if the class matches any IPC in the listIPC
                for ipc in listIPC:
                    if re.match(f'^{ipc}', class_mainIPC):
                        if d['decision'] in ['ACCEPTED', 'REJECTED']:  # Only for accepted and rejected
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
                            non_main_ipc[ipc].extend([ipcr for ipcr in d['ipcr_labels'] if ipcr != d['main_ipcr_label']])

            # Update the progress bar after processing each batch
            pbar.update(min(batch_size, total_files - i))

    for ipc in listIPC:
        expectations_classes = list(set(non_main_ipc[ipc]))  # Keep only unique secondary IPC classes
        good_expectations_classes_dict[ipc] = [ipcr for ipcr in expectations_classes if ipcr[:4] != ipc]  # Exclude main IPC classes
    
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
        print(f"toEval done for {ipc} in {year} ")
        print("toEval shape: ", df.shape)

        # Save IPC in text format for each IPC class
        with open(pathOutput + f'/ES/text/{year}_{ipc}_expectation_IPC_class.txt', 'w') as fp:
            for item in good_expectations_classes_dict[ipc]:
                fp.write("%s\n" % item)
            print(f"Number of secondary IPC for {ipc} in {year}: ", len(good_expectations_classes_dict[ipc]))



# Function taking list of IPC, year studied and one year reference and returns dfs for all ES and KS for that year_yearRef for all IPCs.
def json2_KS_ES(year, yearRef, listIPC, pathData, pathOutput, batch_size=1):
    pathYear = pathData + f"/{yearRef}/"  # Updates with variable year
    jsonNamesYear = [f for f in listdir(pathYear) if isfile(join(pathYear, f))]

    patent_number, titles, backgrounds, claims, summary, abstract, main_ipc, labels, sec_ipc, yearRefVec = {ipc: [] for ipc in listIPC}, {ipc: [] for ipc in listIPC}, {ipc: [] for ipc in listIPC}, {ipc: [] for ipc in listIPC}, {ipc: [] for ipc in listIPC}, {ipc: [] for ipc in listIPC}, {ipc: [] for ipc in listIPC}, {ipc: [] for ipc in listIPC}, {ipc: [] for ipc in listIPC}, {ipc: [] for ipc in listIPC}
    patent_numberE, titlesE, backgroundsE, claimsE, summaryE, abstractE, main_ipcE, labelsE, sec_ipcE, yearRefVecE = {ipc: [] for ipc in listIPC}, {ipc: [] for ipc in listIPC}, {ipc: [] for ipc in listIPC}, {ipc: [] for ipc in listIPC}, {ipc: [] for ipc in listIPC}, {ipc: [] for ipc in listIPC}, {ipc: [] for ipc in listIPC}, {ipc: [] for ipc in listIPC}, {ipc: [] for ipc in listIPC}, {ipc: [] for ipc in listIPC}

    current_date = int(f"{year}0101")

    # Create a dictionary to store the expected classes for each IPC
    expect_classes_ipc_dict = {}

    # Initialize dictionaries to hold dataframes by IPC and yearRef
    df_KS_dict = {ipc: {} for ipc in listIPC}
    df_ES_dict = {ipc: {} for ipc in listIPC}

    # Load expected classes for each IPC
    for ipc in listIPC:
        expect_classes_ipc_yearRef = []
        with open(pathOutput + f'/ES/text/{year}_{ipc}_expectation_IPC_class.txt', 'r') as fp:
            for line in fp:
                x = line.strip()
                expect_classes_ipc_yearRef.append(x)  # Adjust based on your requirements
        expect_classes_ipc_dict[ipc] = expect_classes_ipc_yearRef

    print(f"Iterating through patents of reference year {yearRef} for evalYear {year}")

    total_files = len(jsonNamesYear)

# Creates lists for both Knowledge Space (KS) and Expectation Space (ES) with batch-size tqdm
    with tqdm(total=total_files, desc='Processing patents') as pbar:
        for i in range(0, total_files, batch_size):
            for j in range(i, min(i + batch_size, total_files)):  # Process in batches
                patent_path = pathYear + jsonNamesYear[j]
                with open(patent_path) as f:
                    d = json.load(f)  # Load JSON in d

                class_mainIPC = d['main_ipcr_label']
                class_main = class_mainIPC[0:4]

                # Collect all documents related to the main class for all IPCs
                for ipc in listIPC:
                    # Create Knowledge Space (KS) for this IPC
                    if class_main == ipc:
                        if int(d['date_published']) < current_date or d['decision'] in ['ACCEPTED', 'REJECTED']:
                            patent_number[ipc].append(d['application_number'])
                            titles[ipc].append(d['title'])
                            backgrounds[ipc].append(d['background'])
                            claims[ipc].append(d['claims'])
                            summary[ipc].append(d['summary'])
                            abstract[ipc].append(d['abstract'])
                            main_ipc[ipc].append(d['main_ipcr_label'])
                            labels[ipc].append(d['decision'])
                            sec_ipc[ipc].append(d['ipcr_labels'])
                            yearRefVec[ipc].append(yearRef)

                    # Create Expectation Space (ES) for this IPC
                    if class_mainIPC in expect_classes_ipc_dict[ipc]:
                        if int(d['date_published']) < current_date or d['decision'] in ['ACCEPTED', 'REJECTED']:
                            patent_numberE[ipc].append(d['application_number'])
                            titlesE[ipc].append(d['title'])
                            backgroundsE[ipc].append(d['background'])
                            claimsE[ipc].append(d['claims'])
                            summaryE[ipc].append(d['summary'])
                            abstractE[ipc].append(d['abstract'])
                            main_ipcE[ipc].append(d['main_ipcr_label'])
                            labelsE[ipc].append(d['decision'])
                            sec_ipcE[ipc].append(d['ipcr_labels'])
                            yearRefVecE[ipc].append(yearRef)

            # Update the progress bar after processing each batch
            pbar.update(min(batch_size, total_files - i))

    for ipc in listIPC:
    # Store KS dataframe in the nested dictionary
        df_KS_dict[ipc][yearRef] = pd.DataFrame({
            'application_number': patent_number[ipc],
            'title': titles[ipc],
            'abstract': abstract[ipc],
            'claims': claims[ipc],
            'background': backgrounds[ipc],
            'summary': summary[ipc],
            'ipc': main_ipc[ipc],
            'sec_ipc': sec_ipc[ipc],
            'label': labels[ipc],
            'yearRef': yearRefVec[ipc]
        })
        # Store ES dataframe in the nested dictionary
        df_ES_dict[ipc][yearRef] = pd.DataFrame({
            'application_number': patent_numberE[ipc],
            'title': titlesE[ipc],
            'abstract': abstractE[ipc],
            'claims': claimsE[ipc],
            'background': backgroundsE[ipc],
            'summary': summaryE[ipc],
            'ipc': main_ipcE[ipc],
            'sec_ipc': sec_ipcE[ipc],
            'label': labelsE[ipc],
            'yearRef': yearRefVecE[ipc]
        })
    return (df_KS_dict, df_ES_dict)


# Function that simply loops over json2_KS_ES yearsNeeded times, and binds dataframes together. Writes a CSV for KS and ES.
def loop_KS_ES(year, yearsNeeded, listIPC, pathData, pathOutput, batch_size=1):
    required_years = set(range(year - yearsNeeded, year))
    dfs_temp = {rY: pd.DataFrame() for rY in required_years}
    for rY in required_years:
        dfs_temp[rY] = json2_KS_ES(year, rY, listIPC, pathData, pathOutput, batch_size)
    for ipc in listIPC:
        df_KS = pd.DataFrame()
        df_ES = pd.DataFrame()
        for i in required_years:
            df_KS_temp = dfs_temp[i][0][ipc][i]
            df_ES_temp = dfs_temp[i][1][ipc][i]
            df_KS = pd.concat([df_KS, df_KS_temp], axis=0, ignore_index=True)
            df_ES = pd.concat([df_ES, df_ES_temp], axis=0, ignore_index=True)
            print("ES and KS done for: " + f"{year}_{i}_{ipc}")
        df_KS.to_csv(pathOutput + f'/KS/{year}_{str(list(required_years)[0])[2:4]}{str(list(required_years)[-1])[2:4]}_{ipc}_KS_raw.csv', index=False)
        df_ES.to_csv(pathOutput + f'/ES/{year}_{str(list(required_years)[0])[2:4]}{str(list(required_years)[-1])[2:4]}_{ipc}_ES_raw.csv', index=False)
        print("df_KS shape: ", df_KS.shape)
        print("df_ES shape: ", df_ES.shape)
    return


def check_files_exist(pathOutput, year, listIPC):
    missing_files = []
    for ipc in listIPC:
        # Construct the expected file name
        file_name = f"toEval/{year}_{ipc}_patents_toEval.csv"  # Adjust extension if needed
        file_path = os.path.join(pathOutput, file_name)
        
        # Check if the file exists
        if not os.path.isfile(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Processing patents for toEval")
        return True
    else: print(f"toEval already done for year {year}")
    return False

# Function taking years to be evaluated, number of years as reference and a list of IPC classes.
def loopFinal(listIPC, listYearsEval, nbYearsRef, pathData, pathOutput, batch_size=1):
    # check if valid years
    for year in listYearsEval:   
        cY = checkYears(year, nbYearsRef, pathData)
        if not cY:
            return
        # Loop through each year
        for year in listYearsEval:
            if (check_files_exist(pathOutput, year, listIPC)):
                json2toEval(year, listIPC, pathData, pathOutput, batch_size)
            loop_KS_ES(year, nbYearsRef, listIPC, pathData, pathOutput, batch_size)
