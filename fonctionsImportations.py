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

    pathYear = pathData + f"/{year}/"  # Updates with variable year
    jsonNamesYear = [f for f in listdir(pathYear) if isfile(join(pathYear, f))] 

    # Initialize list for each IPC class
    patent_listIPC_dict = {ipc: [] for ipc in listIPC}  # A list to store patents for each IPC

    # Set up tqdm progress bar
    batch_size = 2  # Set a batch size for updating the progress bar
    total_patents = len(jsonNamesYear)  # Total number of patents to process
    pbar = tqdm(total=total_patents, desc="Processing patents")

    # Creates list of patents for each IPC class
    for i in range(len(jsonNamesYear)):
        patent_path = pathYear + jsonNamesYear[i]
        with open(patent_path) as f:
            d = json.load(f)  # Load json in d

        class_mainIPC = d['main_ipcr_label']
        
        # Check if the class matches any IPC in the listIPC
        for ipc in listIPC:
            if re.match(f'^{ipc}', class_mainIPC):
                patent_listIPC_dict[ipc].append(jsonNamesYear[i])
        
        # Update progress bar in batches
        if (i + 1) % batch_size == 0:
            pbar.update(batch_size)

    # Update the remaining progress
    remaining = total_patents % batch_size
    if remaining > 0:
        pbar.update(remaining)

    final_patents_dict = {ipc: [] for ipc in listIPC}  # Create a final list for each IPC

    # Create list excluding all other than accepted and rejected
    for ipc in listIPC:
        # Access the patents for each IPC class
        patent_listIPC = patent_listIPC_dict[ipc]  
                
        for i in range(len(patent_listIPC)):
            patent_path = pathYear + patent_listIPC[i]
            with open(patent_path) as f:
                d = json.load(f)
            
            # Check if the decision is either ACCEPTED or REJECTED
            if d['decision'] in ['ACCEPTED', 'REJECTED']: 
                final_patents_dict[ipc].append(patent_listIPC[i])  # Add to the corresponding IPC list

        # Update the remaining progress
        remaining = len(patent_listIPC) % batch_size
        if remaining > 0:
            pbar.update(remaining)
    
    # Close the progress bar for the inner loop
    pbar.close()

    good_expectations_classes_dict = {ipc: [] for ipc in listIPC}  # To store good expectations for each IPC

    # Load needed data for patents by IPC class
    data_by_ipc = {ipc: {
        'patent_number': [], 'titles': [], 'backgrounds': [], 'claims': [],
        'summary': [], 'abstract': [], 'main_ipc': [], 'sec_ipc': [], 'labels': []
    } for ipc in listIPC}

    for ipc in listIPC:
        non_main_ipc = []  
        pbar = tqdm(total=len(final_patents_dict[ipc]), desc=f"Create toEval for {ipc} in {year}")
        
        for i in range(len(final_patents_dict[ipc])):
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

            # Update progress bar in batches
            if (i + 1) % batch_size == 0:
                pbar.update(batch_size)

        # Update the remaining progress
        remaining = len(final_patents_dict[ipc]) % batch_size
        if remaining > 0:
            pbar.update(remaining)

        # Keep only unique secondary IPC classes
        expectations_classes = list(set(non_main_ipc))  # Unique secondary IPC classes
        good_expectations_classes_dict[ipc] = [ipcr for ipcr in expectations_classes if ipcr[:4] != ipc]  # Exclude main IPC classes

        pbar.close()

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
        print(f"toEval done for {ipc} in {year} ")

        # Save IPC in text format for each IPC class
        with open(pathOutput + f'/ES/text/{year}_{ipc}_expectation_IPC_class.txt', 'w') as fp:
            for item in good_expectations_classes_dict[ipc]:
                fp.write("%s\n" % item)
            print(f"Number of secondary IPC for {ipc} in {year}: ", len(good_expectations_classes_dict[ipc]))



# Function taking ipc list, year studied, year in which we are searching, and path for input and output, and outputs dfs for ES et KS for the year_yearRef for all IPCs
def json2_KS_ES(year, yearRef, listIPC, pathData, pathOutput):

    pathYear = pathData + f"/{yearRef}/"  # Updates with variable year
    jsonNamesYear = [f for f in listdir(pathYear) if isfile(join(pathYear, f))]

    current_date = int(f"{year}0101")

    # Create dictionaries to store KS and ES for each IPC
    KS_ipc_dict = {ipc: [] for ipc in listIPC}
    ES_ipc_dict = {ipc: [] for ipc in listIPC}

    # Create a dictionary to store the expected classes for each IPC
    expect_classes_ipc_dict = {}

    # Load expected classes for each IPC
    for ipc in listIPC:
        expect_classes_ipc_yearRef = []
        with open(pathOutput + f'/ES/text/{year}_{ipc}_expectation_IPC_class.txt', 'r') as fp:
            for line in fp:
                x = line[:-1]
                expect_classes_ipc_yearRef.append(x)  # Adjust based on your requirements
        expect_classes_ipc_dict[ipc] = expect_classes_ipc_yearRef

    print(f"Iterating through patents of reference year {yearRef} for evalYear {year}")
    
    # Set up tqdm progress bar
    batch_size = 2  # Set a batch size for updating the progress bar
    total_patents = len(jsonNamesYear)  # Total number of patents to process
    pbar = tqdm(total=total_patents, desc="Processing patents for ES/KS")

    for i in range(len(jsonNamesYear)):
        patent_path = pathYear + jsonNamesYear[i]
        with open(patent_path) as f:
            d = json.load(f)
            f.close()

        class_mainIPC = d['main_ipcr_label']
        class_main = class_mainIPC[0:4]

        # Collect all documents related to the main class for all IPCs ###########
        for ipc in listIPC:
            if class_main == ipc:
                KS_ipc_dict[ipc].append(jsonNamesYear[i])
            if class_mainIPC in expect_classes_ipc_dict[ipc]:
                ES_ipc_dict[ipc].append(jsonNamesYear[i])
        
            # Update progress bar in batches
            if (i + 1) % batch_size == 0:
                pbar.update(batch_size)

        # Update the remaining progress
        remaining = len(jsonNamesYear[ipc]) % batch_size
        if remaining > 0:
            pbar.update(remaining)

    pbar.close()

    # Now create KS and ES dataframes for each IPC
    df_KS_dict = {}
    df_ES_dict = {}

    for ipc in listIPC:
        print(f"Creating KS and ES for IPC {ipc}")

        # Create Knowledge Space (KS) for this IPC
        patent_number, titles, backgrounds, claims, summary, abstract, main_ipc, labels, sec_ipc, yearRefVec = [], [], [], [], [], [], [], [], [], []
        for patent_file in tqdm(KS_ipc_dict[ipc]):
            patent_path = pathYear + patent_file
            with open(patent_path) as f:
                d = json.load(f)
                f.close()

            if int(d['date_published']) < current_date or d['decision'] in ['ACCEPTED', 'REJECTED']:
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

        df_KS_dict[ipc] = pd.DataFrame({
            'application_number': patent_number, 'title': titles, 'abstract': abstract,
            'claims': claims, 'background': backgrounds, 'summary': summary, 'ipc': main_ipc,
            'sec_ipc': sec_ipc, 'label': labels, 'yearRef': yearRefVec
        })

        # Create Expectation Space (ES) for this IPC
        patent_number, titles, backgrounds, claims, summary, abstract, main_ipc, labels, sec_ipc, yearRefVec = [], [], [], [], [], [], [], [], [], []
        for patent_file in tqdm(ES_ipc_dict[ipc]):
            patent_path = pathYear + patent_file
            with open(patent_path) as f:
                d = json.load(f)
                f.close()

            if int(d['date_published']) < current_date or d['decision'] in ['ACCEPTED', 'REJECTED']:
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

        df_ES_dict[ipc] = pd.DataFrame({
            'application_number': patent_number, 'title': titles, 'abstract': abstract,
            'claims': claims, 'background': backgrounds, 'summary': summary, 'ipc': main_ipc,
            'sec_ipc': sec_ipc, 'label': labels, 'yearRef': yearRefVec
        })

    return (df_KS_dict, df_ES_dict)


# Function taking list of IPC, year studied and one year reference and returns dfs for all ES and KS for that year_yearRef for all IPCs.
def json2_KS_ES(year, yearRef, listIPC, pathData, pathOutput):
    pathYear = pathData + f"/{yearRef}/"  # Updates with variable year
    jsonNamesYear = [f for f in listdir(pathYear) if isfile(join(pathYear, f))]

    current_date = int(f"{year}0101")

    # Create dictionaries to store KS and ES for each IPC and yearRef
    KS_ipc_dict = {ipc: [] for ipc in listIPC}
    ES_ipc_dict = {ipc: [] for ipc in listIPC}

    # Create a dictionary to store the expected classes for each IPC
    expect_classes_ipc_dict = {}

    # Load expected classes for each IPC
    for ipc in listIPC:
        expect_classes_ipc_yearRef = []
        with open(pathOutput + f'/ES/text/{year}_{ipc}_expectation_IPC_class.txt', 'r') as fp:
            for line in fp:
                x = line.strip()
                expect_classes_ipc_yearRef.append(x)  # Adjust based on your requirements
        expect_classes_ipc_dict[ipc] = expect_classes_ipc_yearRef

    print(f"Iterating through patents of reference year {yearRef} for evalYear {year}")
    
    for i in tqdm(range(len(jsonNamesYear))):
        patent_path = pathYear + jsonNamesYear[i]
        with open(patent_path) as f:
            d = json.load(f)

        class_mainIPC = d['main_ipcr_label']
        class_main = class_mainIPC[0:4]

        # Collect all documents related to the main class for all IPCs
        for ipc in listIPC:
            if class_main == ipc:
                KS_ipc_dict[ipc].append(jsonNamesYear[i])
            if class_mainIPC in expect_classes_ipc_dict[ipc]:
                ES_ipc_dict[ipc].append(jsonNamesYear[i])

    # Initialize dictionaries to hold dataframes by IPC and yearRef
    df_KS_dict = {ipc: {} for ipc in listIPC}
    df_ES_dict = {ipc: {} for ipc in listIPC}

    for ipc in listIPC:
        print(f"Creating KS and ES for IPC {ipc}")

        # Create Knowledge Space (KS) for this IPC
        patent_number, titles, backgrounds, claims, summary, abstract, main_ipc, labels, sec_ipc, yearRefVec = [], [], [], [], [], [], [], [], [], []
        for patent_file in tqdm(KS_ipc_dict[ipc]):
            patent_path = pathYear + patent_file
            with open(patent_path) as f:
                d = json.load(f)

            if int(d['date_published']) < current_date or d['decision'] in ['ACCEPTED', 'REJECTED']:
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

        # Store KS dataframe in the nested dictionary
        df_KS_dict[ipc][yearRef] = pd.DataFrame({
            'application_number': patent_number,
            'title': titles,
            'abstract': abstract,
            'claims': claims,
            'background': backgrounds,
            'summary': summary,
            'ipc': main_ipc,
            'sec_ipc': sec_ipc,
            'label': labels,
            'yearRef': yearRefVec
        })

        # Create Expectation Space (ES) for this IPC
        patent_number, titles, backgrounds, claims, summary, abstract, main_ipc, labels, sec_ipc, yearRefVec = [], [], [], [], [], [], [], [], [], []
        for patent_file in tqdm(ES_ipc_dict[ipc]):
            patent_path = pathYear + patent_file
            with open(patent_path) as f:
                d = json.load(f)

            if int(d['date_published']) < current_date or d['decision'] in ['ACCEPTED', 'REJECTED']:
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

        # Store ES dataframe in the nested dictionary
        df_ES_dict[ipc][yearRef] = pd.DataFrame({
            'application_number': patent_number,
            'title': titles,
            'abstract': abstract,
            'claims': claims,
            'background': backgrounds,
            'summary': summary,
            'ipc': main_ipc,
            'sec_ipc': sec_ipc,
            'label': labels,
            'yearRef': yearRefVec
        })

    return (df_KS_dict, df_ES_dict)



# Function that simply loops over json2_KS_ES yearsNeeded times, and binds dataframes together. Writes a CSV for KS and ES.
def loop_KS_ES(year, yearsNeeded, listIPC, pathData, pathOutput):
    required_years = set(range(year - yearsNeeded, year))
    df_KS = pd.DataFrame()
    df_ES = pd.DataFrame()
    for ipc in listIPC:
        for i in required_years:
            dfs_temp = json2_KS_ES(year, i, listIPC, pathData, pathOutput)
            df_KS_temp = dfs_temp[0][ipc][i]
            df_ES_temp = dfs_temp[1][ipc][i]
            df_KS = pd.concat([df_KS, df_KS_temp], axis=0, ignore_index=True)
            df_ES = pd.concat([df_ES, df_ES_temp], axis=0, ignore_index=True)
            print("ES and KS done for: " + f"{year}_{i}_{ipc}")
        df_KS.to_csv(pathOutput + f'/KS/{year}_{str(list(required_years)[0])[2:4]}{str(list(required_years)[-1])[2:4]}_{ipc}_KS_raw.csv', index=False)
        df_ES.to_csv(pathOutput + f'/ES/{year}_{str(list(required_years)[0])[2:4]}{str(list(required_years)[-1])[2:4]}_{ipc}_ES_raw.csv', index=False)
        print("df_KS shape: ", df_KS.shape)
        print("df_ES shape: ", df_ES.shape)

# Function taking years to be evaluated, number of years as reference and a list of IPC classes.
def loopFinal(listIPC, listYearsEval, nbYearsRef, pathData, pathOutput):
    # check if valid years
    for year in listYearsEval:   
        cY = checkYears(year, nbYearsRef, pathData)
        if not cY:
            return
        # Loop through each year
        for year in listYearsEval:
            json2toEval(year, listIPC, pathData, pathOutput)
            loop_KS_ES(year, nbYearsRef, listIPC, pathData, pathOutput)
