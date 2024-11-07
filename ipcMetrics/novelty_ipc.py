import heapq
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
from divergences import Distance
from collections import Counter, defaultdict
from itertools import combinations, chain
from divergences import Distance
import ast


class Newness():
    """
    Class to calculate Newness metric with IPC codes
    Inputs:
        list_unique_known: list of all known unique IPC codes (Knowledge or Expectation Base).
        list_new: list containing lists of IPCs per patent).
    """
    
    def __init__(self, list_unique_known, list_new):
        self.list_unique_known = list_unique_known
        self.list_new = list_new
    """
    Function that calculates the ratio of new IPC codes on known IPC codes in novel patents.
    Keeping main IPC code in the computation because new main IPC code probably means newness... (arrive à peine)
    Inputs:
        thr: threshold for binary score
    Outputs:
        prop: List of proportions - number of new IPC codes/number of IPC codes in patent, for each new patent
        bin: List of binary variables indicating if prop > thr
        len1: List of binary variables indicating if number of ipc codes (inclusing main) == 1
    """
    def novelty_newness(self, thr=0.1):
        prop = []  
        bin = []
        len1 = []
        total_patents = len(self.list_new)

        for patent in tqdm(range(total_patents)):
            novelty = sum(1 for i in self.list_new[patent] if i not in self.list_unique_known)
            # Normalize the novelty score, handling the division by zero case
            normalized_novelty = novelty / len(self.list_new[patent]) if self.list_new[patent] else novelty
            prop.append(normalized_novelty)
            bin.append(1 if normalized_novelty > thr else 0)
            len1.append(1 if len(self.list_new[patent]) == 1 else 0)
            
        return prop, bin, len1

class Uniqueness():
    """
    Class to calculate Uniqueness metric with IPC codes
    Inputs:
        list_known: list containing lists of IPCs for all known patents.
        list_new: list containing lists of IPCs for all novel patents.
    """
    def __init__(self, list_known, list_new):
        self.list_known = list_known
        self.list_new = list_new

    """
    Method that calculates the jaccard distance between novel patents and the prototype patent.
    Prototype is simply the most common IPC combination (no matter the number of IPC codes).
    Inputs:
        thr: threshold for binary score
    Outputs:
        prop: List of proportions number of new IPC codes/number of IPC codes in patent, for each new patent
        bin: List of binary variables indicating if prop > thr
    """
    def dist_toProto(self, thr=0.05):
        set_new = [set(sublist) for sublist in self.list_new]
        proto = list(Counter([tuple(sublist) for sublist in self.list_known]).most_common(1)[0][0])
        prop = []
        bin = []
        for patent in tqdm(set_new):
            uniqueness = Distance.jaccard_dist(patent, proto)
            prop.append(uniqueness) 
            bin.append(1 if uniqueness > thr else 0)           
        return prop, bin


class Difference():
    
    """
    Class to calculate Newness metric with IPC codes
    Inputs:
        list_known: list containing lists of IPCs for all known patents.
        list_new: list containing lists of IPCs for all novel patents.
        N: Number of neighbors to consider.
        nb_K: Number of patents to consider in list_known (sampled randomly)
        chunkszie: Set at 1000, if RAM problems because jaccard matrix too large, reduce it.

    When initializing, calculation of neighbor_dist, the average across nb_K patent of list_known of mean distance to N closest neighbors.
    """
    def __init__(self, list_known, list_new, N=10, nb_K=1000, chunksize=1000):
        self.list_known = list_known
        self.list_new = list_new
        self.N = N
        self.nb_K = nb_K
        self.chunksize=chunksize
        self.dist_fct = Distance()      
        self.neighbor_dist = self.dist_estimate() # seuil


    def dist_estimate(self):
        """
        Method that calculates the average, across nb_K patent of list_known, of mean distance to N closest neighbors.
        First, calculates mean jaccard distance of all neighbours for nb_K patents of list_known.
        Second, calculate mean of N closest neighbors for all nb_K patents.
        Thirdly, calculate mean of means
        """
        nb_K = self.nb_K
        if nb_K=="all" or nb_K > len(self.list_known):
            nb_K = len(self.list_known)
            
        mean_closest = []
        # Randomly sample nb_K patents from list_known
        list_rand = pd.Series(self.list_known).sample(n=nb_K) #, random_state=41)
        list_rand = [set(sublist) for sublist in list_rand]
        # Process distances in chunks
        for start_idx in range(0, nb_K, self.chunksize):
            end_idx = min(start_idx + self.chunksize, nb_K)
            chunk = list_rand[start_idx:end_idx]
            
            # Compute the Jaccard distance between each patent in the chunk and all patents in list_known
            jaccard_dist_matrix = self.dist_fct.jacc_dist_matrix(chunk, self.list_known)
            
            # Calculate the mean distance to the N closest neighbors for each row in the chunk
            for row in jaccard_dist_matrix:
                closest_indices = np.argsort(row)[:self.N]  # Find the N closest neighbors for this row
                mean_value = np.mean(row[closest_indices])  # Mean of these N closest distances
                mean_closest.append(mean_value)
        
        # Calculate the final mean across all rows
        average_distance = np.mean(mean_closest)

        return average_distance
        
    
    def ratio_toAll(self, n="all", thr=0.95):
        """
        For each new patent in list_new, returns proportion of patents in list_known that are different (greater then neighbor_dist).
        Inputs:
            n is the number of patents taken in the list_new (to shorten time). If n="all", then all list_new is taken.
            thr: threshold for binary score
        """
        # if n=="all":
        #     n=len(self.list_new)
        ratios=[]
        bin = []
        set_new = [set(sublist) for sublist in self.list_new]
        set_known_P = [set(sublist) for sublist in self.list_known]
        #random.seed(42)
        for Q_i in tqdm(set_new):
            count_diff = 0
            for P_i in set_known_P:
                distance = self.dist_fct.jaccard_dist(P_i, Q_i)
                if distance > self.neighbor_dist:
                    count_diff += 1
            difference = count_diff/len(self.list_known)
            ratios.append(difference)
            bin.append(1 if difference > thr else 0) 
        return ratios, bin


    def ratio_toAll_matrix(self, n="all", thr=0.95):
        """
        is a quicker version of ratio_toAll - might have problems with RAM, but chunksize adjustment should do the job.
        For each new patent in list_new, returns proportion of patents in list_known that are different (greater then neighbor_dist).
        Inputs:
            thr: threshold for binary score
        """

        # if n=="all":
        #     n=len(self.list_new)
        ratios=[]
        bin=[]
        #random.seed(42)
        # Sample `n` patents from list_new
        # sample_list_new = random.sample(self.list_new, n)
        
        # Process in chunks of `chunksize`
        for i in tqdm(range(0, len(self.list_new), self.chunksize)):
            # Define the current chunk from sample_list_new
            chunk = self.list_new[i:i + self.chunksize]
            
            # Calculate the Jaccard distance matrix for the current chunk against all patents in list_known
            jaccard_dist_matrix = self.dist_fct.jacc_dist_matrix(chunk, self.list_known)
            
            # Process each row of the Jaccard distance matrix for the current chunk
            for row in jaccard_dist_matrix:
                # Count the patents in list_known that are sufficiently different (distance > neighbor_dist)
                length = len([value for value in row if value > self.neighbor_dist])
                difference = length / len(row)
                ratios.append(difference)
                bin.append(1 if difference > thr else 0) 
        return ratios, bin


    def ratio_toNeighbors(self, n="all", thr=0.85):
        """
        is a quicker version a ratio_toAll_matrix - might have problems with RAM, but chunksize adjustment should do the job.
        For each new patent in list_new, returns proportion of N CLOSEST NEIGHBOR patents in list_known that are different (greater then neighbor_dist).
        Inputs:
            n is the number of patents taken in the list_new (to shorten time). If n="all", then all list_new is taken.
            thr: threshold for binary score
        """

        # if n=="all":
        #     n=len(self.list_new)
        ratios = []
        bin=[]
        set_new = [set(sublist) for sublist in self.list_new]
        set_known_P = [set(sublist) for sublist in self.list_known]
        #random.seed(42)
        for Q_i in tqdm(set_new):
            count_diff = 0
            all_dists = []
            closest_dists = []
            for P_i in set_known_P:
                all_dists.append(self.dist_fct.jaccard_dist(P_i, Q_i))
            closest_dists = heapq.nsmallest(self.N, all_dists)
            for dist in closest_dists:
                if dist > self.neighbor_dist:
                    count_diff += 1
            difference = count_diff/self.N
            ratios.append(difference)
            bin.append(1 if difference > thr else 0) 
        return ratios, bin

    
    def ratio_toNeighbors_matrix(self, n="all", thr=0.85, chunksize=500):
        """
        is a quicker version of ratio_toNeighbors - might have problems with RAM, but chunksize adjustment should do the job.
        For each new patent in list_new, returns proportion of N CLOSEST NEIGHBOR patents in list_known that are different (greater then neighbor_dist).
        Inputs:
            n is the number of patents taken in the list_new (to shorten time). If n="all", then all list_new is taken.
            thr: threshold for binary score
        """
        # if n=="all" or n>len(self.list_new):
        #     n=len(self.list_new)
        #     sample_list_new=self.list_new
        # else:
        #     sample=sample_list_new = random.sample(self.list_new, n)
        ratios=[]
        bin=[]

        # Process in chunks of `chunksize`
        for i in tqdm(range(0, len(self.list_new), chunksize)):
            # Define the current chunk from list_new
            chunk = self.list_new[i:i + chunksize]
            
            # Calculate the Jaccard distance matrix for the current chunk against all patents in list_known
            jaccard_dist_matrix = self.dist_fct.jacc_dist_matrix(chunk, self.list_known)
            
            # Process each row of the Jaccard distance matrix for the current chunk
            for row in jaccard_dist_matrix:
                # Count the patents in list_known that are sufficiently different (distance > neighbor_dist)
                closest_dists = heapq.nsmallest(self.N, row)
                length = len([value for value in closest_dists if value > self.neighbor_dist])
                difference = length / self.N
                ratios.append(difference)
                bin.append(1 if difference > thr else 0) 
        return ratios, bin


  
class Surprise():
    
    """
    Class to calculate Surprise metrics with IPC codes
    Inputs:
        list_known: list containing lists of IPCs for all known patents.
        list_new: list containing lists of IPCs for all novel patents.

    When initializing, defines:
        dict_new: Dictionary with key being pairs of IPCs from new patents.
        dict_known: Dictionary with key being pairs of IPCs from new patents.

    """
    def __init__(self, list_expec, list_new):
        self.list_expec = list_expec
        self.list_new = list_new
        self.dict_expec = Distance.pmi(self.list_expec)
        self.dict_new = Distance.pmi(self.list_expec)

    def surprise_new(self, thr=0):
        """
        Method that calculates the ratio of new pairs of IPCs codes on expected pairs IPC codes for novel patents.
        Inputs
            thr: threshold for binary variable
        """
        # Generate pairs in list_expec
        pairs_expec = set()
        for ipc_list in self.list_expec:
            # Generate all pairs of IPCs in the current patent
            for pair in combinations(ipc_list, 2):
                pairs_expec.add(tuple(sorted(pair))) 

        prop = []
        bin = []
        for patent in tqdm(self.list_new):
            patent_size = len(patent)
            if patent_size < 2:
                prop.append(0)
                bin.append(0)
                continue
            # Count the surprising pairs
            surprise_count = sum(1 for pair in combinations(patent, 2) if tuple(sorted(pair)) not in pairs_expec)
            
            # Normalize by the total number of pairs
            total_pairs = patent_size * (patent_size - 1) / 2
            normalized_novelty = surprise_count / total_pairs
            
            prop.append(normalized_novelty)
            bin.append(1 if normalized_novelty > thr else 0)
            
        return prop, bin
       
    def surprise_div(self, thr):
        """
        Method that calculates the sum of PMIs for novel patents/nb of pairs in patent. PMIs come from dict_expec. 
        If only 1 IPC in patent, PMI is 0, if the IPC pair is not found in dict_expec, max PMI value is given.
        Inputs
            thr: threshold for binary variable
        """
        pmi_max=np.max(list(self.dict_expec.values()))
        dict_expec_lambda = defaultdict(lambda: pmi_max, self.dict_expec)
        pmi_surprise = []
        bin=[]

        for patent in tqdm(self.list_new):
            patent_size = len(patent)

            # Sum the PMI values of the pairs in the new patent
            pmi_sum = sum(
                dict_expec_lambda[tuple(sorted([patent[i], patent[j]]))]
                for i in range(patent_size)
                for j in range(i + 1, patent_size)
            )
            pmi_surprise.append(0 if patent_size<2 else pmi_sum/(patent_size * (patent_size - 1) / 2))
            bin.append(1 if pmi_sum > thr else 0)
        return pmi_surprise, bin
      

def compute_scores(list_unique_known, list_known, list_expec, ipc, year, list_new, N=10, nb_K=1000, thr_surp=0, thr_uniq=0, thr_diff=0, thr_surpNew=0, thr_surpDiv=0):
    print()
    print()
    instanceNewness = Newness(list_unique_known=list_unique_known, list_new=list_new)
    new_ratio, new_bin, len1 = instanceNewness.novelty_newness(thr=thr_surp)
    print(f"Newness for {ipc} in {year}; mean: {np.mean(new_ratio)}; sd: {np.std(new_ratio)}, mean+sdL: {np.mean(new_ratio)+np.std(new_ratio)}")

    instanceUniquess = Uniqueness(list_known=list_known, list_new=list_new)
    uniq_ratio, uniq_bin = instanceUniquess.dist_toProto(thr=thr_uniq)
    print(f"Uniqueness for {ipc} in {year}; mean: {np.mean(uniq_ratio)}; sd: {np.std(uniq_ratio)}, mean+sdL: {np.mean(uniq_ratio)+np.std(uniq_ratio)}")

    instanceDifference = Difference(list_known=list_known, list_new=list_new, N=N, nb_K=nb_K) 
    diff_ratio, diff_bin = instanceDifference.ratio_toNeighbors_matrix(thr=thr_diff)
    print(f"Difference for {ipc} in {year}; mean: {np.mean(diff_ratio)}; sd: {np.std(diff_ratio)}, mean+sdL: {np.mean(diff_ratio)+np.std(diff_ratio)}")

    instanceSurprise = Surprise(list_expec=list_expec, list_new=list_new)
    surpNew_ratio, surpNew_bin = instanceSurprise.surprise_new(thr=thr_surpNew)
    print(f"SurpriseNew for {ipc} in {year}; mean: {np.mean(surpNew_ratio)}; sd: {np.std(surpNew_ratio)}, mean+sdL: {np.mean(surpNew_ratio)+np.std(surpNew_ratio)}")
    surpDiv_ratio, surpDiv_bin = instanceSurprise.surprise_div(thr=thr_surpDiv)
    print(f"SurpriseDiv for {ipc} in {year}; mean: {np.mean(surpDiv_ratio)}; sd: {np.std(surpDiv_ratio)}, mean+sdL: {np.mean(surpDiv_ratio)+np.std(surpDiv_ratio)}")
    print()
    print()
    return new_ratio, new_bin, uniq_ratio, uniq_bin, diff_ratio, diff_bin, surpNew_ratio, surpNew_bin, surpDiv_ratio, surpDiv_bin, len1
        

def ipcMetrics(listIPC, listYear, pathData, pathOutput, thr_surp=0, thr_uniq=0, thr_diff=0, thr_surpNew=0, thr_surpDiv=0):
    new_len1=[]
    new_ratio=[]
    new_bin=[]
    uniq_ratio=[]
    uniq_bin=[]
    diff_ratio=[]
    diff_bin=[]
    surpNew_ratio=[]
    surpNew_bin=[]
    surpDiv_ratio=[]
    surpDiv_bin=[]

    for ipc in listIPC:
        print(f"{ipc}")
        for year in listYear:
            print(f"     {year}")
            toEval = pd.read_csv(pathData + f'/toEval/{year}_{ipc}_patents_toEval.csv')
            KS = pd.read_csv(pathData + f'/KS/{year}_{str(year-5)[2:4]}{str(year-1)[2:4]}_{ipc}_KS_raw.csv')
            ES = pd.read_csv(pathData + f'/ES/{year}_{str(year-5)[2:4]}{str(year-1)[2:4]}_{ipc}_ES_raw.csv')

            toEval_ipc = list([ast.literal_eval(i) for i in toEval.sec_ipc])
            KS_ipc = list(set(chain.from_iterable([ast.literal_eval(s) for s in KS.sec_ipc])))
            KS_sec_ipc = list(pd.Series([ast.literal_eval(i) for i in KS.sec_ipc]))
            ES_sec_ipc = list(pd.Series([ast.literal_eval(i) for i in ES.sec_ipc]))

            results = compute_scores(list_unique_known=KS_ipc, list_known=KS_sec_ipc, list_expec=ES_sec_ipc, list_new=toEval_ipc, ipc=ipc, year=year, N=100, nb_K=1000, thr_surp=thr_surp, thr_uniq=thr_uniq, thr_diff=thr_diff, thr_surpNew=thr_surpNew, thr_surpDiv=thr_surpDiv)
            
            new_ratio = results[0]
            new_bin = results[1]
            new_len1 = results[10]

            uniq_ratio = results[2]
            uniq_bin = results[3]

            diff_ratio = results[4]
            diff_bin = results[5]

            surpNew_ratio = results[6]
            surpNew_bin = results[7]

            surpDiv_ratio = results[8]
            surpDiv_bin = results[9]

            df = pd.DataFrame({
                "application_number": toEval["application_number"],
                "label": toEval["label"],

                "new_ratio": new_ratio,
                "new_bin": new_bin,
                "new_len1": new_len1,

                "uniq_ratio": uniq_ratio,
                "uniq_bin": uniq_bin,

                "diff_ratio": diff_ratio,
                "diff_bin": diff_bin, 

                "surpNew_ratio": surpNew_ratio,
                "surpNew_bin": surpNew_bin,

                "surpDiv_ratio":surpDiv_ratio,
                "surpDiv_bin": surpDiv_bin
            })
            df.to_csv(pathOutput + f'/{year}_{ipc}_ipcMetrics.csv', index=False)