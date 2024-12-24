from novelty import Newness, Uniqueness, Difference, ClusterKS
from surprise import Surprise
from utils import pmi, docs_distribution

def compute_scores(KB_matrix, KB_dist, NewKB_dist, variation_dist, #EB_PMI, dict_know_pmi, New_EB_PMI,
                    newness_type='div', uniq_type='dist', diff_type='local', neighbor_dist=0., useClusters=False, KSCluster=0):
    
    newness = Newness(KB_dist, variation_dist)
    # print("Newness")
    if newness_type=='div':
        newness, novelty_new = newness.divergent_terms(thr_div=0.0041, thr_new=0.0014)
    else:
        newness, novelty_new = newness.probable_terms(thr_prob= 57.14, thr_new=0.0014)

    # print("Uniqueness")
    uniqueness = Uniqueness(KB_dist)
    if uniq_type == 'dist':
        uniqueness, novelty_uniq = uniqueness.dist_to_proto(variation_dist, thr_uniq=0.527)
    else:
        uniqueness, novelty_uniq = uniqueness.proto_dist_shift(NewKB_dist, thr_uniqp=0.1295)
    
    # print("Difference")
    if useClusters==True:
        if neighbor_dist==0.:
            # print("here")
            neighbor_dist = KSCluster.dist_estimate_clusters(iterations=128, nb_clusters=4)
        if diff_type == "local":
            dif_score, dif_bin, mean100 = KSCluster.ratio_to_neighbors_kmeans(variation_dist=variation_dist, nb_clusters=4, neighbor_dist=neighbor_dist) 
    else:
        difference = Difference(KB_matrix, variation_dist, N=3)
        if neighbor_dist==0.:
            neighbor_dist = difference.dist_estimate()
        if diff_type == 'global':
            dif_score, dif_bin = difference.ratio_to_all(neighbor_dist, thr_diff=0.4177)
        else:
            dif_score, dif_bin = difference.ratio_to_neighbors(neighbor_dist, thr_diff=0.1564)
    
    # print("Surprise")
    # surprise = Surprise(New_EB_PMI)
    # newratio_surprise_rate, newn_suprise = surprise.new_surprise(EB_PMI, thr_surp=0.0104)
    # dist_surprise, uniq_surprise = surprise.uniq_surprise(dict_know_pmi, eps= 0.00, thr_surp=0.00256)
    
    return newness, novelty_new, uniqueness, novelty_uniq, dif_score, dif_bin, neighbor_dist, mean100 #, newratio_surprise_rate, newn_suprise, dist_surprise, uniq_surprise