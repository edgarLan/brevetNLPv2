from Novelty import Newness, Uniqueness, Difference
from Surprise import Surprise
from utils import pmi, data_analysis, docs_distribution

def compute_scores(KB_matrix, KB_dist, NewKB_dist, variation_dist, EB_PMI, dict_know_pmi, New_EB_PMI, newness_type='div', uniq_type='dist', diff_type='local', neighbor_dist=0.):
    
    newness = Newness(KB_dist, variation_dist)
    if newness_type=='div':
        newness, novelty_new = newness.divergent_terms(thr_div=0.0041, thr_new=0.0014)
    else:
        newness, novelty_new = newness.probable_terms(thr_prob= 57.14, thr_new=0.0014)

    uniqueness = Uniqueness(KB_dist)
    if uniq_type == 'dist':
        uniqueness, novelty_uniq = uniqueness.dist_to_proto(variation_dist, thr_uniq=0.527)
    else:
        uniqueness, novelty_uniq = uniqueness.proto_dist_shift(NewKB_dist, thr_uniqp=0.1295)
    
    difference = Difference(KB_matrix, variation_dist, N=3)
    if neighbor_dist==0.:
        neighbor_dist = difference.dist_estimate()
    if diff_type == 'global':
        diff_ratio, nolvety_diff = difference.ratio_to_all(neighbor_dist, thr_diff=0.4177)
    else:
        diff_ratio, nolvety_diff = difference.ratio_to_neighbors(neighbor_dist, thr_diff=0.1564)
    
    surprise = Surprise(New_EB_PMI)
    newratio_surprise_rate, newn_suprise = surprise.new_surprise(EB_PMI, thr_surp=0.0104)
    dist_surprise, uniq_surprise = surprise.uniq_surprise(dict_know_pmi, eps= 0.00, thr_surp=0.00256)
    
    return newness, novelty_new, uniqueness, novelty_uniq, diff_ratio, nolvety_diff, neighbor_dist, newratio_surprise_rate, newn_suprise, dist_surprise, uniq_surprise