import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from itertools import groupby
from dictances import bhattacharyya
from scipy.stats import wasserstein_distance
import torch
import sys
sys.path.append('./src/')
from dataBuilders.data_builder import randsplit, randsplit_comb
from evaluation.saliency_metric import nw_matching
from evaluation.multimatch import docomparison
from tqdm import tqdm
#from tabulate import tabulate

def _Levenshtein_Dmatrix_initializer(len1, len2):
    Dmatrix = []

    for i in range(len1):
        Dmatrix.append([0] * len2)

    for i in range(len1):
        Dmatrix[i][0] = i

    for j in range(len2):
        Dmatrix[0][j] = j

    return Dmatrix

def _Levenshtein_cost_step(Dmatrix, string_1, string_2, i, j, substitution_cost=1):
    char_1 = string_1[i - 1]
    char_2 = string_2[j - 1]

    # insertion
    insertion = Dmatrix[i - 1][j] + 1
    # deletion
    deletion = Dmatrix[i][j - 1] + 1
    # substitution
    substitution = Dmatrix[i - 1][j - 1] + substitution_cost * (char_1 != char_2)

    # pick the cheapest
    Dmatrix[i][j] = min(insertion, deletion, substitution)

def _Levenshtein(string_1, string_2, substitution_cost=1):
    # get strings lengths and initialize Distances-matrix
    len1 = len(string_1)
    len2 = len(string_2)
    Dmatrix = _Levenshtein_Dmatrix_initializer(len1 + 1, len2 + 1)

    # compute cost for each step in dynamic programming
    for i in range(len1):
        for j in range(len2):
            _Levenshtein_cost_step(Dmatrix,
                                   string_1, string_2,
                                   i + 1, j + 1,
                                   substitution_cost=substitution_cost)

    if substitution_cost == 1:
        max_dist = max(len1, len2)
    elif substitution_cost == 2:
        max_dist = len1 + len2

    return Dmatrix[len1][len2]


def behavior(result_array, target, gaze, benchmark=False):
    behavior = {'correct':[],'length': [], 'search': [], 'refix': [], 'revisit': []}
    for i in range(len(gaze)):
        if len(gaze[i]) == 0:
            print('GAZE LENGTH IS ZERO')
            continue
        gaze_element = gaze[i][~np.isnan(gaze[i])]
        if len(gaze_element) == 0:
            print('replacing it...')
            gaze_element = gaze[i-1][~np.isnan(gaze[i-1])]
        behavior['correct'].append(int(target == gaze_element[-1]))
        behavior['length'].append(len(gaze_element))
        search_len = 0
        refix_len = 0
        revisit_len = 0
        previous_visited = []
        for k, g in groupby(gaze_element):
            subiterator_len = len(list(g))
            if k in previous_visited:
                revisit_len += 1
            else:
                search_len += 1
            if subiterator_len > 1:
                refix_len += (subiterator_len - 1)
            previous_visited.append(k)
        assert search_len + refix_len + revisit_len == len(gaze_element)
        behavior['search'].append(search_len / len(gaze_element))
        behavior['refix'].append(refix_len / len(gaze_element))
        behavior['revisit'].append(revisit_len / len(gaze_element))
    result_array[0] += np.mean(behavior['correct'])   
    result_array[1] += np.mean(behavior['length']) 
    result_array[2] += np.mean(behavior['search']) 
    result_array[3] += np.mean(behavior['refix']) 
    result_array[4] += np.mean(behavior['revisit']) 

def string_distance(result_array,gaze,gt,col_num,row_num):
    gt = gt[~np.isnan(gt)]
    ED = []
    distance = {'SS':[],'VectorSimilarity': [], 'DirectionSimilarity': [], 'LengthSimilarity': [], 'PositionSimilarity': [],'Average':[]}
    for i in range(len(gaze)):
        gaze_element = gaze[i][~np.isnan(gaze[i])]
        scanpathcomparisons = [0,0,0,0] #docomparison(gaze_element, gt,col_num, row_num)[0]
        distance['SS'].append(nw_matching(gaze_element, gt))
        distance['VectorSimilarity'].append(scanpathcomparisons[0])
        distance['DirectionSimilarity'].append(scanpathcomparisons[1])
        distance['LengthSimilarity'].append(scanpathcomparisons[2])
        distance['PositionSimilarity'].append(scanpathcomparisons[3])
        distance['Average'].append(np.mean(scanpathcomparisons))
        ED.append(_Levenshtein(gaze_element, gt))
    
    result_array[6] += np.mean(distance['SS'])   
    result_array[7] += np.nanmean(distance['VectorSimilarity']) 
    result_array[8] += np.nanmean(distance['DirectionSimilarity']) 
    result_array[9] += np.nanmean(distance['LengthSimilarity']) 
    result_array[10] += np.nanmean(distance['PositionSimilarity']) 
    result_array[11] += np.nanmean(distance['Average'])
    result_array[12] += np.mean(ED)
        


class Evaluation(object):
    def __init__(self, training_dataset_choice, testing_dataset_choice, evaluation_url,
                 datapath, indexFile, ITERATION=100, showBenchmark=True, showExpected=True, leave_one_comb_out=0):
        #gaze_tf = '../dataset/checkEvaluation/gaze_tf.csv'
        self.ITERATION = ITERATION
        self.showBenchmark = showBenchmark
        self.training_dataset_choice = training_dataset_choice
        self.testing_dataset_choice = testing_dataset_choice
        gaze_gt = evaluation_url+'/gaze_gt.csv'
        gaze_max = evaluation_url+'/gaze_max.csv'
        gaze_expect = evaluation_url+'/gaze_expect.csv'
        if showBenchmark:
            gaze_random = './dataset/checkEvaluation/gaze_random.csv'
            gaze_saliency = './dataset/checkEvaluation/gaze_saliency.csv'
            gaze_rgb = './dataset/checkEvaluation/gaze_rgb_similarity.csv'
            gaze_center = './dataset/checkEvaluation/gaze_center.csv'
        
        #raw_data = randsplit(datapath, indexFile, 'Test', testing_dataset_choice, training_dataset_choice)

        if leave_one_comb_out == 0:
            raw_data = randsplit(datapath, indexFile, 'Test', testing_dataset_choice, training_dataset_choice)
        else:
            raw_data = randsplit_comb(datapath, indexFile, 'Test', testing_dataset_choice, training_dataset_choice,
                                      'Q3_10', 'T3_12')

        self.data_length = len(raw_data)
        print(F'len = {self.data_length}')
        self.target = []
        self.ids = []

        for item in raw_data:
            self.target.append(item['package_target'])
            self.ids.append(item['id'])

        self.target = [int(self.target[i][0])-1 for i in range(len(self.target))]
  
        self.gaze_gt = np.array(pd.read_csv(gaze_gt))
        self.gaze_max = np.array(pd.read_csv(gaze_max))
        if showExpected:
            self.gaze_expect = np.array(pd.read_csv(gaze_expect))
        self.showExpected = showExpected
        if showBenchmark:
            self.gaze_random = np.array(pd.read_csv(gaze_random))
            self.gaze_saliency = np.array(pd.read_csv(gaze_saliency))
            self.gaze_rgb = np.array(pd.read_csv(gaze_rgb))
            self.gaze_center = np.array(pd.read_csv(gaze_center))


    def evaluation(self):
        # 7 stands for: correct target, avg.length, avg.search, avg.refix, avg.revisit, distance, heatmap overlapping
        res = {'gt': torch.zeros(13), 'random': torch.zeros(13), 'center': torch.zeros(13),
               'rgb': torch.zeros(13), 'saliency': torch.zeros(13), #'tf': torch.zeros(5),
               'single': torch.zeros(13), 'multi': torch.zeros(13)}

        for i in tqdm(range(self.data_length)):
            if self.training_dataset_choice == self.testing_dataset_choice and self.testing_dataset_choice != 'all':
                if self.testing_dataset_choice == 'wine':
                    TOTAL_PCK = 22
                    col_num = 11
                    row_num = 2
                elif self.testing_dataset_choice == 'yogurt':
                    TOTAL_PCK = 27
                    col_num = 9
                    row_num = 3
                elif self.testing_dataset_choice == 'amazon':
                    TOTAL_PCK = 84
                    col_num = 14
                    row_num = 6
            elif self.training_dataset_choice == self.testing_dataset_choice == 'all':
                if self.ids[i] == 'Q1':
                    TOTAL_PCK = 22
                    col_num = 11
                    row_num = 2
                elif self.ids[i] == 'Q3':
                    TOTAL_PCK = 27
                    col_num = 9
                    row_num = 3
            else:
                print('not implemented')
                quit()
                
            behavior(res['gt'], self.target[i], self.gaze_gt[i:(i+1)])
            behavior(res['single'], self.target[i], self.gaze_max[i:(i + 1)])
            string_distance(res['single'],self.gaze_max[i:(i + 1)],self.gaze_gt[i:(i+1)],col_num,row_num)
            string_distance(res['gt'],self.gaze_gt[i:(i+1)],self.gaze_gt[i:(i+1)],col_num,row_num)
            if self.showExpected:
                behavior(res['multi'], self.target[i],
                         self.gaze_expect[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)])
                string_distance(res['multi'],self.gaze_expect[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)],self.gaze_gt[i:(i+1)],col_num,row_num)
            if self.showBenchmark:
                behavior(res['random'], self.target[i],self.gaze_random[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)],benchmark=True)
                behavior(res['center'], self.target[i], self.gaze_center[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)],benchmark=True)
                behavior(res['rgb'], self.target[i], self.gaze_rgb[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)],benchmark=True)
                behavior(res['saliency'], self.target[i], self.gaze_saliency[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)],benchmark=True)
                string_distance(res['random'],self.gaze_random[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)],self.gaze_gt[i:(i+1)],col_num,row_num)
                string_distance(res['center'],self.gaze_center[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)],self.gaze_gt[i:(i+1)],col_num,row_num)
                string_distance(res['rgb'],self.gaze_rgb[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)],self.gaze_gt[i:(i+1)],col_num,row_num)
                string_distance(res['saliency'],self.gaze_saliency[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)],self.gaze_gt[i:(i+1)],col_num,row_num)
                
        res['gt'] = res['gt'] / self.data_length
        res['single'] = res['single'] / self.data_length
        res['multi'] = res['multi'] / self.data_length 
        if self.showBenchmark:
            res['random'] = res['random'] / self.data_length
            res['center'] = res['center'] / self.data_length 
            res['rgb'] = res['rgb'] / self.data_length 
            res['saliency'] = res['saliency'] / self.data_length

        print('*'*20)
        print('correct Target \t avg.len \t avg.search \t avg.refix \t avg.revisit \t delta \t SS \t VecSim \t DirSim \t LenSim \t PosSim \t AvgMM')
        models = ['gt', 'single', 'multi'] 
        if self.showBenchmark:
            models.extend(['random','center', 'saliency', 'rgb'])

        for i in models:
            a=torch.abs(res[i][:5] - res['gt'][:5])
            b=res['gt'][:5]
            c=torch.nansum(a / b)
            d=c/5
            res[i][5] = d
            print(i, ': ', res[i])
        print('*' * 20)


if __name__ == '__main__':
    ITERATION = 100
    training_dataset_choice = 'amazon'
    testing_dataset_choice = 'amazon'
    datapath = './dataset/processdata/dataset_amazon'
    indexFile = './dataset/processdata/splitlist_all_amazon.txt'
    '''training_dataset_choice = 'all'
    testing_dataset_choice = 'all'
    datapath = './dataset/processdata/dataset_Q123_mousedel_time'
    indexFile = './dataset/processdata/splitlist_all_time.txt'''
    evaluation_url = './dataset/checkEvaluation/amazon_center'

    e = Evaluation(training_dataset_choice, testing_dataset_choice, evaluation_url,
                 datapath, indexFile, ITERATION=100, showBenchmark=False, showExpected=True,
                   leave_one_comb_out=0)
    e.evaluation()
