import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from itertools import groupby
#from dictances import bhattacharyya
from scipy.stats import wasserstein_distance
import sys
sys.path.append('./src/')
from dataBuilders.data_builder import randsplit
from tqdm import tqdm

def behavior(result_array, target, gaze):
    for i in range(len(gaze)):
        if len(gaze[i]) == 0:
            print('GAZE LENGTH IS ZERO')
            continue
        gaze_element = gaze[i][~np.isnan(gaze[i])]
        if len(gaze_element) == 0:
            print('replacing it...')
            gaze_element = gaze[i-1][~np.isnan(gaze[i-1])]
        result_array[0] += int(target == gaze_element[-1])
        result_array[1] += len(gaze_element)
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
        result_array[2] += (search_len / len(gaze_element))
        result_array[3] += (refix_len / len(gaze_element))
        result_array[4] += (revisit_len / len(gaze_element))


class Evaluation(object):
    def __init__(self, training_dataset_choice, testing_dataset_choice, evaluation_url,
                 datapath, indexFile, ITERATION=100, hasExpectedFile=True):
        #gaze_tf = '../dataset/checkEvaluation/gaze_tf.csv'
        self.ITERATION = ITERATION
        index_folder = './dataset/processdata/'
        gaze_gt = evaluation_url+'/gaze_gt.csv'
        gaze_max = evaluation_url+'/gaze_max.csv'
        gaze_expect = evaluation_url+'/gaze_expect.csv'

        #datapath = './dataset/processdata/dataset_Q123_mousedel_time'
        #indexFile = './dataset/processdata/splitlist_all_time.txt'
        if testing_dataset_choice == 'irregular':
            with open(datapath, "rb") as fp:
                raw_data = pickle.load(fp)
        else:
            raw_data = randsplit(datapath, indexFile, 'Test', testing_dataset_choice, training_dataset_choice)

        self.data_length = len(raw_data)
        print(F'len = {self.data_length}')
        self.target = []

        for item in raw_data:
            self.target.append(item['package_target'])

        self.target = [int(self.target[i][0])-1 for i in range(len(self.target))]

        self.gaze_gt = np.array(pd.read_csv(gaze_gt))
        #self.gaze_tf = np.array(pd.read_csv(gaze_tf))
        self.gaze_max = np.array(pd.read_csv(gaze_max))
        self.hasExpectedFile = hasExpectedFile
        if hasExpectedFile:
            self.gaze_expect = np.array(pd.read_csv(gaze_expect))

    def evaluation(self):
        # 7 stands for: correct target, avg.length, avg.search, avg.refix, avg.revisit, distance, heatmap overlapping
        res = {'gt': torch.zeros(7), 'random': torch.zeros(7), 'resnet': torch.zeros(7),
               'rgb': torch.zeros(7), 'saliency': torch.zeros(7), #'tf': torch.zeros(5),
               'single': torch.zeros(7), 'multi': torch.zeros(7)}

        for i in tqdm(range(self.data_length)):
            behavior(res['gt'], self.target[i], self.gaze_gt[i:(i+1)])
            behavior(res['single'], self.target[i], self.gaze_max[i:(i + 1)])
            if self.hasExpectedFile:
                behavior(res['multi'], self.target[i], self.gaze_expect[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)])

        res['gt'] = res['gt'] / self.data_length
        res['single'] = res['single'] / self.data_length
        res['multi'] = res['multi'] / self.data_length / self.ITERATION

        print('*'*20)
        print('correct Target \t avg.len \t avg.search \t avg.refix \t avg.revisit \t overlap \t delta')
        models = ['gt', 'single', 'multi']
        for i in models:
            res[i][6] = torch.sum(torch.abs(res[i][:5] - res['gt'][:5]) / res['gt'][:5]) / 5
            print(i, ': ', res[i])
        print('*' * 20)


if __name__ == '__main__':
    ITERATION = 100
    hasExpectedFile = True
    training_dataset_choice = 'amazon'
    testing_dataset_choice = 'amazon'
    datapath = './dataset/processdata/dataset_amazon'
    indexFile = './dataset/processdata/splitlist_all_amazon.txt'

    evaluation_url = './dataset/checkEvaluation/amazon_random'

    e = Evaluation(training_dataset_choice, testing_dataset_choice, evaluation_url,
                    datapath, indexFile)
    e.evaluation()

