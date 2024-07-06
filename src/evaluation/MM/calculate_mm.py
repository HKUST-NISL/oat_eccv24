import numpy as np
from src.evaluation.MM.multimatch import docomparison
from src.dataBuilders.data_builder import randsplit
import pandas as pd
from tqdm import tqdm
import json


def compute_nss(s_map,gt):
    xy = np.where(gt==1)
    s_map_norm = (s_map - np.mean(s_map))/np.std(s_map)
    return np.mean(s_map_norm[xy])


def multimatch(s1, s2, im_size):
    s1x = s1['X']
    s1y = s1['Y']
    l1 = len(s1x)
    if l1 < 3:
        scanpath1 = np.ones((3, 3), dtype=np.float32)
        scanpath1[:l1, 0] = s1x
        scanpath1[:l1, 1] = s1y
    else:
        scanpath1 = np.ones((l1, 3), dtype=np.float32)
        scanpath1[:, 0] = s1x
        scanpath1[:, 1] = s1y
    s2x = s2['X']
    s2y = s2['Y']
    l2 = len(s2x)
    if l2 < 3:
        scanpath2 = np.ones((3, 3), dtype=np.float32)
        scanpath2[:l2, 0] = s2x
        scanpath2[:l2, 1] = s2y
    else:
        scanpath2 = np.ones((l2, 3), dtype=np.float32)
        scanpath2[:, 0] = s2x
        scanpath2[:, 1] = s2y
    mm = docomparison(scanpath1, scanpath2, sz=im_size)
    return mm[0]


class Evaluation(object):
    def __init__(self, training_dataset_choice, testing_dataset_choice, evaluation_url,
                 datapath, indexFile, ITERATION, patch):
        #gaze_tf = '../dataset/checkEvaluation/gaze_tf.csv'
        self.ITERATION = ITERATION
        self.training_dataset_choice = training_dataset_choice
        self.testing_dataset_choice = testing_dataset_choice
        raw_data = randsplit(datapath, indexFile, 'Test', testing_dataset_choice, training_dataset_choice)

        self.data_length = len(raw_data)
        print(F'len = {self.data_length}')
        self.patch = patch
        if testing_dataset_choice != 'amazon':
            self.im_size = [1680, 1050]
        else:
            json_file = 'dataset/amazon_data/amazon_data.json'
            self.amazon_raw = json.load(open(json_file))
            self.im_size = []

        self.gt_fixations = []
        self.id = []
        self.gaze_expect = np.array(pd.read_csv(evaluation_url))

        #res = [0, 0]
        for item in raw_data:
            self.gt_fixations.append({'X': item['X'], 'Y': item['Y']})
            self.id.append(item['id'])
            if testing_dataset_choice == 'amazon':
                size = self.amazon_raw[item['layout_id']+'.jpg']['size']
                self.im_size.append([size['width'], size['height']])

                #res[0] += size['width']
                #res[1] += size['height']
        '''print(res[0]/len(raw_data)) #2554, 182。57
        print(res[1] / len(raw_data)) #1600, 266。67
        quit()'''

    def evaluation(self):
        final_res = [[], [], [], []]
        nss = []
        for i in tqdm(range(self.data_length)):
            if self.testing_dataset_choice == 'amazon':
                im_size = self.im_size[i]
            else:
                im_size = self.im_size
            gazehat = self.gaze_expect[(i * self.ITERATION):(i * self.ITERATION + self.ITERATION)]
            gazegt = self.gt_fixations[i]
            predicted_map = np.zeros((im_size))
            gt_map = np.zeros((im_size))
            for x,y in zip(gazegt['X'], gazegt['Y']):
                gt_map[round(x),round(y)]=1
            #print(gt_map.sum())

            for j in range(len(gazehat)):
                if self.patch:
                    gaze_element = gazehat[j][~np.isnan(gazehat[j])]
                    if self.id[i] == 'Q1':
                        pac_num = [2, 11]
                        pac_size = [449, 152]
                        title = 152
                    elif self.id[i] == 'Q3':
                        pac_num = [3, 9]
                        pac_size = [305, 186]
                        title = 135
                    elif self.id[i] == 'amazon':
                        pac_num = [6, 14]
                        pac_size = [267, 183]
                        title = 0
                    x = gaze_element % pac_num[1]
                    y = gaze_element // pac_num[1]
                    X = x * pac_size[1] + pac_size[1] / 2
                    #if self.reverseY:
                    #    y_ = pac_num[0] - 1 - y
                    #    Y = y_ * pac_size[0]
                    #else:
                    Y = title + y * pac_size[0] + pac_size[0] / 2
                    gaze_element = {'X': X, 'Y':Y}
                else:
                    X = gazehat[j][0][1:-1]
                    Y = gazehat[j][1][1:-1]
                    X1 = X.split(' ')
                    X1 = [float(a) for a in X1 if a!='']
                    Y1 = Y.split(' ')
                    Y1 = [float(a) for a in Y1 if a != '']
                    gaze_element = {'X': X1, 'Y': Y1}

                mmres = multimatch(gazegt, gaze_element, im_size)
                final_res[0].append(mmres[0])
                final_res[1].append(mmres[1])
                final_res[2].append(mmres[2])
                final_res[3].append(mmres[3])

                for x, y in zip(gaze_element['X'], gaze_element['Y']):
                    #print(gaze_element)
                    x1=min(round(x), im_size[0]-1)
                    y1=min(round(y), im_size[1]-1)
                    predicted_map[x1, y1] = 1
                #print(predicted_map.sum())
            nss_ = compute_nss(predicted_map, gt_map)
            nss.append(nss_)
        a=np.mean(final_res[0])
        b=np.mean(final_res[1])
        c=np.mean(final_res[2])
        d=np.mean(final_res[3])
        nss_avg = np.mean(nss)
        print('Vector similarity = ', np.mean(final_res[0]))
        print('Direction similarity = ', np.mean(final_res[1]))
        print('Length similarity = ', np.mean(final_res[2]))
        print('Position similarity = ', np.mean(final_res[3]))
        print('Avg: ', (a+b+c+d)/4)
        print('NSS: ', nss_avg)


if __name__ == '__main__':
    ITERATION = 100
    training_dataset_choice = 'amazon'
    testing_dataset_choice = 'amazon'
    datapath = './dataset/processdata/dataset_amazon'
    indexFile = './dataset/processdata/splitlist_all_amazon.txt'
    '''training_dataset_choice = 'all'
    testing_dataset_choice = 'all'
    datapath = './dataset/processdata/dataset_Q123_mousedel_time_new'
    indexFile = './dataset/processdata/splitlist_all_time.txt'''

    evaluation_url = '/Users/adia/Documents/HKUST/projects/gazePrediction/trajectory-transformer/dataset/checkEvaluation/amazon_center/gaze_expect.csv'
    patch = True

    e = Evaluation(training_dataset_choice, testing_dataset_choice, evaluation_url,
                 datapath, indexFile, ITERATION, patch)
    e.evaluation()
