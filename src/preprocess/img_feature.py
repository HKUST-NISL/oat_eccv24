#import torch
#import torchvision.transforms as transforms
from cgi import test
import pandas as pd
import numpy as np
import cv2 as cv
from collections import Counter
import pickle
from tqdm import tqdm

img_dir = './dataset/img/Question/'
data_dir0 = '/Users/adia/Documents/HKUST/projects/gazePrediction/Gazeformer-main/pamformer/data with coordinate/time_Q1_mousedel_new.xlsx'
data_dir1 = '/Users/adia/Documents/HKUST/projects/gazePrediction/Gazeformer-main/pamformer/data with coordinate/time_Q2_mousedel_new.xlsx'
data_dir2 = '/Users/adia/Documents/HKUST/projects/gazePrediction/Gazeformer-main/pamformer/data with coordinate/time_Q3_mousedel_new.xlsx'
target_dir = './dataset/img/Target/'


class CUT_PIC(object):
    def __init__(self, wine_choice, file_name):
            self.wine_choice = wine_choice  # pad or resize
            self.file_name = file_name

    def read_excel(self, file_name):
            xls = pd.ExcelFile(file_name)
            self.df = pd.read_excel(xls)

    def cut_pic(self):
        dataset = []
        df_data0 = pd.read_excel(data_dir0)
        df_data1 = pd.read_excel(data_dir1)
        df_data2 = pd.read_excel(data_dir2)
        words0 = [str(item) for item in list(df_data0["ID"])]
        words1 = [str(item) for item in list(df_data1["ID"])]
        words2 = [str(item) for item in list(df_data2["ID"])]

        word_dict_sorted0 = Counter(words0)
        word_dict_sorted1 = Counter(words1)
        word_dict_sorted2 = Counter(words2)

        for i in range(3):
            if i==0:
                dict = word_dict_sorted0
                df_ori = df_data0
                print('Q1 size: ', len(dict))

            elif i==1:
                dict = word_dict_sorted1
                df_ori = df_data1
                print('Q2 size: ', len(dict))

            elif i==2:
                dict = word_dict_sorted2
                df_ori = df_data2
                print('Q3 size: ', len(dict))

            for key in tqdm(dict):
                df1 = df_ori[df_ori["ID"]==int(key)]
                df1.reset_index(drop=True, inplace=True)
                tgt_package = df1["Target"]
                question_name = list(df1["Question"])[0]
                package_seq = list(df1["Choice"])
                X = list(df1["X"])
                Y = list(df1["Y"])
                package_target = list(df1["T_Package"])[0]
                package_target  = [package_target]  # (np.repeat(package_target,27)) 
                sub_id = list(df1["Sub_ID"])[0]
                dataset_dict = {}
                Question_img_feature = []
                if question_name.startswith('Q1'):
                    IMAGE_SIZE_1 = 449
                    IMAGE_SIZE_2 = 152
                    IMAGE_ROW = 2
                    IMAGE_COLUMN = 11
                    CROP_RANGE_1 = 389
                    CROP_RANGE_2 = 106 
                    #dim = (106, 390)
                    if self.wine_choice == 'resize':
                        dim = (93, 150)
                    elif self.wine_choice == 'pad':
                        dim = (51, 150)
                    elif self.wine_choice == 'raw':
                        dim = (53, 195)

                elif question_name.startswith('Q2'):
                    IMAGE_SIZE_1 = 295
                    IMAGE_SIZE_2 = 186
                    IMAGE_ROW = 3
                    IMAGE_COLUMN = 9
                    CROP_RANGE_1 = 239
                    CROP_RANGE_2 = 116
                    dim = (93, 150)

                elif question_name.startswith('Q3'):
                    IMAGE_SIZE_1 = 305
                    IMAGE_SIZE_2 = 186
                    IMAGE_ROW = 3
                    IMAGE_COLUMN = 9
                    CROP_RANGE_1 = 245
                    CROP_RANGE_2 = 162
                    dim = (93, 150)
                
                question_img = cv.imread(img_dir + question_name + '.png')
                question_img = cv.cvtColor(question_img,cv.COLOR_BGR2RGB)

                for y in range(1, IMAGE_ROW + 1):
                    for x in range(1, IMAGE_COLUMN + 1):
                        img_cropped_feature = question_img[((1050-IMAGE_SIZE_1*IMAGE_ROW)+(y-1)*IMAGE_SIZE_1):((1050-IMAGE_SIZE_1*IMAGE_ROW)+y*IMAGE_SIZE_1), ((x-1)*IMAGE_SIZE_2):x*IMAGE_SIZE_2]
                        # img_cropped_feature = img_cropped_feature[int((IMAGE_SIZE_1 - CROP_RANGE_1) / 2) : int(IMAGE_SIZE_1 - (IMAGE_SIZE_1 - CROP_RANGE_1) / 2), int((IMAGE_SIZE_2 - CROP_RANGE_2) / 2) : int(IMAGE_SIZE_2 - (IMAGE_SIZE_2 - CROP_RANGE_2)/2)]
                        img_cropped_feature = cv.resize(img_cropped_feature, dim)
                        if self.wine_choice == 'pad' and question_name.startswith('Q1'):
                            img_cropped_feature = cv.copyMakeBorder(img_cropped_feature, 0, 0, 21, 21, cv.BORDER_CONSTANT, 255)

                        img_cropped_feature = cv.normalize(img_cropped_feature, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
                        Question_img_feature.append(img_cropped_feature)
                        
                dataset_dict['package_target'] = package_target
                dataset_dict['package_seq'] = package_seq
                dataset_dict['X'] = X
                dataset_dict['Y'] = Y
                dataset_dict['question_img_feature'] = Question_img_feature
                dataset_dict['id'] = question_name[:2]
                dataset_dict['tgt_id'] = tgt_package[0]
                dataset_dict['pair'] = tgt_package[0] + question_name
                dataset_dict['layout_id'] = question_name
                dataset_dict['sub_id'] = sub_id

                dataset.append(dataset_dict)

        with open(self.file_name, "wb") as fp:  # Pickling
            pickle.dump(dataset, fp)

        print("Finish...")
    

if __name__ == '__main__':
    CUT_PIC = CUT_PIC('resize', "./dataset/processdata/dataset_Q123_mousedel_time_new")
    CUT_PIC.cut_pic()
    # results: Q2 size 453, Q3 size 439




