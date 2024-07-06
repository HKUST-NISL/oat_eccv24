#import torch
#import torchvision.transforms as transforms
from cgi import test
import pandas as pd
import numpy as np
import cv2 as cv
from collections import Counter
import pickle
from tqdm import tqdm
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity
import json
from tqdm import tqdm
import matplotlib.pyplot as plt


class CUT_PIC_AMAZON(object):
    def __init__(self, file_name, output_path):
        self.image_path = file_name + 'amazon_data' + '/images/'
        self.border_path = file_name + 'amazon_data' + '/binarymask/'
        self.data_path = file_name + 'amazon_data' + '/amazon_data.json'
        self.target_path = file_name + 'amazon_data' + '/target/'
        self.dim = (93, 150)
        self.output_path = output_path

    def find_bounding_boxes(self, border_img):
        gray_image = cv.cvtColor(border_img, cv.COLOR_BGR2GRAY)
        _, binary_image = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            bounding_boxes.append((x, y, x+w, y+h))
        sorted_bounding_boxes = self.sort_boxes(bounding_boxes)
        return sorted_bounding_boxes

    # define a function to sort the bounding boxes from left to right, top to bottom
    def sort_boxes(self, boxes, y_threshold=10):
        def normalize_y(ymin, threshold):
            # Normalize ymin values within a certain threshold
            return ymin // threshold * threshold

        sorted_boxes = sorted(boxes, key=lambda box: (normalize_y(box[1], y_threshold), box[0]))
        return sorted_boxes

    def crop_images_to_borders(self, target_img, border_img):
        bounding_boxes = self.find_bounding_boxes(border_img)
        cropped_images = []
        for i, (x1, y1, x2, y2) in enumerate(bounding_boxes):
            cropped_image = target_img[y1:y2, x1:x2]
            img_cropped_feature = cv.resize(cropped_image, self.dim)
            cropped_images.append(img_cropped_feature)
        return cropped_images


    def is_point_in_box(self, point, bbox):
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def distance(self, point1, point2):
            return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    def center_of_box(self, bbox):
            return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def calculate_iou(self, box1, box2):
        x_left = max(box1[0]['xmin'], box2[0])
        y_top = max(box1[0]["ymin"], box2[1])
        x_right = min(box1[0]['xmax'], box2[2])
        y_bottom = min(box1[0]['ymax'], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        box1_area = (box1[0]['xmax'] - box1[0]['xmin']) * (box1[0]['ymax'] - box1[0]["ymin"])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - intersection_area

        iou = intersection_area / union_area
        return iou

    def find_best_matching_box_index(self, ground_truth, border_img):
        best_iou = 0
        best_match_index = -1
        boxes = self.find_bounding_boxes(border_img)
        for index, box in enumerate(boxes):
            iou = self.calculate_iou(ground_truth, box)
            if iou > best_iou:
                best_iou = iou
                best_match_index = index + 1
        return best_match_index

    def fixation_to_patch(self,fixations, border_img):
        bounding_boxes = self.find_bounding_boxes(border_img)
        fixations_patch = []

        for fixation in fixations:
            nearest_bbox_index = None
            min_distance = float('inf')

            for index, bbox in enumerate(bounding_boxes):
                if self.is_point_in_box(fixation, bbox):
                    nearest_bbox_index = index
                    break
                else:
                    bbox_center = self.center_of_box(bbox)
                    current_distance = self.distance(fixation, bbox_center)
                    if current_distance < min_distance:
                        min_distance = current_distance
                        nearest_bbox_index = index
            fixations_patch.append(nearest_bbox_index + 1)
        return fixations_patch

    def cut_image(self):
        dataset = []
        with open(self.data_path, 'r') as file:
            data = json.load(file)
        seq_len = []

        for imageName, gaze in tqdm(data.items(), desc='Processing gsaze data'):
            img = cv.imread(self.image_path + imageName)
            border_img = cv.imread(self.border_path + '14x6-' + imageName)
            cropped_images = self.crop_images_to_borders(img, border_img)
            target_id = self.find_best_matching_box_index(gaze['ground_truth'], border_img)
            dataset_dict = {}
            for sub, fixation in gaze['fixations'].items():
                fixations_patch = self.fixation_to_patch(fixation, border_img)
                dataset_dict['package_target'] = [target_id]
                dataset_dict['package_seq'] = fixations_patch
                seq_len.append(len(fixations_patch))
                dataset_dict['X'] = [x for x, _ in fixation]
                dataset_dict['Y'] = [y for _, y in fixation]
                dataset_dict['question_img_feature'] = cropped_images
                dataset_dict['id'] = 'amazon'
                dataset_dict['tgt_id'] = gaze['target'][7]
                dataset_dict['pair'] = imageName.split('.')[0] + '_' + gaze['target'][7]
                dataset_dict['layout_id'] = imageName.split('.')[0]
                dataset_dict['sub_id'] = sub
                dataset.append(dataset_dict)

        print('total len=', len(dataset))
        print('avg len=', np.mean(seq_len))
        #plt.hist(seq_len)
        #plt.show()
        avg = np.mean(seq_len)
        std = np.std(seq_len)
        minlen = avg - 3 * std
        maxlen = avg + 3 * std
        final_dataset = []

        new_len = []
        for data in dataset:
            lendata = len(data['package_seq'])
            if lendata <= maxlen and lendata >= minlen:
                final_dataset.append(data)
                new_len.append(lendata)

        print('total len=', len(final_dataset))
        print('avg len=', np.mean(new_len))
        #plt.hist(new_len)
        #plt.xlabel('Seq length')
        #plt.show()

        with open(self.output_path, "wb") as fp:  # Pickling
            pickle.dump(final_dataset, fp)

        print("Finish...")


if __name__ == '__main__':
    CUT_PIC = CUT_PIC_AMAZON("./dataset/", "./dataset/processdata/dataset_amazon")
    CUT_PIC.cut_image()