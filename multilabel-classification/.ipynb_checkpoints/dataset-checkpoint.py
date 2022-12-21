import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import os
import math

class ImageDataset(Dataset):
    def __init__(self, csv_path, df_all_concepts=None, transform=None, image_dir="/scratch/rm5707/roco-dataset-master/data", sub_dataset='train'):
        self.csv_path = csv_path
        self.csv = pd.read_csv(self.csv_path, header=0, sep=",")
        self.df_all_concepts = pd.read_csv(df_all_concepts, sep=",") if df_all_concepts else None
        self.image_names = self.csv[:]['ID']
        self.transform = transform
        self.image_dir = image_dir
        self.sub_dataset = sub_dataset

        # Preprocessing labels
        self.base_dir = os.path.dirname(self.csv_path)
       
        if(self.df_all_concepts is not None):
            all_concepts = self.df_all_concepts["concept"]
            dict_concept = dict()
            for idx, c in enumerate(all_concepts):
                dict_concept[c] = idx

            matrix = np.zeros((len(self.csv["ID"]), len(all_concepts)))
            count_empty = 0
            for i in range(len(self.csv["ID"])):
                if( not (self.csv["cuis"][i] == "" or pd.isnull(self.csv["cuis"][i]))):
                    #print(i, " ", self.csv["cuis"][i])
                    dict_concepts_per_image = self.csv["cuis"][i].split(";")
                    for c in dict_concepts_per_image:
                        matrix[i][dict_concept[c]] = 1
                else:
                    count_empty = count_empty + 1
            print("Total [", count_empty, "] images with no CUIds.")
                    

            self.labels = matrix
            print(self.labels.shape)
        else:
            self.labels = None

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
#         if self.sub_dataset == 'train':
#             image = Image.open(os.path.join(self.image_dir, self.sub_dataset, "radiology/images", self.image_names[index] + '.jpg')).convert("RGB")
#         elif self.sub_dataset == 'test':
#             image = Image.open(os.path.join(self.image_dir, self.sub_dataset, "radiology/images", self.image_names[index] + '.jpg')).convert("RGB")
#         else:
#             image = Image.open(os.path.join(self.image_dir, self.sub_dataset, "radiology/images", self.image_names[index] + '.jpg')).convert("RGB")
        image = Image.open(os.path.join(self.image_dir, self.image_names[index] + '.jpg')).convert("RGB")
        
        # apply image transforms
        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            targets = self.labels[index]
#             print("Image Name   : ", self.image_names[index])
#             print("Image Labels : ", torch.tensor(targets, dtype=torch.float32))

            return {
                'image': image,
                'label': torch.tensor(targets, dtype=torch.float32),
                'image_name': self.image_names[index]
            }
        else:
            return {
                'image': image
            }