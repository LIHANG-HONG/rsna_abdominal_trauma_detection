import json
import os.path
import random

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from albumentations import ReplayCompose
from torch.utils.data import Dataset

class ClassifierDatasetCropsFullResLiverSpleenKidney(Dataset):
    def __init__(
            self,
            mode: str,
            dataset_dir: str,
            fold: int,
            transforms: A.Compose,
            slice_size: int,
            crop_size: int,
            multiplier: int = 1,
            folds_csv="folds.csv",
            metadata="meta.json",
    ):
        df = pd.read_csv(os.path.join(folds_csv))
        if mode == "train":
            self.df = df[df['split'] != fold]
        else:
            self.df = df[(df['split'] == fold)]
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.slice_size = slice_size
        self.crop_size = crop_size

        if self.mode == "train":
            self.df = pd.concat([self.df] * multiplier)
        if self.is_train:
            assert len(self.df[self.df['split'] == fold]) == 0
        else:
            assert len(self.df[self.df['split'] != fold]) == 0
        with open(metadata, "r") as f:
            self.metadata = json.load(f)
        self.cache = {}

    def __getitem__(self, i):
        return self.getitem(i)

    def find_or_box(self,image_cube,boxes,row):

        if self.is_train:
            injured = []
            non_injured = []
            for li in [1,2,3]:
                if li in boxes and boxes[li][1] > 512:
                    flag = 0
                    for l in self.parts_map[li]:
                        flag += row[l]
                    if flag>0:
                        injured.append(li)
                    else:
                        non_injured.append(li)

            if len(injured)>0 and (not self.is_train or random.random() < 0.7):
                li = random.choice(injured)
                bbox, area = boxes[li]
            elif len(non_injured)>0:
                li = random.choice(non_injured)
                bbox, area = boxes[li]
            elif len(injured)>0:
                li = random.choice(injured)
                bbox, area = boxes[li]
            else:
                bbox = None
                li = -1
            return bbox, li
            
        else:
            pass
          
        return bbox, liver_found, spleen_found, kidney_found, found

    def process_images(self,images,masks,image_mean,image_std,slice_size):
        replay = None
        image_crops = []
        mask_crops = []
        for i in range(images.shape[0]):
            image = images[i]
            mask = masks[i]
            if replay is None:
                sample = self.transforms(image=image, mask=mask)
                replay = sample["replay"]
            else:
                sample = ReplayCompose.replay(replay, image=image, mask=mask)
            image_ = sample["image"]
            image_crops.append(image_)
            mask_crops.append(sample["mask"])
        images = np.array(image_crops).astype(np.float32)
        masks = np.array(mask_crops).astype(np.float32)


        images = np.expand_dims(images, -1)
        masks = np.expand_dims(masks, -1)

        images = (images - image_mean) / image_std


        images = np.concatenate([images, images, masks], axis=-1)

        h=images.shape[0]
        if h != slice_size:
            tmp = np.zeros((slice_size, *images.shape[1:]))
            tmp[:h] = images
            images = tmp

        images = np.moveaxis(images, -1, 0)
        return images

    def getitem(self, i):
        row = self.df.iloc[i]
        series = row.series.split('|')
        cube_id = random.choice(series)

        mask_cube = nib.load(os.path.join(self.dataset_dir, f"seg", f"{cube_id}.nii.gz")).get_fdata()
        mask_cube = mask_cube.transpose((2,1,0))
        mask_cube[mask_cube==4] = 3
        mask_cube[mask_cube==5] = 0
        image_cube = nib.load(os.path.join(self.dataset_dir, f"cube", f"{cube_id}.nii.gz")).get_fdata()
        image_cube = image_cube.transpose((2,1,0))


        meta = self.metadata[cube_id]
        image_mean = meta["image_mean"]
        image_std = meta["image_std"]
        sums = meta["sums"]
        boxes = meta["boxes"]
        boxes = {int(k):v for k, v in boxes.items()}
        slice_size = self.slice_size

        if self.is_train:
            bbox, li = self.find_or_box(image_cube,boxes,row)

            if bbox is not None:
                z1, z2 = bbox[0], bbox[3]
                y1, y2 = max(bbox[1] - 16, 0), min(bbox[4] + 16, 256)
                x1, x2 = max(bbox[2] - 16, 0), min(bbox[5] + 16, 256)
                images = image_cube[z1:z2, y1:y2, x1:x2].copy()
                masks = mask_cube[z1:z2, y1:y2, x1:x2].copy()

                area_proportions = [0]
                for mi in [1,2,3]:
                    sum = sums[mi]
                    crop_sum = int((masks == mi).sum())
                    if sum!=0:
                        area_proportions.append(crop_sum/sum)
                    else:
                        area_proportions.append(0.0)
                labels = {
                        #'bowel_healthy':row['bowel_healthy'],
                        #'bowel_injury':row['bowel_injury'],
                        #'extravasation_healthy':row['extravasation_healthy'],
                        #'extravasation_injury':row['extravasation_injury'],
                        #'kidney_healthy':row['kidney_healthy']
                        'kidney_low':row['kidney_low'] * area_proportions[3],
                        'kidney_high':row['kidney_high'] * area_proportions[3],
                        #'liver_healthy':row['liver_healthy']
                        'liver_low':row['liver_low'] * area_proportions[1],
                        'liver_high':row['liver_high'] * area_proportions[1],
                        #'spleen_healthy':row['spleen_healthy']
                        'spleen_low':row['spleen_low'] * area_proportions[2],
                        'spleen_high':row['spleen_high'] * area_proportions[2],
                        #'any_injury':row['any_injury']
                }
                labels = np.array(list(labels.values())).astype(np.float32)


                subsample_num = int((z2-z1)/slice_size) if (z2-z1)%slice_size==0 else int((z2-z1)/slice_size)+1


                subsample_id=random.randint(0, subsample_num-1)
                images = images[subsample_id::subsample_num]
                masks = masks[subsample_id::subsample_num]

                images = self.process_images(images,masks,image_mean,image_std,slice_size)
            else:
                images = np.zeros((3, self.slice_size, self.crop_size, self.crop_size))
                labels = {
                        #'bowel_healthy':row['bowel_healthy'],
                        #'bowel_injury':row['bowel_injury'],
                        #'extravasation_healthy':row['extravasation_healthy'],
                        #'extravasation_injury':row['extravasation_injury'],
                        #'kidney_healthy':row['kidney_healthy']
                        'kidney_low':0,
                        'kidney_high':0,
                        #'liver_healthy':row['liver_healthy']
                        'liver_low':0,
                        'liver_high':0,
                        #'spleen_healthy':row['spleen_healthy']
                        'spleen_low':0,
                        'spleen_high':0,
                        #'any_injury':row['any_injury']
                }
                labels = np.array(list(labels.values())).astype(np.float32)
            sample = {}

            sample['image'] = torch.from_numpy(images).float()
            sample['label'] = torch.from_numpy(labels).float()
            sample['cube_id'] = cube_id
        else:
            all_images = []

            for li in [1,2,3]:
                if li not in boxes:
                    #print("missing")
                    all_images.append(np.zeros((3, self.slice_size, self.crop_size, self.crop_size)))
                else:
                    bbox, area = boxes[li]
                    z1, z2 = bbox[0], bbox[3]
                    y1, y2 = max(bbox[1] - 16, 0), min(bbox[4] + 16, 256)
                    x1, x2 = max(bbox[2] - 16, 0), min(bbox[5] + 16, 256)
                    images = image_cube[z1:z2, y1:y2, x1:x2].copy()
                    masks = mask_cube[z1:z2, y1:y2, x1:x2].copy()
                    subsample_num = int((z2-z1)/slice_size) if (z2-z1)%slice_size==0 else int((z2-z1)/slice_size)+1

                    subsample_id=random.randint(0, subsample_num-1)
                    images_sub = images[subsample_id::subsample_num]
                    masks_sub = masks[subsample_id::subsample_num]
                    images_sub = self.process_images(images_sub,masks_sub,image_mean,image_std,slice_size)
                    all_images.append(images_sub)
            labels = {
                    #'bowel_healthy':row['bowel_healthy'],
                    #'bowel_injury':row['bowel_injury'],
                    #'extravasation_healthy':row['extravasation_healthy'],
                    #'extravasation_injury':row['extravasation_injury'],
                    #'kidney_healthy':row['kidney_healthy']
                    'kidney_low':row['kidney_low'],
                    'kidney_high':row['kidney_high'],
                    #'liver_healthy':row['liver_healthy']
                    'liver_low':row['liver_low'],
                    'liver_high':row['liver_high'],
                    #'spleen_healthy':row['spleen_healthy']
                    'spleen_low':row['spleen_low'],
                    'spleen_high':row['spleen_high'],
                    #'any_injury':row['any_injury']
            }

            labels = np.array(list(labels.values())).astype(np.float32)

            sample = {}
            sample['image'] = torch.from_numpy(np.array(all_images)).float()
            sample['label'] = torch.from_numpy(labels).float()
            sample['cube_id'] = cube_id

        return sample

    @property
    def is_train(self):
        return self.mode == "train"

    @property
    def parts_map(self):
        parts_map = {
            1:['liver_low','liver_high'],
            2:['spleen_low','spleen_high'],
            3:['kidney_low','kidney_high'],
            #4:['bowel_injury'],
        }
        return parts_map

    @property
    def parts_map_health(self):
        parts_map_health = {
              1:'liver_healthy',
              2:'spleen_healthy',
              3:'kidney_healthy',
              #4:'bowel_healthy',
          }
        return parts_map_health

    def __len__(self):
        return len(self.df)

    def get_weights(self):
        overall = self.df['kidney_liver_spleen_healthy_num'].values
        weights = np.zeros((len(overall),))
        weights[overall == 0] = (len(overall) / len(overall[overall==0]))
        weights[overall == 1] = (len(overall) / len(overall[overall==1]))
        weights[overall == 2] = (len(overall) / len(overall[overall==2]))
        weights[overall == 3] = (len(overall) / len(overall[overall==3]))
        return weights


class ClassifierDatasetCropsFullRes_triple_organ(Dataset):
    def __init__(
            self,
            mode: str,
            dataset_dir: str,
            fold: int,
            transforms: A.Compose,
            slice_size: int,
            crop_size: int,
            multiplier: int = 1,
            folds_csv="folds.csv",
            metadata="meta.json",
    ):
        df = pd.read_csv(os.path.join(folds_csv))
        if mode == "train":
            self.df = df[df['split'] != fold]
        else:
            self.df = df[(df['split'] == fold)]
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.slice_size = slice_size
        self.crop_size = crop_size

        if self.mode == "train":
            self.df = pd.concat([self.df] * multiplier)
        if self.is_train:
            assert len(self.df[self.df['split'] == fold]) == 0
        else:
            assert len(self.df[self.df['split'] != fold]) == 0
        with open(metadata, "r") as f:
            self.metadata = json.load(f)
        self.cache = {}

    def __getitem__(self, i):
        return self.getitem(i)

    def find_or_box(self,image_cube,boxes):

        z10,y10,x10,z20,y20,x20 = [image_cube.shape[0],image_cube.shape[1],image_cube.shape[2],0,0,0]
        liver_found = 1 in boxes.keys()
        spleen_found = 2 in boxes.keys()
        kidney_found = 3 in boxes.keys()
        found = liver_found | spleen_found | kidney_found
        for li in [1,2,3]:
            if li in boxes.keys():
              (z1,y1,x1,z2,y2,x2),area = boxes[li]
              if z1<z10: z10=z1
              if y1<y10: y10=y1
              if x1<x10: x10=x1
              if z2>z20: z20=z2
              if y2>y20: y20=y2
              if x2>x20: x20=x2
        
        bbox = [z10,y10,x10,z20,y20,x20] if found else [0,0,0,image_cube.shape[0],image_cube.shape[1],image_cube.shape[2]]
        return bbox, liver_found, spleen_found, kidney_found, found

    def process_images(self,images,masks,image_mean,image_std,slice_size):
        replay = None
        image_crops = []
        mask_crops = []                
        for i in range(images.shape[0]):
            image = images[i]
            mask = masks[i]
            h, w, = mask.shape
            if replay is None:
                sample = self.transforms(image=image, mask=mask)
                replay = sample["replay"]
            else:
                sample = ReplayCompose.replay(replay, image=image, mask=mask)
            image_ = sample["image"]
            image_crops.append(image_)
            mask_crops.append(sample["mask"])
        images = np.array(image_crops).astype(np.float32)
        masks = np.array(mask_crops).astype(np.float32)


        images = np.expand_dims(images, -1)
        masks = np.expand_dims(masks, -1)

        images = (images - image_mean) / image_std


        images = np.concatenate([images, images, masks], axis=-1)

        h=images.shape[0]
        if h != slice_size:
            tmp = np.zeros((slice_size, *images.shape[1:]))
            tmp[:h] = images
            images = tmp

        images = np.moveaxis(images, -1, 0)
        return images

    def getitem(self, i):
        row = self.df.iloc[i]
        series = row.series.split('|')
        cube_id = random.choice(series)

        mask_cube = nib.load(os.path.join(self.dataset_dir, f"seg", f"{cube_id}.nii.gz")).get_fdata()
        mask_cube = mask_cube.transpose((2,1,0))
        mask_cube[mask_cube==4] = 3
        mask_cube[mask_cube==5] = 0
        image_cube = nib.load(os.path.join(self.dataset_dir, f"cube", f"{cube_id}.nii.gz")).get_fdata()
        image_cube = image_cube.transpose((2,1,0))


        meta = self.metadata[cube_id]
        image_mean = meta["image_mean"]
        image_std = meta["image_std"]
        sums = meta["sums"]
        boxes = meta["boxes"]
        boxes = {int(k):v for k, v in boxes.items()}
        slice_size = self.slice_size

        bbox, liver_found, spleen_found, kidney_found, found = self.find_or_box(image_cube,boxes)
        labels = {
                #'bowel_healthy':row['bowel_healthy'],
                #'bowel_injury':row['bowel_injury'],
                #'extravasation_healthy':row['extravasation_healthy'],
                #'extravasation_injury':row['extravasation_injury'],
                #'kidney_healthy':row['kidney_healthy'] if kidney_found else 1,
                'kidney_low':row['kidney_low'] if kidney_found else 0,
                'kidney_high':row['kidney_high']  if kidney_found else 0,
                #'liver_healthy':row['liver_healthy'] if liver_found else 1,
                'liver_low':row['liver_low'] if liver_found else 0,
                'liver_high':row['liver_high']  if liver_found else 0,
                #'spleen_healthy':row['spleen_healthy'] if spleen_found else 1,
                'spleen_low':row['spleen_low'] if spleen_found else 0,
                'spleen_high':row['spleen_high']  if spleen_found else 0,
                #'any_injury':row['any_injury']
        }
        labels = np.array(list(labels.values())).astype(np.float32)


#        if self.is_train:
#            z1, z2 = bbox[0], bbox[3]
#            if mask_cube.shape[0] > slice_size * 2 and random.random() < 0.0:
#                #downsample
#                image_cube = image_cube[::2]
#                mask_cube = mask_cube[::2]
#                z1 = z1//2
#                z2 = z2//2
#        else:
#            pass

        z1, z2 = bbox[0], bbox[3]
        y1, y2 = max(bbox[1] - 16, 0), min(bbox[4] + 16, 256)
        x1, x2 = max(bbox[2] - 16, 0), min(bbox[5] + 16, 256)
        images = image_cube[z1:z2, y1:y2, x1:x2].copy()
        masks = mask_cube[z1:z2, y1:y2, x1:x2].copy()

        subsample_num = int((z2-z1)/slice_size) if (z2-z1)%slice_size==0 else int((z2-z1)/slice_size)+1
        if self.is_train:
            subsample_id=random.randint(0, subsample_num-1)
            images = images[subsample_id::subsample_num]
            masks = masks[subsample_id::subsample_num]

            images = self.process_images(images,masks,image_mean,image_std,slice_size)
      
            sample = {}

            sample['image'] = torch.from_numpy(images).float()
            sample['label'] = torch.from_numpy(labels).float()
            sample['cube_id'] = cube_id
        else:
            all_images = []
            for subsample_id in range(subsample_num):
                images_sub = images[subsample_id::subsample_num]
                masks_sub = masks[subsample_id::subsample_num]
                images_sub = self.process_images(images_sub,masks_sub,image_mean,image_std,slice_size)
                all_images.append(images_sub)
            sample = {}
            sample['image'] = torch.from_numpy(np.array(all_images)).float()
            sample['label'] = torch.from_numpy(labels).float()
            sample['cube_id'] = cube_id

        return sample

    @property
    def is_train(self):
        return self.mode == "train"

    @property
    def parts_map(self):
        parts_map = {
            1:['liver_low','liver_high'],
            2:['spleen_low','spleen_high'],
            3:['kidney_low','kidney_high'],
            #4:['bowel_injury'],
        }
        return parts_map

    @property
    def parts_map_health(self):
        parts_map_health = {
              1:'liver_healthy',
              2:'spleen_healthy',
              3:'kidney_healthy',
              #4:'bowel_healthy',
          }
        return parts_map_health

    def __len__(self):
        return len(self.df)

    def get_weights(self):
        overall = self.df['kidney_liver_spleen_healthy_num'].values
        weights = np.zeros((len(overall),))
        weights[overall == 0] = (len(overall) / len(overall[overall==0]))
        weights[overall == 1] = (len(overall) / len(overall[overall==1]))
        weights[overall == 2] = (len(overall) / len(overall[overall==2]))
        weights[overall == 3] = (len(overall) / len(overall[overall==3]))
        return weights