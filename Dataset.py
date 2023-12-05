
import os
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import train_transforms, get_boxes_from_mask, init_point_sampling
import json
import random


class TestingDataset(Dataset):
    
    def __init__(self, data_path, image_size=256, mode='test', requires_name=True, point_num=1, return_ori_mask=True, prompt_path=None):
        """
        Initializes a TestingDataset object.
        Args:
            data_path (str): The path to the data.
            image_size (int, optional): The size of the image. Defaults to 256.
            mode (str, optional): The mode of the dataset. Defaults to 'test'.
            requires_name (bool, optional): Indicates whether the dataset requires image names. Defaults to True.
            point_num (int, optional): The number of points to retrieve. Defaults to 1.
            return_ori_mask (bool, optional): Indicates whether to return the original mask. Defaults to True.
            prompt_path (str, optional): The path to the prompt file. Defaults to None.
        """
        self.image_size = image_size
        self.return_ori_mask = return_ori_mask
        self.prompt_path = prompt_path
        self.prompt_list = {} if prompt_path is None else json.load(open(prompt_path, "r"))
        self.requires_name = requires_name
        self.point_num = point_num

        json_file = open(os.path.join(data_path, f'label2image_{mode}.json'), "r")
        dataset = json.load(json_file)
    
        self.image_paths = list(dataset.values())
        self.label_paths = list(dataset.keys())
      
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
    
    def __getitem__(self, index):
        """
        Retrieves and preprocesses an item from the dataset.
        Args:
            index (int): The index of the item to retrieve.
        Returns:
            dict: A dictionary containing the preprocessed image and associated information.
        """
        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        mask_path = self.label_paths[index]
        ori_np_mask = cv2.imread(mask_path, 0)
        
        if ori_np_mask.max() == 255:
            ori_np_mask = ori_np_mask / 255

        assert np.array_equal(ori_np_mask, ori_np_mask.astype(bool)), f"Mask should only contain binary values 0 and 1. {self.label_paths[index]}"

        h, w = ori_np_mask.shape
        ori_mask = torch.tensor(ori_np_mask).unsqueeze(0)

        transforms = train_transforms(self.image_size, h, w)
        augments = transforms(image=image, mask=ori_np_mask)
        image, mask = augments['image'], augments['mask'].to(torch.int64)

        if self.prompt_path is None:
            boxes = get_boxes_from_mask(mask)
            point_coords, point_labels = init_point_sampling(mask, self.point_num)
        else:
            prompt_key = mask_path.split('/')[-1]
            boxes = torch.as_tensor(self.prompt_list[prompt_key]["boxes"], dtype=torch.float)
            point_coords = torch.as_tensor(self.prompt_list[prompt_key]["point_coords"], dtype=torch.float)
            point_labels = torch.as_tensor(self.prompt_list[prompt_key]["point_labels"], dtype=torch.int)

        image_input["image"] = image
        image_input["label"] = mask.unsqueeze(0)
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["boxes"] = boxes
        image_input["original_size"] = (h, w)
        image_input["label_path"] = '/'.join(mask_path.split('/')[:-1])

        if self.return_ori_mask:
            image_input["ori_label"] = ori_mask
     
        image_name = self.label_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input

    def __len__(self):
        return len(self.label_paths)


class TrainingDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]

        dataset = json.load(open(os.path.join(data_dir, f'image2label_{mode}.json'), "r"))
        self.image_paths = list(dataset.keys())
        self.label_paths = list(dataset.values())
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
    
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        
        mask_path = random.choices(self.label_paths[index], k=self.mask_num)
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)

class EyesDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [20.86, 42.41, 77.64]
        self.pixel_std = [20.03, 36.05, 59.37]
        image_name = os.listdir(os.path.join(data_dir, 'image'))
        self.image_paths = [os.path.join(data_dir, 'image', i) for i in image_name if i.endswith('jpg')]
        self.label_paths = [i.replace('image/','mask/Result_').replace('jpg','png') for i in self.image_paths]
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
    
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_path = []
        for _ in range(self.mask_num):
            mask_path.append(self.label_paths[index])
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            # boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            # boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        # boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        #image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)

    def get_mean_std(self):
        img_channels = 3
        cumulative_mean = np.zeros(img_channels)
        cumulative_std = np.zeros(img_channels)

        for image_path in tqdm(self.image_paths):
            image = cv2.imread(image_path)
            for d in range(3):
                cumulative_mean[d] += image[:, :, d].mean()
                cumulative_std[d] += image[:, :, d].std()
        self.pixel_mean = cumulative_mean / len(self.image_paths)
        self.pixel_std = cumulative_std / len(self.image_paths)
        print(f"mean: {self.pixel_mean}")
        print(f"std: {self.pixel_std}")

class DriveDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [41.42, 69.02, 126.84]
        self.pixel_std = [25.18, 45.48, 84.57]
        image_name = os.listdir(os.path.join(data_dir, mode, 'images'))
        self.image_paths = [os.path.join(data_dir, mode, 'images', i) for i in image_name if i.endswith('tif')]
        self.label_paths = [i.replace('images','1st_manual').replace('training.tif','manual1.gif') for i in self.image_paths]
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
    
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_path = []
        for _ in range(self.mask_num):
            mask_path.append(self.label_paths[index])
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            # boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            # boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        # boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        #image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)

    def get_mean_std(self):
        img_channels = 3
        cumulative_mean = np.zeros(img_channels)
        cumulative_std = np.zeros(img_channels)

        for image_path in tqdm(self.image_paths):
            image = cv2.imread(image_path)
            for d in range(3):
                cumulative_mean[d] += image[:, :, d].mean()
                cumulative_std[d] += image[:, :, d].std()
        self.pixel_mean = cumulative_mean / len(self.image_paths)
        self.pixel_std = cumulative_std / len(self.image_paths)
        print(f"mean: {self.pixel_mean}")
        print(f"std: {self.pixel_std}")


class Chasedb1Dataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [7.11, 41.80, 115.48]
        self.pixel_std = [8.97, 35.76, 86.67]
        image_name = os.listdir(data_dir)
        self.image_paths = [os.path.join(data_dir, i) for i in image_name if i.endswith('jpg')]
        self.label_paths = [i.replace('.jpg','.png').replace('L','L_1stHO').replace('R','R_1stHO') for i in self.image_paths]
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
    
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_path = []
        for _ in range(self.mask_num):
            mask_path.append(self.label_paths[index])
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            # boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            # boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        # boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        #image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)

    def get_mean_std(self):
        img_channels = 3
        cumulative_mean = np.zeros(img_channels)
        cumulative_std = np.zeros(img_channels)

        for image_path in tqdm(self.image_paths):
            image = cv2.imread(image_path)
            for d in range(3):
                cumulative_mean[d] += image[:, :, d].mean()
                cumulative_std[d] += image[:, :, d].std()
        self.pixel_mean = cumulative_mean / len(self.image_paths)
        self.pixel_std = cumulative_std / len(self.image_paths)
        print(f"mean: {self.pixel_mean}")
        print(f"std: {self.pixel_std}")

class FivesDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [16.20, 38.92, 86.11]
        self.pixel_std = [10.52, 26.44, 54.57]
        image_name = os.listdir(os.path.join(data_dir, mode ,'Original'))
        self.image_paths = [os.path.join(data_dir, mode ,'Original',i) for i in image_name if i.endswith('png')]
        self.label_paths = [i.replace('Original','Ground truth') for i in self.image_paths]
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
    
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_path = []
        for _ in range(self.mask_num):
            mask_path.append(self.label_paths[index])
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            # boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            # boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        # boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        #image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)

    def get_mean_std(self):
        img_channels = 3
        cumulative_mean = np.zeros(img_channels)
        cumulative_std = np.zeros(img_channels)

        for image_path in tqdm(self.image_paths):
            image = cv2.imread(image_path)
            for d in range(3):
                cumulative_mean[d] += image[:, :, d].mean()
                cumulative_std[d] += image[:, :, d].std()
        self.pixel_mean = cumulative_mean / len(self.image_paths)
        self.pixel_std = cumulative_std / len(self.image_paths)
        print(f"mean: {self.pixel_mean}")
        print(f"std: {self.pixel_std}")

class StareDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [28.91, 85.12, 150.18]
        self.pixel_std = [17.85, 45.35, 86.13]
        image_name = os.listdir(os.path.join(data_dir, 'image'))
        self.image_paths = [os.path.join(data_dir, 'image', i) for i in image_name if i.endswith('ppm')]
        self.label_paths = [i.replace('image','mask').replace('.ppm','.vk.ppm') for i in self.image_paths]
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
    
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_path = []
        for _ in range(self.mask_num):
            mask_path.append(self.label_paths[index])
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            # boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            # boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        #boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        # image_input["boxes"] = None
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)

    def get_mean_std(self):
        img_channels = 3
        cumulative_mean = np.zeros(img_channels)
        cumulative_std = np.zeros(img_channels)

        for image_path in tqdm(self.image_paths):
            image = cv2.imread(image_path)
            for d in range(3):
                cumulative_mean[d] += image[:, :, d].mean()
                cumulative_std[d] += image[:, :, d].std()
        self.pixel_mean = cumulative_mean / len(self.image_paths)
        self.pixel_std = cumulative_std / len(self.image_paths)
        print(f"mean: {self.pixel_mean}")
        print(f"std: {self.pixel_std}")

def stack_dict_batched(batched_input):
    out_dict = {}
    for k,v in batched_input.items():
        if isinstance(v, list):
            out_dict[k] = v
        else:
            out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict

class FivesPPO(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [16.20, 38.92, 86.11]
        self.pixel_std = [10.52, 26.44, 54.57]
        image_name = os.listdir(os.path.join(data_dir, mode ,'Original'))
        self.image_paths = [os.path.join(data_dir, mode ,'Original',i) for i in image_name if i.endswith('png')]
        self.label_paths = [i.replace('Original','Ground truth') for i in self.image_paths]
    
    def pad_and_crop(self, image, patch_height=256, patch_width=256):

        height, width, _ = image.shape

        # 计算需要填充的像素数量
        pad_height = (patch_height - (height % patch_height))% patch_height
        pad_width = (patch_width - (width % patch_width))% patch_width

        # 进行填充操作
        padded_image = cv2.copyMakeBorder(image, pad_height // 2, pad_height - (pad_height // 2), 
                                        pad_width // 2, pad_width - (pad_width // 2), 
                                        cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if padded_image.shape != (2048,2048,3):
            padded_image = np.reshape(padded_image, (padded_image.shape[0], padded_image.shape[1], 1))
        cropped_images = []
        for i in range(0, height + pad_height, 256):
            for j in range(0, width + pad_width, 256):
                cropped_images.append(padded_image[i:i+256, j:j+256, :])
        return cropped_images

    def __getitem__(self, index):
        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])
        image_list = np.array(self.pad_and_crop(image))
        mask = cv2.imread(self.label_paths[index], 0)
        if mask.max() == 255:
            mask = mask / 255
            mask = mask[:,:,np.newaxis]
        
        mask_list = np.array(self.pad_and_crop(mask))
        image_input["image"] = torch.tensor(image_list, dtype=torch.float32)
        image_input["label"] = torch.tensor(mask_list, dtype=torch.float32)

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    # train_dataset = TrainingDataset("data_demo", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    # print("Dataset:", len(train_dataset))
    # eyes_dataset = EyesDataset("data/eyes_selected", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    # eyes_dataset = FivesDataset("data/FIVES", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    # eyes_dataset = DriveDataset("data/drive", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    # eyes_dataset = Chasedb1Dataset("data/chasedb1_patch", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    # eyes_dataset = StareDataset("data/stare", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    # train_loader = DataLoader(eyes_dataset, batch_size = 64, shuffle=True, num_workers=0)
    eyes_dataset = FivesPPO("data/FIVES", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    print(eyes_dataset[0])
    # for batch, batched_input in enumerate(train_loader):
    #     print(batched_input)
    # train_batch_sampler = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=4)
    # for i, batched_image in enumerate(tqdm(train_batch_sampler)):
    #     batched_image = stack_dict_batched(batched_image)
    #     print(batched_image["image"].shape, batched_image["label"].shape)

