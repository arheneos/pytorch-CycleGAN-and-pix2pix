import glob
import os

import cv2

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import tqdm
import h5py
import torchvision.transforms as T


def correct_plane(image):
    """
    Correct the image by subtracting a fitted 2D-plane on the data

    Parameters
    ----------
    inline : bool
        If True the data of the current image will be updated otherwise a new image is created
    mask : None or 2D numpy array
        If not None define on which pixels the data should be taken.
    """
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    X0, Y0 = np.meshgrid(x, y)
    Z0 = image
    X = X0
    Y = Y0
    Z = Z0
    A = np.column_stack((np.ones(Z.ravel().size), X.ravel(), Y.ravel()))
    c, resid, rank, sigma = np.linalg.lstsq(A, Z.ravel(), rcond=-1)
    image -= c[0] * np.ones(image.shape) + c[1] * X0 + c[2] * Y0
    return image - np.min(image)


def normalize_min_max(data, R=1.0):
    """
    데이터를 [-R, R] 범위로 Min-Max 정규화 (기본 R=1.0)
    """
    min_val = np.min(data)
    max_val = np.max(data)

    # 분모가 0이 되는 경우 방지 (모든 값이 동일할 때)
    if max_val == min_val:
        return np.zeros_like(data)

    data_norm = 2 * (data - min_val) / (max_val - min_val) - 1
    data_norm = data_norm * R
    data_norm = np.clip(data_norm, -1, 1)
    return data_norm


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  # create a path '/path/to/data/trainB'
        with h5py.File('/home/psdl/Workspace/HS-AFM-UPSCALER/Master_Dataset_v2.h5', 'r') as f:
            self.uids = list(f.keys())
        self.file = h5py.File('/home/psdl/Workspace/HS-AFM-UPSCALER/Master_Dataset_v2.h5', 'r')

        self.A_paths = sorted([])  # load images from '/path/to/data/trainA'
        B_paths = sorted(glob.glob('/home/psdl/Workspace/SUNDAE_GAN/Real/*.npy'))  # load images from '/path/to/data/trainB'

        self.B_paths = []
        for single in tqdm.tqdm(B_paths):
            data = -np.load(single).copy()
            h, w = data.shape
            data = data - np.mean(data)
            if h < 100 or w < 100:
                continue
            if np.isclose(np.std(data), 0):
                continue
            if not np.isfinite(data).all():
                continue
            self.B_paths.append(single)

        self.A_size = len(self.uids)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == "BtoA"
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image

        # B: LR real AFM data at crop_size (e.g. 64x64)
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1), convert=False)

        # A: HR simulation data at 4x crop_size (e.g. 256x256)
        hr_crop = opt.crop_size * 4
        print(hr_crop)
        transform_A_list = [T.RandomCrop(hr_crop)]
        if not opt.no_flip:
            transform_A_list.append(T.RandomHorizontalFlip())
        transform_A_list.append(T.ToTensor())
        self.transform_A = T.Compose(transform_A_list)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.uids[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        uid = self.uids[index % self.A_size]
        group = self.file[uid]
        data = group['mask'][:]
        # data = cv2.resize(data, (120, 120), interpolation=cv2.INTER_LINEAR)
        # data = data - np.mean(data)
        # data = data / (np.std(data) + 1e-8)
        A_img = Image.fromarray(data)
        b = -np.load(B_path)
        b = correct_plane(b)
        # b = b - np.mean(b)
        b = b / (np.std(b) + 1e-8)
        B_img = Image.fromarray(b)
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
