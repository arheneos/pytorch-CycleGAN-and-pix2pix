import glob
import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np


def normalize_min_max(data, R=1.0):
    """
    데이터를 [-R, R] 범위로 Min-Max 정규화 (기본 R=1.0)
    """
    min_val = np.min(data)
    max_val = np.max(data)

    # 분모가 0이 되는 경우 방지 (모든 값이 동일할 때)
    if max_val == min_val:
        return np.zeros_like(data), min_val, max_val

    # [수식] x_norm = 2 * (x - min) / (max - min) - 1
    # 여기에 R을 곱하여 [-R, R] 범위를 조정합니다.
    data_norm = 2 * (data - min_val) / (max_val - min_val) - 1
    data_norm = data_norm * R

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

        self.A_paths = sorted(glob.glob('/home/psdl/Workspace/SUNDAE_GAN/train/*.bin'))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(glob.glob('/home/psdl/Workspace/SUNDAE_GAN/Real/*.npy'))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == "BtoA"
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1), convert=False)
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1), convert=False)

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
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        with open(A_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.float32)

        data = np.reshape(data[:120 * 120], (120, 120)).copy()
        data = normalize_min_max(data)
        A_img = Image.fromarray(data)
        b = -np.load(B_path)
        if b.shape[0] < 64 and b.shape[1] < 64:
            B_path = self.B_paths[index_B + 1]
            b = -np.load(B_path)
        b = normalize_min_max(b)
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
