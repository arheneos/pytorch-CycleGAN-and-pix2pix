import copy
from PIL import Image
from data.base_dataset import get_transform
from models import networks
import torch
import glob
import cv2
import numpy as np
import torchvision.transforms as transforms

netG_B = networks.define_G(1, 1, 128, 'resnet_9blocks')

state_dict = torch.load('/home/psdl/Workspace/pytorch-CycleGAN-and-pix2pix/checkpoints/maps_cyclegan/latest_net_G_A.pth', map_location='cuda:1')

netG_B.load_state_dict(state_dict, strict=True)
netG_B.eval()
ret = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor()])
i = 0
for single in glob.glob('train/*.bin'):
    i += 1
    with open(single, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    data = np.reshape(data[:120 * 120], (120, 120)).copy()
    data = data - np.median(data)
    data = data[32:96, 32:96]
    A_img = Image.fromarray(data)
    rs = ret(A_img)
    rs = rs[None, :, :, :]
    rs = rs.repeat(4, 1, 1, 1)
    print(rs.shape)
    orig = copy.deepcopy(data)
    res: torch.types.Tensor = netG_B(rs).cpu()
    image_numpy = res[0].data.cpu().float().numpy()  # convert it into a numpy array
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    img = image_numpy.astype(np.float32)
    img_gray = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_magma = cv2.applyColorMap(img_gray, cv2.COLORMAP_MAGMA)
    orig = cv2.normalize(orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    orig = cv2.applyColorMap(orig, cv2.COLORMAP_MAGMA)
    cv2.imwrite(f'imgs/img{i}.png', img_magma)
    print(res)
