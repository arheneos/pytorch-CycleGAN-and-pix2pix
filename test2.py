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


def normalize_real_range(data, R=1.0):
    # 실수 범위 전체에서 통계량 추출
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)  # 혹은 np.median(data)

    # 분모가 0이 되는 것을 방지 (모든 값이 같을 경우)
    if max_val == min_val:
        return np.zeros_like(data), mean_val, min_val, max_val

    z_norm = np.zeros_like(data, dtype=float)

    # [구간 1] Min ~ Mean -> [-R, 0]
    lower_mask = (data <= mean_val)
    denom_lower = mean_val - min_val
    if denom_lower > 0:
        z_norm[lower_mask] = R * (data[lower_mask] - mean_val) / denom_lower
    else:
        z_norm[lower_mask] = 0.0

    # [구간 2] Mean ~ Max -> [0, R]
    upper_mask = (data > mean_val)
    denom_upper = max_val - mean_val
    if denom_upper > 0:
        z_norm[upper_mask] = R * (data[upper_mask] - mean_val) / denom_upper
    else:
        z_norm[upper_mask] = 0.0

    return z_norm, mean_val, min_val, max_val


def restore_real_range(z_norm, mean_val, min_val, max_val, R=1.0):
    recovered = np.zeros_like(z_norm, dtype=float)

    # [복원 1] -R ~ 0 -> Min ~ Mean
    lower_mask = (z_norm <= 0)
    recovered[lower_mask] = (z_norm[lower_mask] * (mean_val - min_val) / R) + mean_val

    # [복원 2] 0 ~ R -> Mean ~ Max
    upper_mask = (z_norm > 0)
    recovered[upper_mask] = (z_norm[upper_mask] * (max_val - mean_val) / R) + mean_val

    return recovered


state_dict = torch.load('latest_net_G_A.pth', map_location='cpu')

netG_B.load_state_dict(state_dict, strict=True)
netG_B.eval()
"""
dummy_input = torch.randn(1, 1, 64, 64)

torch.onnx.export(
    netG_B,
    dummy_input,
    "G_A.onnx",
    input_names=["img"],
    output_names=["output"],
    opset_version=20,
    do_constant_folding=True,
    dynamic_axes={
        "img": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 1: "height", 2: "width"}
    },
)
"""
ret = transforms.Compose([transforms.ToTensor()])
i = 0
for single in glob.glob('train/*.bin'):
    i += 1
    with open(single, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    od = np.reshape(data[:120 * 120], (120, 120)).copy()
    protein = np.reshape(data[120 * 120:], (120 * 4, 120 * 4)).copy()
    print(protein)
    zn, mu, mv, v = normalize_real_range(protein)
    print(np.min(zn), np.max(zn))
    recons = restore_real_range(zn, mu, mv, v)

    data = od - np.mean(od)
    orig = copy.deepcopy(data)
    data = np.asarray([data])
    std = data.std()
    dz = data / (std * 10)
    dz = np.clip(dz, -1, 1)
    rs = torch.from_numpy(dz)
    rs = rs[None, :, :, :]
    print(rs.max())
    print(rs.shape)
    res: torch.types.Tensor = netG_B(rs).cpu()
    image_numpy = res[0].data.cpu().float().numpy()  # convert it into a numpy array
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    img = image_numpy.astype(np.float32)
    img_gray = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_magma = cv2.applyColorMap(img_gray, cv2.COLORMAP_MAGMA)
    maxv = np.where(orig > orig.std() * 10, 1.0, 0.0)
    orig = cv2.normalize(orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    orig = cv2.applyColorMap(orig, cv2.COLORMAP_MAGMA)
    cv2.imshow('orig', orig)
    cv2.imshow('maxv', maxv)
    cv2.imshow('frame', img_magma)
    cv2.waitKey(0)
    print(res)
