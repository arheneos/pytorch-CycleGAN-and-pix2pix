import cv2
import numpy as np


with open('train/5536BD74-FB63-4A90-9EE4-39CBB9183A48.bin', 'rb') as f:
    d = np.frombuffer(f.read(), dtype=np.float32)

d = np.reshape(d[:120 * 120], (120, 120)).copy()
d = d - np.mean(d)
print(np.std(d))

d = np.load('real/5160fdee-5499-4132-9327-ec476ac9365e.npy')
d = d - np.mean(d)
d = -d
d /= 7
print(np.std(d))
v = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
v = cv2.applyColorMap(v, cv2.COLORMAP_MAGMA)
cv2.imshow('frame', v)
cv2.waitKey(0)
print(d)