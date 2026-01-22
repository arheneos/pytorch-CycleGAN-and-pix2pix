import cv2
import numpy as np

with open('train/5536BD74-FB63-4A90-9EE4-39CBB9183A48.bin', 'rb') as f:
    d = np.frombuffer(f.read(), dtype=np.float32)

d = np.reshape(d[:120 * 120], (120, 120)).copy()
d = d - np.mean(d)
print(d)
v = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
v = cv2.applyColorMap(v, cv2.COLORMAP_MAGMA)
cv2.imshow('frame', v)
cv2.waitKey(0)
print(d)
