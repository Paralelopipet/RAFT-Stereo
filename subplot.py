import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots(1,7)
for i in range(7):
    img = np.array(cv2.imread("parallax/layer" + str(i) + ".png"))
    ax[i].imshow(img)
    ax[i].axis('off')
plt.show()
input()

