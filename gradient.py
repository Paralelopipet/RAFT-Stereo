import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib as mpl
import numpy as np

folder_path = "gradient_img/"

# mask = np.zeros(shape=(1920,1200), dtype="uint8")

# # Draw a bounding box.
# # Draw a white, filled rectangle on the mask image
# cv2.rectangle(img=mask,
#             pt1=(1010, 703), pt2=(1320, 1200),
#             color=(255, 255, 255),
#             thickness=-1)
left, top, right, bottom = 1010, 703, 1320, 1200

for i in [0,3,4]:

    img = cv2.imread(folder_path + str(i)+'.png')

    kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    edges_x = cv2.filter2D(img,cv2.CV_8U,kernelx)
    edges_y = cv2.filter2D(img,cv2.CV_8U,kernely)

    edges = np.sqrt(edges_x**2 + edges_y**2)
    rect = edges[left:right, top:bottom]
    mean = np.mean(edges)
    std = np.std(edges)
    max = np.max(edges)
    min = np.min(edges)
    mean_ = np.mean(rect)
    std_ = np.std(rect)
    max_ = np.max(rect)
    min_ = np.min(rect)

    print("Stats for image " + str(i))
    print("Mean: ", mean)
    print("Std: ", std)
    print("Max: ", max)
    print("Min: ", min)
    print("Mean_: ", mean_)
    print("Std_: ", std_)
    print("Max_: ", max_)
    print("Min_: ", min_)
    # if i ==0:
    #     max_gt = np.max(edges)
    # edges = (edges*255/max_gt).astype(np.uint8)
    # plt.figure()
    # plt.imshow(edges)
    # plt.show(block=False)
# cv2.imshow('Gradients_Y',edges_y)
input()

