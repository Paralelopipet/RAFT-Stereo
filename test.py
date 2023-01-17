# code
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage import data,morphology
from skimage import segmentation as seg
from skimage.color import rgb2gray, label2rgb
import scipy.ndimage as nd
from scipy.ndimage import sobel
from PIL import Image
plt.rcParams["figure.figsize"] = (12,8)

 
imfile = "records/scene2/2.png"
dpfile = "demo_output/scene2.npy"
img = np.array(Image.open(imfile)).astype(np.uint8)
# depth = np.array(Image.open(dpfile))
h = img.shape[0]
w = img.shape[1]

# coord_grid = np.stack(np.meshgrid(np.linspace(-1, 1,img.shape[0]), np.linspace(-1, 1,img.shape[1]), indexing='ij'), axis=-1)

# depth = np.zeros([img.shape[0], img.shape[1]])
depth = np.load(dpfile)
depth = depth[0:h,0:w]

#grayscale image
grayscale = np.array(Image.open(imfile).convert('L')).astype(np.uint8)

rocket_wh = (depth)
 
# Region Segmentation
# First we print the elevation map
elevation_map = sobel(rocket_wh)
print(np.min(elevation_map), np.max(elevation_map))
elevation_map = elevation_map.flatten()
elevation_map[np.abs(elevation_map)>10] = 255
elevation_map[np.abs(elevation_map)<=10] = 0.0
elevation_map = elevation_map.reshape([h,w])

plt.imshow(elevation_map, cmap='gray')
plt.show()

elevation_gray = canny(grayscale, sigma=2)
plt.imshow(elevation_gray, cmap='gray')
plt.show()


# combine the edges
# Iterate over the certain edges

for k in range(5):
    for i in range(elevation_map.shape[0]):
        for j in range(elevation_map.shape[1]):
            if elevation_map[i,j] != 0:
                # Check the neighborhood of the certain edge pixel
                neighborhood = np.stack([elevation_gray[i-1:i+2, j-1],elevation_gray[i-1:i+2, j],elevation_gray[i-1:i+2, j+1]])
                if np.any(neighborhood):
                    if elevation_map[i,j]==0:
                        elevation_map[i,j] = 1.0
                        changes = True
# after change
plt.imshow(elevation_map, cmap='gray')
plt.show()

# apply edge segmentation
# plot canny edge detection
edges = canny(elevation_map, sigma=0.1)
plt.imshow(edges, interpolation='gaussian', cmap='gray')
plt.title('Canny detector')
plt.show()
 
# fill regions to perform edge segmentation
fill_im = nd.binary_fill_holes(elevation_map)
plt.imshow(fill_im)
plt.title('Region Filling')
plt.show()


# Since, the contrast difference is not much. Anyways we will perform it
markers = np.zeros_like(rocket_wh)
markers[rocket_wh < 0.1171875] = 1 # 30/255
markers[rocket_wh > 0.5859375] = 2 # 150/255
 
plt.imshow(markers)
plt.title('markers')
plt.show()
# Perform watershed region segmentation
segmentation = seg.watershed(elevation_map, markers)
 
plt.imshow(segmentation)
plt.title('Watershed segmentation')

# plot overlays and contour
segmentation = nd.binary_fill_holes(segmentation - 1)
label_rock, _ = nd.label(segmentation)
# overlay image with different labels
image_label_overlay = label2rgb(label_rock, image=rocket_wh)
 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 16), sharey=True)
ax1.imshow(rocket_wh)
ax1.contour(segmentation, [0.8], linewidths=1.8, colors='w')
ax2.imshow(image_label_overlay)
plt.show()
 
# fig.subplots_adjust(**margins)