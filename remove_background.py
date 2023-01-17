from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn import cluster
from bisect import bisect
from skimage import filters
from findpeaks import findpeaks
from scipy import signal
import cv2
imfile = "demo_output/scene2.png"
dpfile = "demo_output/scene2.npy"
img = np.array(Image.open(imfile)).astype(np.uint8)
# depth = np.array(Image.open(dpfile))
h = img.shape[0]
w = img.shape[1]

# coord_grid = np.stack(np.meshgrid(np.linspace(-1, 1,img.shape[0]), np.linspace(-1, 1,img.shape[1]), indexing='ij'), axis=-1)

# depth = np.zeros([img.shape[0], img.shape[1]])
depth = np.load(dpfile)
depth = depth[0:h,0:w]
depth = depth - np.max(depth)


depth_bilateral = cv2.bilateralFilter(depth, 15, 75, 75)


# calculate histogram
hist = np.array(np.histogram(depth.flatten(), bins=100))
hist[1] = hist[1][:100]

plt.figure()
plt.imshow(depth)
plt.show()
plt.imshow(depth_bilateral)
plt.show()

thresholds = 5, 10, 20, 30
# calculate median pixel and remove it from the photo
fig, ax = plt.subplots(1,len(thresholds))
for it, threshold in enumerate(thresholds):
    median_clean = np.median(depth,axis=1)
    median_matrix =  np.array(np.tile(median_clean,(w,1)).T)
    filter_map = np.abs(depth-median_matrix)>=threshold
    depth_clean = depth
    depth_clean[np.invert(filter_map)] = 0
    ax[it].imshow(depth_clean, cmap='gray')
    ax[it].set_title("Threshold:" + str(threshold))
plt.show()


# plot histogram
plt.figure()
plt.hist(depth.flatten(), bins = 100)
plt.show()

#find peaks in histogram

# fp = findpeaks('peak', lookahead=3, interpolate=5)
# peaks = fp.fit(hist[0])
# # fp.plot()
# peaks, _ = signal.find_peaks(hist[0],distance=10,wlen=5)
# num_layers = len(peaks)

# k_means = cluster.KMeans(num_layers,init=peaks.reshape(-1, 1))

# k_means.fit(hist[1].reshape(-1, 1), hist[0].reshape(-1, 1))

# cluster_id = k_means.labels_
# plt.figure()

# boundaries = []
# for ii in np.unique(cluster_id):
#     subset = hist[1][cluster_id==ii]
#     boundaries += [np.max(subset)]
# boundaries= boundaries[:-1]
# # show histogram# show histogram
# #     plt.hist(subset, bins=20, alpha=0.5, label=f"Cluster {ii}")

# # plt.show()


# # deriving layers
# layers = np.zeros([num_layers, img.shape[0], img.shape[1], img.shape[2]]).astype(np.uint8)
# for i in range(depth.shape[0]):
#     for j in range(depth.shape[1]):
#         index = bisect(boundaries, depth[i, j])
#         layers[index, i, j, :] = img[i,j]

# for i, layer in enumerate(layers):
#     plt.imsave("parallax/layer" + str(i) + ".png", layer)

# d1 = 30
# depth = d1 + depth

# for i in range(1,10):
#     # calculate new depth map
#     depth_new = depth-i
#     #calculate new coordinates
#     coords_new = np.zeros(coord_grid.shape) 
#     coords_new[:,:,0]= np.multiply(coord_grid[:,:,0],depth/depth_new)*h/2+h/2
#     coords_new[:,:,1]= np.multiply(coord_grid[:,:,1],depth/depth_new)*w/2+w/2
#     coords_new = coords_new.astype(np.float32)
#     img_new = Image.new('RGB', (w, h)) # np.zeros(img.shape).astype(np.uint8)
#     # img_new = ImageOps.invert(img_new)
#     #generate new image
#     for j in range(len(layers)-1, -1,-1):
#         depth_coef = depth/depth_new
#         mean_depth = np.mean(depth_coef[((layers[j]!=0).sum(axis=-1)/3).astype(np.int)])
#         print("mean depth: " + str(mean_depth))
#         print("min depth: " + str(np.min(depth)))
#         print("max depth: " + str(np.max(depth)))
#         depth_coef[:,:] = mean_depth
#         # depth_coef[((layers[j]==0).sum(axis=-1)/3).astype(np.int)] = mean_depth
#         # depth_coef = filters.gaussian(depth_coef, sigma=(5,5))
#         coords_new[:,:,0]= np.multiply(coord_grid[:,:,0], depth_coef)*h/2+h/2
#         coords_new[:,:,1]= np.multiply(coord_grid[:,:,1], depth_coef)*w/2+w/2
#         coords_new = coords_new.astype(np.float32)

#         img_temp = cv.remap(layers[j], coords_new[:,:,1], coords_new[:,:,0], cv.INTER_LINEAR)
#         mask_temp = ((img_temp!=0).sum(axis=-1)/3*255).astype(np.uint8)
        
#         # calculate new coordinates
        

#         img_temp = Image.fromarray(img_temp)
#         mask_temp = Image.fromarray(mask_temp)

#         # img_new[(img_temp!=[0,0,0]).sum(axis=-1)] == img_temp[(img_temp!=[0,0,0]).sum(axis=-1)]

#         img_new.paste(img_temp, (0,0), mask=mask_temp)
#         if j == 0:
#             img_temp.save("parallax/test_"+str(i)+str(j)+".png")
#             plt.imsave("parallax/depth_" +str(i)+str(j)+".png", depth)
#         #     # plt.imsave(, img_temp)


#     # save enw img
#     img_new.save("parallax/res_"+str(i)+".png")
#     # plt.imsave("parallax/res_"+str(i)+".png", img_new)



# print(depth)