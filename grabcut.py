from PIL import Image, ImageOps, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn import cluster
from bisect import bisect
from skimage import filters
from findpeaks import findpeaks
from scipy import signal

def find_rectangle(mask):
    x0, x1, y0, y1 = -1, -1, -1, -1
    for it, m in enumerate(mask):
        if (m>0).any():
            if y0 == -1:
                y0 = it
            y1 = it
    
    for it, m in enumerate(mask.T):
        if (m>0).any():
            if x0 == -1:
                x0 = it
            x1 = it
    return x0-10, y0-10, x1-x0+20, y1-y0+20

imfile = "records/scene7/2.png"
dpfile = "demo_output/scene7.npy"
maskfile = "records/scene7/mask.png"
img = np.array(Image.open(imfile)).astype(np.uint8)
mask = np.array(Image.open(maskfile).convert('L')).astype(np.uint8)
# depth = np.array(Image.open(dpfile))
h = img.shape[0]
w = img.shape[1]

coord_grid = np.stack(np.meshgrid(np.linspace(-1, 1,img.shape[0]), np.linspace(-1, 1,img.shape[1]), indexing='ij'), axis=-1)

# depth = np.zeros([img.shape[0], img.shape[1]])
depth = np.load(dpfile)
depth = depth[0:h,0:w]
depth = depth - np.max(depth)

mask = mask.flatten()
mask[mask<150] = 0
mask[mask>=150] = 1

mask = mask.reshape(h,w)

rect = find_rectangle(mask)

# img1 = ImageDraw.Draw((Image.open(maskfile)))
# img1.rectangle(rect, fill ="#800080", outline ="green")
# img1 = np.array(img1)
print(rect)

bgModel = np.zeros((1,65), np.float64)
fgModel = np.zeros((1,65), np.float64)
cv.grabCut(img, None, rect,  bgModel,
	fgModel, iterCount=5, mode=cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

img_new = img*mask2[:,:,np.newaxis]

fig, ax = plt.subplots(1,2)
ax[0] = plt.imshow(mask2)
ax[1] = plt.imshow(img_new)
plt.show()

mask_int = mask*255



# deriving layers
layers = np.zeros([num_layers, img.shape[0], img.shape[1], img.shape[2]]).astype(np.uint8)
for i in range(depth.shape[0]):
    for j in range(depth.shape[1]):
        index = bisect(boundaries, depth[i, j])
        layers[index, i, j, :] = img[i,j]

for i, layer in enumerate(layers):
    plt.imsave("parallax/layer" + str(i) + ".png", layer)

# d1 = 30
# depth = d1 + depth

for i in range(1,10):
    # calculate new depth map
    depth_new = depth-i
    #calculate new coordinates
    coords_new = np.zeros(coord_grid.shape) 
    coords_new[:,:,0]= np.multiply(coord_grid[:,:,0],depth/depth_new)*h/2+h/2
    coords_new[:,:,1]= np.multiply(coord_grid[:,:,1],depth/depth_new)*w/2+w/2
    coords_new = coords_new.astype(np.float32)
    img_new = Image.new('RGB', (w, h)) # np.zeros(img.shape).astype(np.uint8)
    # img_new = ImageOps.invert(img_new)
    #generate new image
    for j in range(len(layers)-1, -1,-1):
        depth_coef = depth/depth_new
        mean_depth = np.mean(depth_coef[((layers[j]!=0).sum(axis=-1)/3).astype(np.int)])
        print("mean depth: " + str(mean_depth))
        print("min depth: " + str(np.min(depth)))
        print("max depth: " + str(np.max(depth)))
        depth_coef[:,:] = mean_depth
        # depth_coef[((layers[j]==0).sum(axis=-1)/3).astype(np.int)] = mean_depth
        # depth_coef = filters.gaussian(depth_coef, sigma=(5,5))
        coords_new[:,:,0]= np.multiply(coord_grid[:,:,0], depth_coef)*h/2+h/2
        coords_new[:,:,1]= np.multiply(coord_grid[:,:,1], depth_coef)*w/2+w/2
        coords_new = coords_new.astype(np.float32)

        img_temp = cv.remap(layers[j], coords_new[:,:,1], coords_new[:,:,0], cv.INTER_LINEAR)
        mask_temp = ((img_temp!=0).sum(axis=-1)/3*255).astype(np.uint8)
        
        # calculate new coordinates
        

        img_temp = Image.fromarray(img_temp)
        mask_temp = Image.fromarray(mask_temp)

        # img_new[(img_temp!=[0,0,0]).sum(axis=-1)] == img_temp[(img_temp!=[0,0,0]).sum(axis=-1)]

        img_new.paste(img_temp, (0,0), mask=mask_temp)
        if j == 0:
            img_temp.save("parallax/test_"+str(i)+str(j)+".png")
            plt.imsave("parallax/depth_" +str(i)+str(j)+".png", depth)
        #     # plt.imsave(, img_temp)


    # save enw img
    img_new.save("parallax/res_"+str(i)+".png")
    # plt.imsave("parallax/res_"+str(i)+".png", img_new)



# print(depth)