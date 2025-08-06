import os
import shutil

imgSrcPrefix = "./data/kitti/image/"
depthSrcPrefix = "./data/kitti/groundtruth_depth/"
dstPath = "C:\\Users\\PC\\Desktop\\Temporary\\projects\\custom_depthAnythingV2\\data\\kitti_temporal/"
varietyDstPrefix = "C:\\Users\\PC\\Desktop\\Temporary\\projects\\custom_depthAnythingV2\\data\\kitti_variety\\"

filenames = os.listdir(imgSrcPrefix)
sorted_filenames = sorted(filenames)
numFiles = len(sorted_filenames)

for i in range(numFiles):

    imgName = sorted_filenames[i]
    dpName = imgName.replace("sync_image_", "sync_groundtruth_depth_", 1)

    print(f"processing file {i} -->", imgName)

    rgbSrc = imgSrcPrefix + imgName
    rgbDst = dstPath + f"{i:05d}_colors.png"

    depthSrc = depthSrcPrefix + dpName
    depthDst = dstPath + f"{i:05d}_depth.png"

    if not os.path.exists(rgbSrc):
        raise FileNotFoundError(f"RGB source path not found: {rgbSrc}")
    if not os.path.exists(depthSrc): 
        raise FileNotFoundError(f"Depth source path not found: {depthSrc}")
    if not os.path.exists(dstPath):
        os.makedirs(dstPath)

    shutil.copy(rgbSrc, rgbDst)
    shutil.copy(depthSrc, depthDst)

    if i % 20 == 0:
        var_i = i // 20
        if not os.path.exists(varietyDstPrefix):
            os.makedirs(varietyDstPrefix)

        rgbVarDst   = varietyDstPrefix + f"{var_i:05d}_colors.png"
        depthVarDst = varietyDstPrefix + f"{var_i:05d}_depth.png"
        shutil.copy(rgbSrc,   rgbVarDst)
        shutil.copy(depthSrc, depthVarDst)

    



