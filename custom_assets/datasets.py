import os
import cv2
import numpy as np
from custom_assets.utils import *
from enum import Enum

class Dataset(Enum):
    IPHONE = 1
    NYU2 = 2
    KITTI = 3

#==============================================================================================
#==============================================================================================

class DataManager:
    def __init__(self, dtset, inputPath):

        self.dtset = dtset
        self.inputPath = inputPath
        self.numFiles = len(os.listdir(inputPath)) // 2

    #=======================================================================

    def getNumFiles(self):
        return self.numFiles
    
    #=======================================================================

    def getIntrinsics(self):
        
        if self.dtset == Dataset.IPHONE:
            fx = fy = 1521
            cx = 718
            cy = 965
        elif self.dtset == Dataset.NYU2:
            fx = 518.858
            fy = 519.470
            cx = 325.582
            cy = 253.736
        elif self.dtset == Dataset.KITTI:
            pass
            # fx = fy = 721.537
            # cx = (609.559 + 596.559) / 2.0
            # cy = 149.854
        else:
            raise ValueError("Unsupported dataset")

        intrinsics = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]], dtype=np.float32)

        return intrinsics

    #=======================================================================

    def getSamplePair(self, idx):

        if self.dtset == Dataset.IPHONE:

            rgbFileName = f"RGB_{idx:04d}.png"
            rgbPath = self.inputPath + rgbFileName 
            bgr = cv2.imread(rgbPath)
            bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)

            gtPath = self.inputPath + f"ARKit_DepthValues_{idx:04d}.txt" 
            gt = loadMatrixFromFile(gtPath)
            gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)

        elif self.dtset == Dataset.NYU2:

            rgbFileName = f"{idx:05d}_colors.png"
            rgbPath = self.inputPath + rgbFileName 
            bgr = cv2.imread(rgbPath)

            gtPath = self.inputPath + f"{idx:05d}_depth.png"
            gt = cv2.imread(gtPath, cv2.IMREAD_UNCHANGED)
            gt = gt.astype(np.float64) / 1000.0           # scale to meters

            margin = 8   # remove white margin
            bgr = bgr[margin:-margin, margin:-margin, :]
            gt = gt[margin:-margin, margin:-margin]

        elif self.dtset == Dataset.KITTI:

            rgbFileName = f"{idx:05d}_colors.png"
            rgbPath = self.inputPath + rgbFileName 
            bgr = cv2.imread(rgbPath)

            gtPath = self.inputPath + f"{idx:05d}_depth.png"
            gt = cv2.imread(gtPath, cv2.IMREAD_UNCHANGED)
            gt = gt.astype(np.float64) / 256.0           # scale to meters

        else:
            raise ValueError("Unsupported dataset")
        

        return bgr, gt
        
