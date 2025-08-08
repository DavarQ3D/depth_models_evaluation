
import numpy as np
from custom_assets.utils import *

#==============================================================================================
#==============================================================================================

class Analyzer:
    def __init__(self):
        self.totalError = 0.0
        self.meanErr = 0.0
        self.sampleWithLowestError = 0
        self.samplewithHighestError = 0
        self.minRMSE = float('inf')
        self.maxRMSE = float('-inf')
        self.sampleCounter = 0

    #=======================================================================  

    def sampleProcessed(self):
        self.sampleCounter += 1    

    #=======================================================================  

    def runAnalysis(self, pred, ref, mask, idx):

        err = np.abs(ref - pred)
        valid = err[mask]
        rmse = np.sqrt((valid**2).mean())
        print("valid pixels RMSE =", fp(rmse, 6))

        if rmse < self.minRMSE:
            self.minRMSE = rmse
            self.sampleWithLowestError = idx
        if rmse > self.maxRMSE:
            self.maxRMSE = rmse
            self.samplewithHighestError = idx

        self.totalError += rmse
        meanErr = self.totalError / (self.sampleCounter + 1)
        self.sampleCounter += 1
        print("\nmean across all images so far --> RMSE =", fp(meanErr, 6))
        print("\nimage with lowest error:", self.sampleWithLowestError, "--> RMSE =", fp(self.minRMSE, 6))
        print("image with highest error:", self.samplewithHighestError, "--> RMSE =", fp(self.maxRMSE, 6))

        err = np.clip(err, 0, valid.max())
        err = normalize(err) 
        err = denormalize(err)
        err = cv2.cvtColor(err, cv2.COLOR_GRAY2BGR)
        err = cv2.applyColorMap(err, cv2.COLORMAP_JET)
        err[mask == 0] = 0      

        return err
    
    #=======================================================================  

    def compareIntrinsics(self, pred, gt):
    
        print(f"\nPred --> fx: {fp(pred[0, 0], 3)}, fy: {fp(pred[1, 1], 3)}, cx: {fp(pred[0, 2], 3)}, cy: {fp(pred[1, 2], 3)}")
        print(f"GT   --> fx: {fp(gt[0, 0], 3)}, fy: {fp(gt[1, 1], 3)}, cx: {fp(gt[0, 2], 3)}, cy: {fp(gt[1, 2], 3)}\n")

