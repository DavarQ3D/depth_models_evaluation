
import numpy as np
from custom_assets.utils import *
from enum import Enum

class ErrorType(Enum):
    ABS_REL = 1
    RMSE = 2

#==============================================================================================
#==============================================================================================

class Analyzer:
    def __init__(self, errorType=ErrorType.ABS_REL):
        self.totalError = 0.0
        self.sampleWithLowestError = 0
        self.samplewithHighestError = 0
        self.minErr = float('inf')
        self.maxErr = float('-inf')
        self.sampleCounter = 0
        self.errorType = errorType

    #=======================================================================  

    def sampleProcessed(self):
        self.sampleCounter += 1    

    #=======================================================================  

    def runAnalysis(self, pred, ref, mask, idx):

        if self.errorType == ErrorType.ABS_REL:
            err = np.abs(ref - pred) / (ref + 1e-6)
            valid = err[mask]
            perImgError = valid.mean()
        elif self.errorType == ErrorType.RMSE:
            err = np.abs(ref - pred)
            valid = err[mask]
            perImgError = np.sqrt((valid**2).mean())   
        else:
            raise ValueError("Unsupported error type")

        print("ErrType:", self.errorType.name, "--> per image (valid pixels) err =", fp(perImgError, 6))    

        if perImgError < self.minErr:
            self.minErr = perImgError
            self.sampleWithLowestError = idx
        if perImgError > self.maxErr:
            self.maxErr = perImgError
            self.samplewithHighestError = idx

        self.totalError += perImgError
        meanErr = self.totalError / (self.sampleCounter + 1)
        print("\ndataset err (mean) so far --> err =", fp(meanErr, 6))
        print("\nimage with lowest error:", self.sampleWithLowestError, "--> err =", fp(self.minErr, 6))
        print("image with highest error:", self.samplewithHighestError, "--> err =", fp(self.maxErr, 6))

        err = np.clip(err, 0, valid.max())
        err = normalize(err) 
        err = denormalize(err)
        err = cv2.cvtColor(err, cv2.COLOR_GRAY2BGR)
        err = cv2.applyColorMap(err, cv2.COLORMAP_JET)
        err[mask == 0] = 0      

        return err
    
    #=======================================================================  

    def printIntrinsComparison(self, pred, gt):
    
        print(f"\nPred --> fx: {fp(pred[0, 0], 3)}, fy: {fp(pred[1, 1], 3)}, cx: {fp(pred[0, 2], 3)}, cy: {fp(pred[1, 2], 3)}")
        print(f"GT   --> fx: {fp(gt[0, 0], 3)}, fy: {fp(gt[1, 1], 3)}, cx: {fp(gt[0, 2], 3)}, cy: {fp(gt[1, 2], 3)}\n")

