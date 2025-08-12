
import numpy as np
from custom_assets.utils import *
from enum import Enum

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import MultipleLocator

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
        self.imgValidPixsErrs = None
        self.datasetValidPixsErrs = []

    #=======================================================================  

    def sampleProcessed(self):
        self.sampleCounter += 1    

    #=======================================================================  

    def generateCDEgraph(self, errors):
        errors = np.asarray(errors, dtype=np.float32)
        errors.sort()
        n = errors.size
        Ps = (np.arange(1, n + 1, dtype=np.float32) * 100.0) / n

        fig = Figure(figsize=(15, 9), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(errors, Ps, linestyle='-')

        ax.xaxis.set_major_locator(MultipleLocator(0.03))
        # ax.xaxis.set_major_locator(MaxNLocator(nbins=major_nbins))
        # ax.xaxis.set_minor_locator(AutoMinorLocator(minor_divisions))
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # nice labels

        ax.set_title("Cumulative Error Distribution")
        ax.set_xlabel("Error")
        ax.set_ylabel("Percentage of Samples")
        ax.set_yticks(np.linspace(0, 100, 21))
        ax.grid(True, which='major', linewidth=0.8, alpha=0.6)
        ax.grid(True, which='minor', linewidth=0.5, alpha=0.3)
        fig.tight_layout()

        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        bgr  = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        return bgr

    #=======================================================================  

    def runAnalysis(self, pred, ref, mask, idx):

        if self.errorType == ErrorType.ABS_REL:
            err = np.abs(ref - pred) / (ref + 1e-6)
            valid = err[mask]
            self.imgValidPixsErrs = valid
            self.datasetValidPixsErrs.append(valid)
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

