
import numpy as np
from custom_assets.utils import *
from enum import Enum
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator
from concurrent.futures import ThreadPoolExecutor

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

        fig = Figure(figsize=(13, 9), dpi=75)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot()
        ax.plot(errors, Ps, linestyle='-')
        ax.set_ylim(0, 100)  

        ax.set_xlim(0, 1.0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=30))

        ax.set_title("Cumulative Error Distribution")
        ax.set_xlabel("Error")
        ax.set_ylabel("Percentage of Samples")
        ax.set_yticks(np.linspace(0, 100, 21))
        ax.grid(True, which='major', linewidth=0.8, alpha=0.6)
        ax.grid(True, which='minor', linewidth=0.5, alpha=0.3)
        fig.tight_layout()

        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
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

    def getDatasetErrors(self):
        return np.concatenate(self.datasetValidPixsErrs) 

    #======================================================================= 

    def printIntrinsComparison(self, pred, gt):
    
        print(f"\nPred --> fx: {fp(pred[0, 0], 3)}, fy: {fp(pred[1, 1], 3)}, cx: {fp(pred[0, 2], 3)}, cy: {fp(pred[1, 2], 3)}")
        print(f"GT   --> fx: {fp(gt[0, 0], 3)}, fy: {fp(gt[1, 1], 3)}, cx: {fp(gt[0, 2], 3)}, cy: {fp(gt[1, 2], 3)}\n")

    #======================================================================= 

    def loadErrorVecsFromFolder(self, folderPath):

        errorVecs = {}        

        for fileName in os.listdir(folderPath):
            if fileName.endswith(".txt"):
                filePath = os.path.join(folderPath, fileName)
                vec = loadVecFromFile(filePath)
                key = os.path.splitext(fileName)[0]
                errorVecs[key] = vec

        return errorVecs

    #======================================================================= 

    def loadErrorVecsFromFolderParallel(self, folderPath):
        # Accept both .npy (preferred) and .txt (legacy)
        exts = (".npy", ".npz", ".txt")
        files = [f for f in os.listdir(folderPath) if f.lower().endswith(exts)]
        if not files:
            return {}

        def _load(name):
            path = os.path.join(folderPath, name)
            vec = loadVecFromFile(path)
            key = os.path.splitext(name)[0]
            return key, np.asarray(vec, dtype=np.float32)

        # Threaded = good for I/O; numpy loads release the GIL during I/O
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            items = list(pool.map(_load, sorted(files)))  # sorted for stable legend order

        return dict(items)

    #======================================================================= 

    def generateCombingedCDEgraph(self, errDict):

        fig = Figure(figsize=(13, 9), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot()

        plotted_any = False

        # Use common bin edges across series (consistent with xlim=[0, 1.0])
        # Large arrays take the fast histogram path; small arrays keep the exact sort.
        bins = 4096
        x_min, x_max = 0.0, 0.72
        edges = np.linspace(x_min, x_max, bins + 1, dtype=np.float32)

        for label, arr in errDict.items():
            arr = np.asarray(arr, dtype=np.float32)
            if arr.size == 0:
                continue

            # ---------- FAST PATH: histogram-based CDF (no O(N log N) sort) ----------
            if arr.size > 200_000:
                # keep consistency with axis limits; values outside [0,1] would be clipped visually anyway
                arr = np.clip(arr, x_min, x_max)
                h, _ = np.histogram(arr, bins=edges)
                Ps = (np.cumsum(h, dtype=np.int64) * 100.0) / arr.size
                xs = edges[1:]
                ax.plot(xs, Ps, linestyle='-', label=str(label))

            # ---------- EXACT PATH: original behavior for smaller arrays ----------
            else:
                arr = np.sort(arr, kind='quicksort')  # returns a writable sorted copy
                n = arr.size
                Ps = (np.arange(1, n + 1, dtype=np.float32) * 100.0) / n
                ax.plot(arr, Ps, linestyle='-', label=str(label))

            plotted_any = True

        ax.set_ylim(0, 100)
        ax.set_xlim(0, x_max)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=25))

        ax.set_title("Cumulative Error Distribution")
        ax.set_xlabel("Error")
        ax.set_ylabel("Percentage of Samples")
        ax.set_yticks(np.linspace(0, 100, 21))
        ax.grid(True, which='major', linewidth=0.8, alpha=0.6)
        ax.grid(True, which='minor', linewidth=0.5, alpha=0.3)

        if plotted_any:
            ax.legend(loc='lower right', frameon=False)

        fig.tight_layout()
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        return bgr
