from custom_assets.datasets import Dataset, DataManager
from custom_assets.visualizer import Visualizer
from custom_assets.utils import *
from custom_assets.models import Model, ModelManager
from custom_assets.analyzer import Analyzer
import os

#=============================================================================================================

if __name__ == '__main__':

    outdir = "./data/outputs"
    os.makedirs(outdir, exist_ok=True)

    #--------------------- settings
    
    dtset = Dataset.KITTI
    
    mdType = Model.Torch_depthAnythingV2_Rel
    encoder = "vits"

    alignDepth = False or mdType != Model.Torch_depthAnythingV2_Rel
    fitScale = False
    fitShift = False
    k_hi = 2.5 if dtset == Dataset.IPHONE else 3.0

    makeSquareInput = True and (mdType == Model.Torch_depthAnythingV2_Rel or mdType == Model.Torch_depthAnythingV2_Metric)
    borderType = cv2.BORDER_CONSTANT

    showVisuals = True

    #--------------------- dataset 

    if dtset == Dataset.IPHONE:
        inputPath = f"./data/iphone/"
    elif dtset == Dataset.NYU2:
        inputPath = f"./data/nyu2_test/"
    elif dtset == Dataset.KITTI:
        inputPath = f"./data/kitti_variety/"
        # inputPath = f"./data/kitti_temporal/"
    else:
        raise ValueError("Unsupported dataset")

    dtManager = DataManager(dtset, inputPath)
    numFiles = dtManager.getNumFiles()

    #--------------------- visualizer 

    visualizer = Visualizer()

    #--------------------- model manager
    
    max_depth = None

    if mdType == Model.Torch_depthAnythingV2_Rel:
        mPath = f'./methods/relativeDepthAnythingV2/checkpoints/depth_anything_v2_{encoder}.pth'

    elif mdType == Model.Torch_depthAnythingV2_Metric:
        temp = "./methods/metricDepthAnythingV2/checkpoints" 
        mPath = f'{temp}/depth_anything_v2_metric_vkitti_{encoder}.pth' if dtset == Dataset.KITTI else f'{temp}/depth_anything_v2_metric_hypersim_{encoder}.pth'
        max_depth = 80 if dtset == Dataset.KITTI else 20

    else:
        raise ValueError("Unsupported model")
    
    mdManager = ModelManager(mdType, mPath, encoder, max_depth)

    #--------------------- analyzer

    analyzer = Analyzer()

    #------------------ inference loop
    #------------------------------------------------------------------

    for idx in range(0, numFiles):

        print('\n'"========================================")
        print(f'============= sample --> {idx} =============')
        print("========================================", '\n')

        bgr, gt = dtManager.getSamplePair(idx)

        #------------- static mask
        #-------------------------------------------------
        
        maxVal = 50.0 if dtset == Dataset.KITTI else 15.0
        staticMask, gt = getValidMaskAndClipExtremes(gt, minVal=0.01, maxVal=maxVal) 
        
        #------------- metric depth inference
        #-------------------------------------------------
        
        if mdType == Model.Torch_depthAnythingV2_Rel: 
            bgr = mdManager.adaptShapeForInference(bgr, makeSquareInput, borderType, dim=518)
            relDisparity = mdManager.infer(bgr)
            bgr, relDisparity, gt = mdManager.alignShapes(bgr, relDisparity, gt)
            metricDepth = mdManager.alignInDisparitySpace(relDisparity, gt, staticMask, k_hi)

        elif mdType == Model.Torch_depthAnythingV2_Metric:
            bgr = mdManager.adaptShapeForInference(bgr, makeSquareInput, borderType)
            metricDepth = mdManager.infer(bgr)
            bgr, metricDepth, gt = mdManager.alignShapes(bgr, metricDepth, gt)

        else:
            raise ValueError("Unsupported model type")

        #------------- metric depth alignment
        #-------------------------------------------------
        
        if alignDepth:
            metricDepth = mdManager.alignInDepthSpace(metricDepth, gt, staticMask, k_hi, fitScale, fitShift, maxVal)
        
        #------------- error analysis
        #-------------------------------------------------

        errImage = analyzer.runAnalysis(metricDepth, gt, staticMask, idx)
        analyzer.sampleProcessed()

        if showVisuals:
            sc = 2.5 if dtset == Dataset.IPHONE else 0.7 
            visualizer.showResults(bgr, metricDepth, gt, errImage, sc, vertConcat=(dtset == Dataset.KITTI))
        

