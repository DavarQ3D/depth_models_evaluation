import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "methods", "unidepthv2"))

from custom_assets.datasets import Dataset, DataManager
from custom_assets.visualizer import Visualizer
from custom_assets.utils import *
from custom_assets.models import Model, ModelManager, AlignmentType
from custom_assets.analyzer import Analyzer, ErrorType

#=============================================================================================================

if __name__ == '__main__':

    outdir = "./data/outputs"
    os.makedirs(outdir, exist_ok=True)

    #--------------------- settings
    
    dtset = Dataset.IPHONE
    
    mdType = Model.Torch_UNIDEPTH_V2
    encoder = "vits"

    alignDepth = True                                 # alignment is highly encouraged even on the best metric depth models
    alignmentType = AlignmentType.Fitting   
    alignShift = True                                 # scale is always applied if alignDepth is True, shift is optional
    k_hi = 2.5 if dtset == Dataset.IPHONE else 3.0

    makeSquareInput = True and (mdType == Model.Torch_depthAnythingV2_Rel or mdType == Model.Torch_depthAnythingV2_Metric)
    borderType = cv2.BORDER_CONSTANT

    useIntrinsics = True and (dtset != Dataset.KITTI)

    errType = ErrorType.ABS_REL
    maxNumSamplesToAnalyze = 100
    showVisuals = False
    showPerImageCDE = True and (errType == ErrorType.ABS_REL)

    alignmentID = f"aligType[{alignmentType.name}]_alignShift[{alignShift}]" if alignDepth else "noAlignmentInDepthSpace"
    experimentID = f"{mdType.name}_{alignmentID}"
    experimentFolder = dtset.name
    folderPath = os.path.join(outdir, experimentFolder)

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

    elif mdType == Model.Torch_UNIDEPTH_V2:    
        mPath = f"lpiccinelli/unidepth-v2-{encoder}14"

    else:
        raise ValueError("Unsupported model")
    
    mdManager = ModelManager(mdType, mPath, encoder, max_depth)

    #--------------------- analyzer

    analyzer = Analyzer(errType)

    #------------------ inference loop
    #------------------------------------------------------------------
    upperBound = min(numFiles, maxNumSamplesToAnalyze) 

    for idx in range(0, upperBound):

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
            relDisparity, _ = mdManager.infer(bgr)
            bgr, relDisparity, gt = mdManager.alignShapes(bgr, relDisparity, gt)
            metricDepth = mdManager.alignInDisparitySpace(relDisparity, gt, staticMask, k_hi)

        elif mdType == Model.Torch_depthAnythingV2_Metric:
            bgr = mdManager.adaptShapeForInference(bgr, makeSquareInput, borderType)
            metricDepth, _ = mdManager.infer(bgr)
            bgr, metricDepth, gt = mdManager.alignShapes(bgr, metricDepth, gt)

        elif mdType == Model.Torch_UNIDEPTH_V2:
            gtIntrinsics = dtManager.getIntrinsics() if useIntrinsics else None
            metricDepth, intrinsics = mdManager.infer(bgr, gtIntrinsics)
            bgr, metricDepth, gt = mdManager.alignShapes(bgr, metricDepth, gt)     
            if dtset != Dataset.KITTI:
                analyzer.printIntrinsComparison(intrinsics, gtIntrinsics) 

        else:
            raise ValueError("Unsupported model type")

        #------------- metric depth alignment
        #-------------------------------------------------
        
        if alignDepth:
            metricDepth = mdManager.alignInDepthSpace(metricDepth, gt, staticMask, alignmentType, k_hi, alignShift, maxVal)
        
        #------------- error analysis
        #-------------------------------------------------

        errImage = analyzer.runAnalysis(metricDepth, gt, staticMask, idx)
        analyzer.sampleProcessed()

        if showVisuals:
            if showPerImageCDE:
                perImageCDE = analyzer.generateCDEgraph(analyzer.imgValidPixsErrs)
                visualizer.displayImage("Per Image CDE", perImageCDE, waitTime=1)
            sc = 2.5 if dtset == Dataset.IPHONE else 0.7 
            visualizer.showResults(bgr, metricDepth, gt, errImage, sc, vertConcat=(dtset == Dataset.KITTI))

    #--------------------- end of inference loop
    #----------------------------------------------------------------------

    os.makedirs(folderPath, exist_ok=True)
    writeVecOnDisk(folderPath + f"/{experimentID}.npy", analyzer.getDatasetErrors())    # save results to disk
    
    #---------- update evaluation graphs
    #------------------------------------------------------------------
    errDict = analyzer.loadErrorVecsFromFolderParallel(folderPath)
    combinedCDE = analyzer.generateCombingedCDEgraph(errDict)
    cv2.imwrite(folderPath + f"/{dtset.name}_data.png", combinedCDE)
    visualizer.displayImage(f"{dtset.name}_data", combinedCDE, waitTime=0)

        

