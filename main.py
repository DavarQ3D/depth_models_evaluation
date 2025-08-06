from custom_assets.datasets import Dataset, DataManager
from custom_assets.visualizer import Visualizer
from custom_assets.utils import resizeImage
from custom_assets.models import Model, ModelManager

import os

#=============================================================================================================

if __name__ == '__main__':

    dtset = Dataset.IPHONE
    outdir = "./data/outputs"
    os.makedirs(outdir, exist_ok=True)

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
    
    mdType = Model.Torch_depthAnythingV2_Rel
    encoder = "vits"
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

    #------------------ inference loop
    #------------------------------------------------------------------

    for idx in range(0, numFiles):

        print('\n'"========================================")
        print(f'============= sample --> {idx} =============')
        print("========================================", '\n')

        bgr, gt = dtManager.getSamplePair(idx)





