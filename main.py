from custom_assets.datasets import Dataset, DataManager
from custom_assets.visualizer import Visualizer
from custom_assets.utils import resizeImage
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

    #------------------ inference loop
    #------------------------------------------------------------------

    for idx in range(0, numFiles):

        print('\n'"========================================")
        print(f'============= sample --> {idx} =============')
        print("========================================", '\n')

        bgr, gt = dtManager.getSamplePair(idx)





