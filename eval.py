from custom_assets.visualizer import Visualizer
from custom_assets.utils import *
from custom_assets.analyzer import Analyzer

#=============================================================================================================

if __name__ == '__main__':

    visualizer = Visualizer()
    analyzer = Analyzer()
    folderPath = "./data/outputs/draft"

    #---------- update evaluation graphs
    #------------------------------------------------------------------
    errDict = analyzer.loadErrorVecsFromFolderParallel(folderPath)
    combinedCDE = analyzer.generateCombingedCDEgraph(errDict)
    cv2.imwrite(folderPath + f"/eval_result.png", combinedCDE)
    visualizer.displayImage(f"eval_result", combinedCDE, waitTime=0)

        

