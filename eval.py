from custom_assets.visualizer import Visualizer
from custom_assets.utils import *
from custom_assets.analyzer import Analyzer
import os

#=============================================================================================================

if __name__ == '__main__':

    analyzer = Analyzer()
    visualizer = Visualizer()
    path = "./data/outputs/graphs"

    errDict = analyzer.loadErrorVecsFromFolder(path)
    combinedCDE = analyzer.generateCombingedCDEgraph(errDict)
    cv2.imwrite(os.path.join(path, "result.png"), combinedCDE)
    visualizer.displayImage("result", combinedCDE, waitTime=0)
    