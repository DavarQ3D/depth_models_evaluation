import cv2
from custom_assets.utils import *

#==============================================================================================
#==============================================================================================

class Visualizer:
    def __init__(self):
        pass
    
    #=======================================================================

    def displayImage(self, title, image):
        cv2.imshow(title, image)
        key = cv2.waitKey(0)
        if key == 27:  
            cv2.destroyAllWindows()
            exit()

    #=======================================================================

    def showResults(self, bgr, metricDepth, gt, errImage, sc, vertConcat):
        metricDepth = makeMultiChannelImage(metricDepth)
        gt = makeMultiChannelImage(gt)
        result = cv2.vconcat([bgr, metricDepth, gt, errImage]) if vertConcat else cv2.hconcat([bgr, metricDepth, gt, errImage])  
        result = cv2.resize(result, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        displayImage("visualRes", result)      