from custom_assets.utils import *
import torch
from methods.unidepthV2.unidepth.models import UniDepthV2
from methods.unidepthV2.unidepth.utils.camera import Pinhole
from enum import Enum

class Model(Enum):
    Torch_depthAnythingV2_Rel = 1
    Torch_depthAnythingV2_Metric = 2
    Torch_UNIDEPTH_V2 = 3

class AlignmentType(Enum):
    MedianBased = 1
    Fitting = 2    

#==============================================================================================
#==============================================================================================

class ModelManager:
    def __init__(self, mdType, mPath, encoder=None, max_depth=None):

        #--- stuff related to depthAnythingV2
        #------------------------------------------
        self.makeSquareInput = None
        self.pad_top = None
        self.pad_left = None
        self.crop_top = None
        self.crop_left = None
        self.dim = None
        self.resizedShape = None
        #------------------------------------------

        self.mdType = mdType
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        if mdType == Model.Torch_depthAnythingV2_Rel or mdType == Model.Torch_depthAnythingV2_Metric:
            self.model = self.loadDepthAnyTorchModel(mPath, encoder, max_depth)
        
        elif mdType == Model.Torch_UNIDEPTH_V2:
            self.model = UniDepthV2.from_pretrained(mPath)
            self.model.interpolation_mode = "bilinear"      # set interpolation mode (only V2)
            self.model.resolution_level = 9                 # set resolution level [0-9] (only V2)

        else:
            raise ValueError("Unsupported model type")

        self.model = self.model.to(self.device).eval()

    #=======================================================================

    def loadDepthAnyTorchModel(self, modelPath, encoder, max_depth=None):
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }    

        if max_depth is None:
            from methods.relativeDepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2                # relative depth head
            model = DepthAnythingV2(**model_configs[encoder])
        else:
            from methods.metricDepthAnythingV2.depth_anything_v2 .dpt import DepthAnythingV2                 # metric depth head
            model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    
        model.load_state_dict(torch.load(modelPath, map_location='cpu'))
        
        return model.eval()    

    #=======================================================================

    def adaptShapeForInference(self, image, makeSquare, borderType, dim=518):

        self.dim = dim

        if makeSquare:
            sc = dim / max(image.shape[:2])
            resized = cv2.resize(image, (int(image.shape[1] * sc), int(image.shape[0] * sc)), interpolation=cv2.INTER_CUBIC)
            self.resizedShape = resized.shape[:2]
            r = dim
            c = dim
            padded, pad_top, pad_left, _, _ = center_crop_or_pad(resized, r, c, borderType)
            self.makeSquareInput = True
            self.pad_top = pad_top
            self.pad_left = pad_left
            return padded

        else:    
            sc = dim / min(image.shape[:2])
            resized = cv2.resize(image, (int(image.shape[1] * sc), int(image.shape[0] * sc)), interpolation=cv2.INTER_CUBIC)
            self.resizedShape = resized.shape[:2]
            r = ensure_multiple_of(resized.shape[0], multiple_of=14)
            c = ensure_multiple_of(resized.shape[1], multiple_of=14)
            cropped, _, _, crop_top, crop_left = center_crop_or_pad(resized, r, c)
            self.makeSquareInput = False
            self.crop_top = crop_top
            self.crop_left = crop_left
            return cropped

    #=======================================================================

    def infer(self, bgr, intrinsics=None):      

        if self.mdType == Model.Torch_depthAnythingV2_Rel or self.mdType == Model.Torch_depthAnythingV2_Metric:
            return self.model.infer_image(bgr, self.dim, doResize=False), None

        elif self.mdType == Model.Torch_UNIDEPTH_V2:
            
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)       # convert bgr to rgb
            rgbTensor = torch.from_numpy(np.array(rgb))
            rgbTensor = rgbTensor.permute(2, 0, 1)           # c, h, w

            camera = None
            if intrinsics is not None:
                K = torch.from_numpy(intrinsics).float().to(self.device)  # 3 x 3
                camera = Pinhole(K=K)
            
            with torch.no_grad():
                predictions = self.model.infer(rgbTensor, camera=camera)

            depth = predictions["depth"]                                 # metric depth
            depth = depth[0,0].cpu().numpy()
            
            xyz = predictions["points"]                                  # point cloud in camera coordinates
            xyz = xyz[0].cpu().permute(1, 2, 0).numpy()

            intrinsics = predictions["intrinsics"]                       # intrinsics prediction
            intrinsics = intrinsics[0].cpu().numpy()

            return depth, intrinsics

        else:
            raise ValueError("Unsupported model type for inference")
            
    #=======================================================================  

    def alignShapes(self, image, pred, gt):

        if self.mdType == Model.Torch_depthAnythingV2_Rel or self.mdType == Model.Torch_depthAnythingV2_Metric:

            resizedShape = self.resizedShape

            if self.makeSquareInput:
                pad_top = self.pad_top
                pad_left = self.pad_left
                pred  = pred [pad_top : pad_top + resizedShape[0], pad_left: pad_left + resizedShape[1]]
                image = image[pad_top : pad_top + resizedShape[0], pad_left: pad_left + resizedShape[1], :]
            else:
                crop_top = self.crop_top
                crop_left = self.crop_left
                vertMarg  = (crop_top * 2) / (resizedShape[0] / gt.shape[0])
                horizMarg = (crop_left * 2) / (resizedShape[1] / gt.shape[1])
                vertMarg = round(vertMarg / 2)                                # round to the nearest even number
                horizMarg = round(horizMarg / 2)
                gt = gt[vertMarg or None : (-vertMarg) or None, horizMarg or None : (-horizMarg) or None]    
        
        pred  = cv2.resize(pred,  (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        image = cv2.resize(image, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)

        return image, pred, gt
          
    #=======================================================================        

    def alignInDisparitySpace(self, pred_disparity, gt_depth, mask, k_hi):

        gt_disparity = 1 / (gt_depth + 1e-8)                      # convert depth to disparity (inverse depth)
        predDisparityMask, pred_disparity = getValidMaskAndClipExtremes(pred_disparity, minVal=0.01, maxVal=100) 
        mask = mask & predDisparityMask

        scale, shift, mask = weightedLeastSquared(pred_disparity, gt_disparity, guessInitPrms=True, k_lo=0.2, k_hi=k_hi, num_iters=10, fit_shift=True, verbose=False, mask=mask)
        # scale, shift, mask = estimateParametersRANSAC(pred_disparity, gt_disparity) 

        pred_disparity = scale * pred_disparity + shift
        pred_depth = 1 / (pred_disparity + 1e-8)            # convert back to depth
        return pred_depth
    
    #=======================================================================  

    def alignInDepthSpace(self, pred, gt, mask, alignmentType, k_hi, alignShift, maxVal):
    
        predMask, pred = getValidMaskAndClipExtremes(pred, minVal=0.01, maxVal=maxVal)
        mask = mask & predMask
        
        if alignmentType == AlignmentType.Fitting:
            scale, shift, mask = weightedLeastSquared(pred, gt, guessInitPrms=True, k_lo=0.2, k_hi=k_hi, num_iters=10, fit_shift=alignShift, verbose=False, mask=mask)

        elif alignmentType == AlignmentType.MedianBased:
            x = pred[mask].ravel()  
            y = gt[mask].ravel()   
            scale, shift = estimateInitialParams(x, y, alignShift=alignShift)     

        else:
            raise ValueError("Unsupported alignment type")

        print("Scale:", fp(scale), ", Shift:", fp(shift), '\n')

        pred = scale * pred + shift     
        return pred
