from enum import Enum
import torch

class Model(Enum):
    Torch_depthAnythingV2_Rel = 1
    Torch_depthAnythingV2_Metric = 2
    Torch_UNIDEPTH_V2 = 3

#==============================================================================================
#==============================================================================================

class ModelManager:
    def __init__(self, mdType, mPath, encoder, max_depth=None):

        self.mdType = mdType
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        if mdType == Model.Torch_depthAnythingV2_Rel or mdType == Model.Torch_depthAnythingV2_Metric:
            self.model = self.loadDepthAnyTorchModel(mPath, encoder, max_depth)

        else:
            raise ValueError("Unsupported model type")

        self.model = self.model.to(self.device).eval()

    #=======================================================================

    def loadDepthAnyTorchModel(modelPath, encoder, max_depth=None):
        
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
        
        return model    
    

    #=======================================================================

    # def infer(self, image):

    #     with torch.no_grad():
    #         image = image.to(self.device).unsqueeze(0)