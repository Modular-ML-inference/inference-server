from inference_application.code.inferencers.base_inferencer import BaseInferencer
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import gc
import os
import numpy as np
import torch


class TorchRCNNInferencer(BaseInferencer):

    def load_model(self, path, use_cuda=True):
        for p in os.listdir(path):
            # check if current path is a file
            if os.path.isfile(os.path.join(path, p)):
                is_cuda = use_cuda and torch.cuda.is_available()
                device = "cuda" if is_cuda else "cpu"
                scratch = torch.load(model_path=os.path.join(path, p), map_location=torch.device(device))
                model = self.get_model_instance_segmentation()
                model.load_state_dict(scratch['model_state_dict'])
                model.eval()
                model.to(device)
        return model

    def prepare(self, inferencer, format):
        self._format = format
        self.inferencer = inferencer

    def format(self):
        return self._format

    def predict(self, data):
        with torch.no_grad():
            outputs = self.inferencer(data)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return outputs
    
    
    def get_model_instance_segmentation(num_classes=2):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)#pretrained=True)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.rpn.nms_thresh = 0.6
        model.rpn.score_thresh = 0.01
        model.rpn.anchor_generator.aspect_ratios = ((0.2, 0.5, 1.0, 2.0, 3.0, 4.0), (0.2, 0.5, 1.0, 2.0, 3.0, 4.0), 
                                                    (0.2, 0.5, 1.0, 2.0, 3.0, 4.0), (0.2, 0.5, 1.0, 2.0, 3.0, 4.0), 
                                                    (0.2, 0.5, 1.0, 2.0, 3.0, 4.0))
        model.rpn.anchor_generator.sizes = ((32,), (64,), (128,), (256,), (512,))
        model.rpn._pre_nms_top_n['training'] = 2000
        model.rpn._pre_nms_top_n['testing'] = 1000
        model.rpn._post_nms_top_n['training'] = 1000
        model.rpn._post_nms_top_n['testing'] = 500


        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.roi_heads.detections_per_img = 12 
        model.roi_heads.box_detections_per_img = 12
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)

        return model