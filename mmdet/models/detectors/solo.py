from .single_stage_ins import SingleStageInsDetector
from ..builder import DETECTORS

@DETECTORS.register_module()
class SOLO(SingleStageInsDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)