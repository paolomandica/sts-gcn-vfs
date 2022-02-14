# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import RECOGNIZERS, build_head
from .base import BaseGCN
from mmaction.utils import add_prefix

import time


@RECOGNIZERS.register_module()
class SiamSkeletonGCN(BaseGCN):
    """Siamese Spatial temporal graph convolutional networks."""

    def __init__(self, backbone, sim_head=None, train_cfg=None, test_cfg=None):
        super().__init__(backbone, train_cfg, test_cfg)
        self.sim_head_cfg = sim_head
        if sim_head is not None:
            self.sim_head = build_head(sim_head)
        self.init_extra_weights()

    # TODO: adjust initialization of the stsgcn backbone and the head

    def init_weights(self):
        """Initialize the model network weights.
        
        Added for STSGCN.
        """
        if self.with_cls_head:
            self.cls_head.init_weights()

    @property
    def with_sim_head(self):
        """bool: whether the detector has sim head"""
        return hasattr(self, 'sim_head') and self.sim_head is not None
    
    def init_extra_weights(self):
        if self.with_sim_head:
            self.sim_head.init_weights()

    def forward_sim_head(self, x1, x2, T):

        B, C, _, V = x1.shape
        
        losses = dict()
        z1, p1 = self.sim_head(x1)
        z2, p2 = self.sim_head(x2) #.permute(0,2,1).flatten(0,1)   
        
        loss_weight = 1. / T

        losses.update(
            add_prefix(
                self.sim_head.loss(p1, z1, p2, z2, weight=loss_weight),
                prefix='0'))

        z2_v, p2_v = z2.view(B, T, C), p2.view(B, T, C)

        for i in range(1, T):
            losses.update(
                add_prefix(
                    self.sim_head.loss(
                        p1,
                        z1,
                        p2_v.roll(i, dims=1).flatten(0, 1),
                        z2_v.roll(i, dims=1).flatten(0, 1),
                        weight=loss_weight),
                    prefix=f'{i}'))
        
        return losses

    def forward_train(self, skeletons, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_sim_head
        losses = dict()

        # (64, 3, 300, 25, 2)
        B, C, T, V, M = skeletons.shape
        skeletons_1, skeletons_2 = skeletons[:,:,:T//2], skeletons[:,:,T//2:]

        x1 = self.extract_feat(skeletons_1)
        x2 = self.extract_feat(skeletons_2)

        # x1, x2 are (128, 256, 150, 25) = (BxM, C_tilde, T/2, V)

        if self.with_sim_head:
            loss_sim_head = self.forward_sim_head(x1, x2, self.sim_head_cfg.aggregation_time_out)
            losses.update(add_prefix(loss_sim_head, prefix='sim_head'))

        return losses

    def forward_test(self, skeletons):
        """Defines the computation performed at every call when evaluation and
        testing."""
        x = self.extract_feat(skeletons)
        assert self.with_cls_head
        output = self.cls_head(x)

        return output.data.cpu().numpy()
