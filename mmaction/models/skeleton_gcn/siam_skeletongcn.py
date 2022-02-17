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

    def compact_clips(self, x, N):
        return x.view(x.shape[0]//N, N, -1).mean(dim=1)

    def batch2clips(self, x, N):
        return x.view(x.shape[0]//N, N, -1)

    def clis2batch(self, x):
        return x.view(x.shape[0]*x.shape[1], -1)

    def forward_sim_head(self, x1, x2, N):
        # (BxNxM, C_h, T, V)
        B, C, T, V = x1.shape

        losses = dict()
        z1, p1 = self.sim_head(x1)
        z2, p2 = self.sim_head(x2)

        # z1, p1 = self.compact_clips(z1, N), self.compact_clips(p1, N)
        # z2, p2 = self.compact_clips(z2, N), self.compact_clips(p2, N)

        # loss_weight = 1. / T
        loss_weight = 1.

        losses.update(
            add_prefix(
                self.sim_head.loss(p1, z1, p2, z2, weight=loss_weight),
                prefix='0'))

        z2_v, p2_v = self.batch2clips(z2, N), self.batch2clips(p2, N)
        for i in range(1, N):
            losses.update(
                add_prefix(
                    self.sim_head.loss(
                        p1,
                        z1,
                        self.clis2batch(p2_v.roll(i, dims=1)),
                        self.clis2batch(z2_v.roll(i, dims=1)),
                        weight=loss_weight),
                    prefix=f'{i}'))

        return losses

    def forward_train(self, skeletons, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_sim_head
        losses = dict()

        # (64, 3, 300, 25, 2)
        B, C, T, V, M = skeletons.shape
        skeletons = skeletons.view(B, C, -1, T//10, V, M)
        B, C, N, T, V, M = skeletons.shape
        skeletons_1 = skeletons[:, :, :N//2].permute(0, 2, 1, 3, 4, 5).reshape(B*N//2, C, T, V, M)
        skeletons_2 = skeletons[:, :, N//2:].permute(0, 2, 1, 3, 4, 5).reshape(B*N//2, C, T, V, M)

        x1 = self.extract_feat(skeletons_1)
        x2 = self.extract_feat(skeletons_2)

        if self.with_sim_head:
            loss_sim_head = self.forward_sim_head(
                x1, x2, N//2)
            losses.update(add_prefix(loss_sim_head, prefix='sim_head'))

        return losses

    def forward_test(self, skeletons):
        """Defines the computation performed at every call when evaluation and
        testing."""
        x = self.extract_feat(skeletons)
        assert self.with_cls_head
        output = self.cls_head(x)

        return output.data.cpu().numpy()
