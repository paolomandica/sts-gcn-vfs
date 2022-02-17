# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import RECOGNIZERS
from .base import BaseGCN

import time


@RECOGNIZERS.register_module()
class SkeletonGCN(BaseGCN):
    """Spatial temporal graph convolutional networks."""

    def init_weights(self):
        super().init_weights()
        self.backbone.freeze_weights()

    def forward_train(self, skeletons, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        losses = dict()

        B, C, T, V, M = skeletons.shape
        skeletons = skeletons.view(B, C, -1, T//10, V, M)
        B, C, N, T, V, M = skeletons.shape
        skeletons = skeletons.reshape(B*N, C, T, V, M)

        x = self.extract_feat(skeletons)
        output = self.cls_head(x)
        output = output.view(B, N, -1).mean(dim=1)
        gt_labels = labels.squeeze(-1)
        loss = self.cls_head.loss(output, gt_labels)
        losses.update(loss)

        return losses

    def forward_test(self, skeletons):
        """Defines the computation performed at every call when evaluation and
        testing."""

        B, C, T, V, M = skeletons.shape
        skeletons = skeletons.view(B, C, -1, T//10, V, M)
        B, C, N, T, V, M = skeletons.shape
        skeletons = skeletons.reshape(B*N, C, T, V, M)

        x = self.extract_feat(skeletons)
        assert self.with_cls_head
        output = self.cls_head(x)

        return output.data.cpu().numpy()
