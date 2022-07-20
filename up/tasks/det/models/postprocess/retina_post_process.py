import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from up.utils.general.registry_factory import MODULE_PROCESS_REGISTRY
from up.utils.model import accuracy as A  # noqa F401
from up.tasks.det.models.utils.anchor_generator import build_anchor_generator
from up.models.losses import build_loss
from up.tasks.det.models.losses.entropy_loss import apply_class_activation
from up.tasks.det.models.utils.bbox_helper import offset2bbox, bbox_iou_overlaps
from up.utils.env.dist_helper import allreduce, env
from .roi_supervisor import build_roi_supervisor
from .roi_predictor import build_roi_predictor


__all__ = ['BaseDetPostProcess', 'IOUPostProcess', 'RPNPostProcess']


@MODULE_PROCESS_REGISTRY.register('retina_post')
class BaseDetPostProcess(nn.Module):
    """
    Head for the first stage detection task

    .. note::

        0 is always for the background class.
    """

    def __init__(self, num_classes, cfg, prefix=None, class_first=False):
        super(BaseDetPostProcess, self).__init__()
        self.prefix = prefix if prefix is not None else self.__class__.__name__
        self.class_first = class_first
        self.num_classes = num_classes
        assert self.num_classes > 1

        self.anchor_generator = build_anchor_generator(cfg['anchor_generator'])
        self.supervisor = build_roi_supervisor(cfg['roi_supervisor'])
        # rpn needs predict when training
        if 'train' in cfg:
            train_cfg = copy.deepcopy(cfg)
            train_cfg.update(train_cfg['train'])
            self.train_predictor = build_roi_predictor(train_cfg['roi_predictor'])
        else:
            self.train_predictor = None

        test_cfg = copy.deepcopy(cfg)
        test_cfg.update(test_cfg.get('test', {}))
        self.test_predictor = build_roi_predictor(test_cfg['roi_predictor'])

        self.cls_loss = build_loss(cfg['cls_loss'])
        self.loc_loss = build_loss(cfg['loc_loss'])
        self.cls_use_ghm = 'ghm' in self.cls_loss.name
        self.loc_use_ghm = 'ghm' in self.loc_loss.name
        self.cls_use_qfl = 'quality_focal_loss' in self.cls_loss.name

    @property
    def num_anchors(self):
        return self.anchor_generator.num_anchors

    @property
    def class_activation(self):
        return self.cls_loss.activation_type

    @property
    def with_background_channel(self):
        return self.class_activation == 'softmax'

    def apply_activation(self, mlvl_preds, remove_background_channel_if_any):
        """apply activation on permuted class prediction
        Arguments:
            - mlvl_pred (list (levels) of tuple (predictions) of Tensor): first element must
            be permuted class prediction with last dimension be class dimension
        """
        mlvl_activated_preds = []
        for lvl_idx, preds in enumerate(mlvl_preds):
            cls_pred = apply_class_activation(preds[0], self.class_activation)
            if self.with_background_channel and remove_background_channel_if_any:
                cls_pred = cls_pred[..., 1:]
            mlvl_activated_preds.append((cls_pred, *preds[1:]))
        return mlvl_activated_preds

    def permute_preds(self, mlvl_preds):
        if not self.class_first:
            """Permute preds from [B, A*C, H, W] to [B, H*W*A, C] """
            mlvl_permuted_preds, mlvl_shapes = [], []
            for lvl_idx, preds in enumerate(mlvl_preds):
                b, _, h, w = preds[0].shape
                k = self.num_anchors * h * w
                preds = [p.permute(0, 2, 3, 1).contiguous().view(b, k, -1) for p in preds]
                mlvl_permuted_preds.append(preds)
                mlvl_shapes.append((h, w, k))
            return mlvl_permuted_preds, mlvl_shapes
        else:
            """Permute preds from [B, C*A, H, W] to [B, H*W*A, C] """
            """Permute preds from [B, C*A*4, H, W] to [B, H*W*A, C*4]"""
            mlvl_permuted_preds, mlvl_shapes = [], []
            for lvl_idx, preds in enumerate(mlvl_preds):
                b, _, h, w = preds[0].shape
                k = self.num_anchors * h * w

                cls, loc = preds
                cls = cls.view(b, -1, self.num_anchors, h, w)
                cls = cls.permute(0, 3, 4, 2, 1).contiguous().view(b, k, -1)
                loc = loc.view(b, -1, self.num_anchors, 4, h, w)
                loc = loc.permute(0, 4, 5, 2, 1, 3).contiguous().view(b, k, -1)
                preds = [cls, loc]

                mlvl_permuted_preds.append(preds)
                mlvl_shapes.append((h, w, k))
            return mlvl_permuted_preds, mlvl_shapes

    def export(self, mlvl_preds):
        output = {}
        for idx, preds in enumerate(mlvl_preds):
            cls_pred, loc_pred = preds[:2]
            if self.class_activation == 'sigmoid':
                cls_pred = cls_pred.sigmoid()
            else:
                assert self.class_activation == 'softmax'
                _, _, h, w = cls_pred.shape
                c = cls_pred.shape[1] // self.num_anchors
                cls_pred = cls_pred.view(-1, c, h, w).permute(0, 2, 3, 1).contiguous()
                cls_pred = F.softmax(cls_pred, dim=-1)
                cls_pred = cls_pred.permute(0, 3, 1, 2).contiguous().view(-1, self.num_anchors * c, h, w)
            output[self.prefix + '.blobs.cls' + str(idx)] = cls_pred
            output[self.prefix + '.blobs.loc' + str(idx)] = loc_pred
        output['base_anchors'] = self.anchor_generator.export()
        return output

    def forward(self, input):
        strides = input['strides']
        image_info = input['image_info']
        mlvl_raw_preds = input['preds']
        device = mlvl_raw_preds[0][0].device

        # [B, hi*wi*A, :]
        mlvl_preds, mlvl_shapes = self.permute_preds(mlvl_raw_preds)
        mlvl_shapes = [(*shp, s) for shp, s in zip(mlvl_shapes, strides)]

        # [hi*wi*A, 4], for C4 there is only one layer, for FPN generate anchors for each layer
        mlvl_anchors = self.anchor_generator.get_anchors(mlvl_shapes, device=device)
        self.mlvl_anchors = mlvl_anchors
        output = {}

        if self.training:
            targets = self.supervisor.get_targets(mlvl_anchors, input, mlvl_preds)
            losses = self.get_loss(targets, mlvl_preds, mlvl_shapes)
            output.update(losses)

            if self.train_predictor is not None:
                mlvl_preds = self.apply_activation(mlvl_preds, remove_background_channel_if_any=True)
                results = self.train_predictor.predict(mlvl_anchors, mlvl_preds, image_info)
                output.update(results)
                if 'RPN' in self.prefix:
                    output.update({'rpn_dt_bboxes': output['dt_bboxes']})
        else:
            mlvl_preds = self.apply_activation(mlvl_preds, remove_background_channel_if_any=True)
            results = self.test_predictor.predict(mlvl_anchors, mlvl_preds, image_info)
            output.update(results)
            if 'RPN' in self.prefix:
                output.update({'rpn_dt_bboxes': output['dt_bboxes']})
        return output

    def get_loss(self, targets, mlvl_preds, mlvl_shapes=None):
        mlvl_shapes = [shape[:-1] for shape in mlvl_shapes]
        cls_target, loc_target, cls_mask, loc_mask = targets
        mlvl_cls_pred, mlvl_loc_pred = zip(*mlvl_preds)
        cls_pred = torch.cat(mlvl_cls_pred, dim=1)
        loc_pred = torch.cat(mlvl_loc_pred, dim=1)
        del mlvl_cls_pred, mlvl_loc_pred

        pos_normalizer = max(1, torch.sum(loc_mask.float()).item())

        # cls loss
        if self.cls_use_ghm:
            cls_loss = self.cls_loss(cls_pred, cls_target, mlvl_shapes=mlvl_shapes)
            acc = self._get_acc(cls_pred, cls_target)
        elif self.cls_use_qfl:
            B, K, _ = cls_pred.size()
            scores = cls_pred.new_zeros((B, K))
            pos_loc_target, pos_loc_pred = self._mask_tensor([loc_target, loc_pred], loc_mask)
            if pos_loc_pred.numel():
                pos_decode_loc_target, pos_decode_loc_pred = self.decode_loc(pos_loc_target, pos_loc_pred.detach(), loc_mask)  # noqa
                scores[loc_mask] = bbox_iou_overlaps(pos_decode_loc_pred, pos_decode_loc_target, aligned=True)
            cls_loss = self.cls_loss(cls_pred, cls_target, normalizer_override=pos_normalizer, scores=scores)
            acc = self._get_acc(cls_pred, cls_target)
        else:
            cls_loss = self.cls_loss(cls_pred, cls_target, normalizer_override=pos_normalizer)
            acc = self._get_acc(cls_pred, cls_target)

        # loc loss
        if self.loc_use_ghm:
            loc_loss = self.loc_loss(loc_pred, loc_target, loc_mask=loc_mask, mlvl_shapes=mlvl_shapes)
        else:
            loc_target, loc_pred = self._mask_tensor([loc_target, loc_pred], loc_mask)
            if loc_mask.sum() == 0:
                loc_loss = loc_pred.sum()
            else:
                loc_loss_key_fields = getattr(self.loc_loss, "key_fields", set())
                loc_loss_kwargs = {}
                if "anchor" in loc_loss_key_fields:
                    loc_loss_kwargs["anchor"] = self.get_sample_anchor(loc_mask)
                loc_loss = self.loc_loss(loc_pred, loc_target, normalizer_override=pos_normalizer, **loc_loss_kwargs)
        return {
            self.prefix + '.cls_loss': cls_loss,
            self.prefix + '.loc_loss': loc_loss,
            self.prefix + '.accuracy': acc
        }

    def _get_acc(self, cls_pred, cls_target):
        acc = A.accuracy_v2(cls_pred, cls_target, self.class_activation)
        return acc

    def _mask_tensor(self, tensors, mask):
        """
        Arguments:
            - tensor: [[M, N], K]
            - mask: [[M, N]]
        """
        mask = mask.reshape(-1)
        masked_tensors = []
        for tensor in tensors:
            n_dim = tensor.shape[-1]
            tensor = tensor.reshape(-1, n_dim)
            masked_tensor = tensor[mask]
            masked_tensors.append(masked_tensor)
        return masked_tensors

    def get_sample_anchor(self, loc_mask):
        all_anchors = torch.cat(self.mlvl_anchors, dim=0)
        batch_anchor = all_anchors.view(1, -1, 4).expand((loc_mask.shape[0], loc_mask.shape[1], 4))
        sample_anchor = batch_anchor[loc_mask].contiguous()
        return sample_anchor

    def decode_loc(self, loc_target, loc_pred, loc_mask):
        sample_anchor = self.get_sample_anchor(loc_mask)
        decode_loc_pred = offset2bbox(sample_anchor, loc_pred)
        decode_loc_target = offset2bbox(sample_anchor, loc_target)
        return decode_loc_target, decode_loc_pred


@MODULE_PROCESS_REGISTRY.register('retina_post_iou')
class IOUPostProcess(BaseDetPostProcess):
    """
    Head for the first stage detection task

    .. note::

        0 is always for the background class.
    """

    def __init__(self, num_classes, cfg, prefix=None, class_first=True):
        super(IOUPostProcess, self).__init__(num_classes, cfg, prefix, class_first)
        self.iou_branch_loss = build_loss(cfg['iou_branch_loss'])

    def apply_activation(self, mlvl_preds, remove_background_channel_if_any):
        mlvl_activated_preds = []
        for lvl_idx, preds in enumerate(mlvl_preds):
            cls_pred = apply_class_activation(preds[0], self.class_activation)
            iou_pred = apply_class_activation(preds[2], 'sigmoid')
            mlvl_activated_preds.append((cls_pred * iou_pred, *preds[1:]))
        return mlvl_activated_preds

    def export(self, mlvl_preds):
        output = {}
        for idx, preds in enumerate(mlvl_preds):
            cls_pred, loc_pred, iou_pred = preds[:]
            if self.class_activation == 'sigmoid':
                _, c, h, w = cls_pred.shape
                iou_pred = F.sigmoid(iou_pred.view(-1, 1, h, w)).view(-1, self.num_anchors * 1, h, w)
                cls_pred = cls_pred.sigmoid() * torch.cat([iou_pred for _ in range(c)], dim=1)
            else:
                raise NotImplementedError
            output[self.prefix + '.blobs.cls' + str(idx)] = cls_pred
            output[self.prefix + '.blobs.loc' + str(idx)] = loc_pred
        output['base_anchors'] = self.anchor_generator.export()
        return output

    def get_loss(self, targets, mlvl_preds, mlvl_shapes=None):
        mlvl_shapes = [shape[:-1] for shape in mlvl_shapes]
        cls_target, loc_target, iou_target, cls_mask, loc_mask = targets
        mlvl_cls_pred, mlvl_loc_pred, mlvl_iou_pred = zip(*mlvl_preds)
        cls_pred = torch.cat(mlvl_cls_pred, dim=1)
        loc_pred = torch.cat(mlvl_loc_pred, dim=1)
        iou_pred = torch.cat(mlvl_iou_pred, dim=1)

        del mlvl_cls_pred, mlvl_loc_pred, mlvl_iou_pred

        pos_normalizer = self.get_ave_normalizer(loc_mask)

        # cls loss
        if self.cls_use_ghm:
            cls_loss = self.cls_loss(cls_pred, cls_target, mlvl_shapes=mlvl_shapes)
            acc = self._get_acc(cls_pred, cls_target)
        elif self.cls_use_qfl:
            B, K, _ = cls_pred.size()
            scores = cls_pred.new_zeros((B, K))
            pos_loc_target, pos_loc_pred = self._mask_tensor([loc_target, loc_pred], loc_mask)
            if pos_loc_pred.numel():
                pos_decode_loc_target, pos_decode_loc_pred = self.decode_loc(pos_loc_target, pos_loc_pred.detach(), loc_mask)  # noqa
                temp = bbox_iou_overlaps(pos_decode_loc_pred, pos_decode_loc_target, aligned=True)
                scores[loc_mask] = temp.to(scores.dtype)
            cls_loss = self.cls_loss(cls_pred, cls_target, normalizer_override=pos_normalizer, scores=scores)
            acc = self._get_acc(cls_pred, cls_target)
        else:
            cls_loss = self.cls_loss(cls_pred, cls_target, normalizer_override=pos_normalizer)
            acc = self._get_acc(cls_pred, cls_target)

        # loc loss
        if self.loc_use_ghm:
            loc_loss = self.loc_loss(loc_pred, loc_target, loc_mask=loc_mask, mlvl_shapes=mlvl_shapes)
        else:
            weights_normalizer = self.get_weights_normalizer(iou_target)
            loc_target, loc_pred, iou_pred = self._mask_tensor([loc_target, loc_pred, iou_pred], loc_mask)
            weights_normalizer = self.get_weights_normalizer(iou_target)
            if loc_mask.sum() == 0:
                loc_loss = loc_pred.sum()
                iou_loss = iou_pred.sum()
            else:
                loc_loss_key_fields = getattr(self.loc_loss, "key_fields", set())
                loc_loss_kwargs = {}
                if "anchor" in loc_loss_key_fields:
                    loc_loss_kwargs["anchor"] = self.get_sample_anchor(loc_mask)
                if "weights" in loc_loss_key_fields:
                    loc_loss_kwargs["weights"] = iou_target
                loc_loss = self.loc_loss(loc_pred, loc_target, normalizer_override=weights_normalizer, **loc_loss_kwargs)  # noqa
                iou_loss = self.iou_branch_loss(iou_pred.view(-1), iou_target, normalizer_override=pos_normalizer)
        return {
            self.prefix + '.cls_loss': cls_loss,
            self.prefix + '.loc_loss': loc_loss,
            self.prefix + '.accuracy': acc,
            self.prefix + '.iou_loss': iou_loss
        }

    def get_weights_normalizer(self, centerness_target):
        sum_centerness_targets = centerness_target.sum()
        if env.world_size > 1:
            allreduce(sum_centerness_targets)
        num_gpus = env.world_size
        ave_centerness_targets = max(sum_centerness_targets.item(), 1) / float(num_gpus)
        return ave_centerness_targets

    def get_ave_normalizer(self, loc_mask):
        ave_loc_mask = torch.sum(loc_mask)
        allreduce(ave_loc_mask)
        num_gpus = env.world_size
        ave_normalizer = max(ave_loc_mask.item(), 1) / float(num_gpus)
        return ave_normalizer


@MODULE_PROCESS_REGISTRY.register('rpn_post')
class RPNPostProcess(BaseDetPostProcess):
    """
    hzh_22_1_5
    Classify Anchors (2 cls): foreground or background. This module is usually
    used with :class:`~pod.models.heads.bbox_head.bbox_head.BboxNet`
    """

    def __init__(self, num_classes, cfg, prefix=None):
        super(RPNPostProcess, self).__init__(num_classes, cfg, prefix)

    def get_loss(self, targets, mlvl_preds, mlvl_shapes=None):
        cls_target, loc_target, cls_mask, loc_mask = targets
        mlvl_cls_pred, mlvl_loc_pred = zip(*mlvl_preds)
        cls_pred = torch.cat(mlvl_cls_pred, dim=1)
        loc_pred = torch.cat(mlvl_loc_pred, dim=1)
        del mlvl_cls_pred, mlvl_loc_pred
        num_pos = max(1, torch.sum(cls_mask.float()).item())
        normalizer = num_pos if 'focal' in self.cls_loss.name else None
        ### here can print loc_mask ##
        cls_loss = self.cls_loss(cls_pred, cls_target, normalizer_override=normalizer)
        ### here can not print loc_mask ##
        loc_target, loc_pred = self._mask_tensor([loc_target, loc_pred], loc_mask)
        if loc_mask.sum() == 0:
            loc_loss = loc_pred.sum()
        else:
            loc_loss_key_fields = getattr(self.loc_loss, "key_fields", set())
            loc_loss_kwargs = {}
            if "anchor" in loc_loss_key_fields:
                loc_loss_kwargs["anchor"] = self.get_sample_anchor(loc_mask)
            loc_loss = self.loc_loss(loc_pred, loc_target, normalizer_override=num_pos, **loc_loss_kwargs)

        acc = self._get_acc(cls_pred, cls_target)

        return {
            self.prefix + '.cls_loss': cls_loss,
            self.prefix + '.loc_loss': loc_loss,
            self.prefix + '.accuracy': acc
        }
