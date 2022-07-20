# modified from https://github.com/ModelTC/United-Perception
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat

from up.utils.model.normalize import build_conv_norm
from up.utils.model.initializer import init_bias_focal, initialize_from_cfg, init_weights_normal
from up.models.losses import build_loss
from up.tasks.det.models.losses.entropy_loss import apply_class_activation
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.utils.general.global_flag import FP16_FLAG
from up.utils.env.dist_helper import allreduce, env
from up.tasks.det.models.utils.anchor_generator import build_anchor_generator
from up.utils.model import accuracy as A
from up.tasks.det.models.utils.bbox_helper import offset2bbox
from up.tasks.det.models.utils.transformer import TransformerBlock

from up.tasks.det.models.postprocess.roi_supervisor import build_roi_supervisor
from up.tasks.det.models.postprocess.roi_predictor import build_roi_predictor


class BaseRoINet(nn.Module):
    def __init__(self, inplanes, num_classes, cfg):
        super(BaseRoINet, self).__init__()
        self.prefix = self.__class__.__name__
        self.tocaffe = False
        self.toskme = False

        self.num_classes = num_classes
        assert self.num_classes > 1

        self.anchor_generator = build_anchor_generator(cfg['anchor_generator'])
        self.supervisor = build_roi_supervisor(cfg['roi_supervisor'])
        if 'train' in cfg:
            train_cfg = copy.deepcopy(cfg)
            train_cfg.update(train_cfg['train'])
            self.train_predictor = build_roi_predictor(train_cfg['roi_predictor'])
        else:
            self.train_predictor = None

        test_cfg = copy.deepcopy(cfg)
        test_cfg.update(test_cfg.get('test', {}))
        self.test_predictor_type = test_cfg['roi_predictor']['type']
        self.test_predictor = build_roi_predictor(test_cfg['roi_predictor'])

        if isinstance(inplanes, list):
            inplanes_length = len(inplanes)
            for i in range(1, inplanes_length):
                if inplanes[i] != inplanes[0]:
                    raise ValueError('list inplanes elements are inconsistent with {}'.format(inplanes[i]))
            inplanes = inplanes[0]
        assert isinstance(inplanes, int)
        self.inplanes = inplanes

        self.cls_loss = build_loss(cfg['cls_loss'])
        self.loc_loss = build_loss(cfg['loc_loss'])

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
        mlvl_activated_preds = []
        for lvl_idx, preds in enumerate(mlvl_preds):
            cls_pred = apply_class_activation(preds[0], self.class_activation)
            if self.with_background_channel and remove_background_channel_if_any:
                cls_pred = cls_pred[..., 1:]
            mlvl_activated_preds.append((cls_pred, *preds[1:]))
        return mlvl_activated_preds

    def get_loss(self, targets, mlvl_preds, mlvl_shapes):
        raise NotImplementedError

    def forward_net(self, x):
        raise NotImplementedError

    def permute_preds(self, mlvl_preds):
        mlvl_permuted_preds, mlvl_shapes = [], []
        for lvl_idx, preds in enumerate(mlvl_preds):
            b, _, h, w = preds[0].shape
            k = self.num_anchors * h * w
            preds = [p.permute(0, 2, 3, 1).contiguous().view(b, k, -1) for p in preds]
            mlvl_permuted_preds.append(preds)
            mlvl_shapes.append((h, w, k))
        return mlvl_permuted_preds, mlvl_shapes

    def forward(self, input):
        features = input['features']
        strides = input['strides']
        image_info = input['image_info']
        device = features[0].device

        mlvl_raw_preds = [self.forward_net(x, lvl) for lvl, x in enumerate(features)]

        mlvl_preds, mlvl_shapes = self.permute_preds(mlvl_raw_preds)
        mlvl_shapes = [(*shp, s) for shp, s in zip(mlvl_shapes, strides)]

        mlvl_anchors = self.anchor_generator.get_anchors(mlvl_shapes, device=device)
        self.mlvl_anchors = mlvl_anchors
        output = {}

        if self.training:
            targets = self.supervisor.get_targets(mlvl_anchors, input, mlvl_preds)
            losses = self.get_loss(targets, mlvl_preds, mlvl_shapes)
            output.update(losses)
        else:
            mlvl_preds = self.apply_activation(mlvl_preds, remove_background_channel_if_any=True)
            results = self.test_predictor.predict(mlvl_anchors, mlvl_preds, image_info)
            output.update(results)

        return output

    def apply_score_op(self, mlvl_pred):
        cls_pred = mlvl_pred[0]
        if self.class_activation == 'sigmoid':
            return cls_pred.sigmoid()
        else:
            assert self.class_activation == 'softmax'
            _, _, h, w = cls_pred.shape
            c = cls_pred.shape[1] // self.num_anchors
            cls_pred = cls_pred.view(-1, c, h, w)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_pred = F.softmax(cls_pred, dim=-1)
            cls_pred = cls_pred.permute(0, 3, 1, 2).contiguous()
            cls_pred = cls_pred[:, 1:c, ...]
            cls_pred = cls_pred.view(-1, self.num_anchors * (c - 1), h, w)
            return cls_pred

    def _get_acc(self, cls_pred, cls_target):
        cls_pred = cls_pred.reshape(cls_target.numel(), -1)
        cls_target = cls_target.reshape(-1)
        is_focal = 'focal' in self.cls_loss.name
        if self.class_activation == 'softmax' and not is_focal:
            acc = A.accuracy(cls_pred, cls_target)[0]
        elif self.class_activation == 'softmax' and is_focal:
            acc = A.accuracy(cls_pred, cls_target, ignore_indices=[0, -1])[0]
        elif self.class_activation == 'sigmoid' and self.num_classes > 2:
            acc = A.accuracy(cls_pred, cls_target.long() - 1, ignore_indices=[-1, -2])[0]
        elif self.class_activation == 'sigmoid' and self.num_classes <= 2:
            acc = A.binary_accuracy(cls_pred, cls_target)[0]
        else:
            raise NotImplementedError
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
            masked_tensor = tensor.reshape(-1, n_dim)[mask]
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


@MODULE_ZOO_REGISTRY.register('UniAtssHead')
class UniAtssHead(BaseRoINet):
    def __init__(self, inplanes, feat_planes, num_classes, cfg, normalize=None, initializer=None,
                 num_point=16, dropout=0.1, cls_block_num=1, loc_block_num=1,
                 offset_bias_val=3.0, checkpoint=True):
        super(UniAtssHead, self).__init__(inplanes, num_classes, cfg)
        init_prior = self.cls_loss.init_prior
        self.num_point = num_point
        self.centerness_loss = build_loss(cfg['centerness_loss'])
        self.cfg = cfg
        self.checkpoint = checkpoint

        self.offset_pred = self._build_offset_layers(inplanes, feat_planes, normalize)

        class_channel = {'sigmoid': -1, 'softmax': 0}[self.class_activation] + self.num_classes
        self.class_channel = class_channel

        if cls_block_num > 1:
            self.cls_transformer = nn.Sequential(
                *[TransformerBlock(
                    inplanes, num_heads=8, activation='gelu', dropout=dropout, norm_style='pre_norm')
                  for _ in range(cls_block_num)])
        else:
            self.cls_transformer = TransformerBlock(
                inplanes, num_heads=8, activation='gelu', dropout=dropout, norm_style='pre_norm')

        self.cls_subnet_pred = nn.Sequential(
            nn.Linear(inplanes * (self.num_point + 1), feat_planes),
            nn.ReLU(inplace=True),
            nn.Linear(feat_planes, class_channel))

        if loc_block_num > 1:
            self.loc_transformer = nn.Sequential(
                *[TransformerBlock(
                  inplanes, num_heads=8, activation='gelu', dropout=dropout, norm_style='pre_norm')
                  for _ in range(loc_block_num)])
        else:
            self.loc_transformer = TransformerBlock(
                  inplanes, num_heads=8, activation='gelu', dropout=dropout, norm_style='pre_norm')

        self.loc_subnet_pred = nn.Sequential(
            nn.Linear(inplanes * self.num_point, feat_planes),
            nn.ReLU(inplace=True),
            nn.Linear(feat_planes, self.num_point * 2))
        self.centerness_pred = nn.Sequential(
            nn.Linear(inplanes, feat_planes),
            nn.ReLU(inplace=True),
            nn.Linear(feat_planes, 1))

        initialize_from_cfg(self, initializer)
        init_weights_normal(self.cls_subnet_pred, 0.01)
        init_weights_normal(self.centerness_pred, 0.01)
        init_weights_normal(self.offset_pred, 0.001)
        init_bias_focal(self.cls_subnet_pred[-1], self.class_activation, init_prior, self.num_classes)

        with torch.no_grad():
            loc_bias_val = 1 - 0.2 * offset_bias_val
            offset_bias_tensor = self._init_offset_bias_tensor(offset_bias_val).to(self.offset_pred[-1].bias)
            self.offset_pred[-1].bias.copy_(offset_bias_tensor)
            loc_bias_tensor = self._init_loc_bias_tensor(loc_bias_val).to(self.loc_subnet_pred[-1].bias)
            self.loc_subnet_pred[-1].bias.copy_(loc_bias_tensor)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.inplanes))
        self.centerness_token = nn.Parameter(torch.randn(1, 1, self.inplanes))

    def _init_loc_bias_tensor(self, bias_val=1.0):
        assert self.num_point % 4 == 0, 'num_point must be divided by 4'
        bias_tensor = torch.tensor([[-bias_val, 0], [0, -bias_val], [bias_val, 0], [0, bias_val]]).reshape(4, 1, 2)
        bias_tensor = bias_tensor.repeat(1, self.num_point // 4, 1).reshape(self.num_point, 2)
        return bias_tensor.reshape(-1)

    def _init_offset_bias_tensor(self, bias_val=1.0):
        assert self.num_point % 4 == 0, 'num_point must be divided by 4'
        bias_tensor = torch.tensor([[-bias_val, 0], [0, -bias_val], [bias_val, 0], [0, bias_val]]).reshape(4, 1, 2)
        bias_tensor = bias_tensor.repeat(1, self.num_point // 4, 1).reshape(self.num_point, 2)
        bias_tensor = bias_tensor.repeat(self.num_anchors, 1, 1)
        return bias_tensor.reshape(-1)

    def _build_offset_layers(self, inplanes, feat_planes, normalize):
        offset_layers = []
        module = build_conv_norm(inplanes, feat_planes,
                                 kernel_size=3, stride=1, padding=1,
                                 normalize=normalize, activation=True)
        for child in module.children():
            offset_layers.append(child)
        offset_layers.append(nn.Conv2d(feat_planes, self.num_anchors * self.num_point * 2,
                                       kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*offset_layers)

    def bilinear_gather(self, feat_map, index):
        '''
            input:
                index: FloatTensor, N*2 (w coordinate, h coordinate)
                feat_map: C*H*W
            return:
                bilinear inperpolated features: C*N
        '''
        assert feat_map.ndim == 3
        height, width = feat_map.shape[1:]
        w, h = index[..., 0], index[..., 1]

        h_low = torch.floor(h)
        h_low = torch.clamp(h_low, min=0, max=height - 1)
        h_high = torch.where(h_low >= height - 1, h_low, h_low + 1)
        h = torch.where(h_low >= height - 1, h_low, h)

        w_low = torch.floor(w)
        w_low = torch.clamp(w_low, min=0, max=width - 1)
        w_high = torch.where(w_low >= width - 1, w_low, w_low + 1)
        w = torch.where(w_low >= width - 1, w_low, w)

        h_low = h_low.long()
        w_low = w_low.long()
        h_high = h_high.long()
        w_high = w_high.long()

        h_low = h_low.detach()
        w_low = w_low.detach()
        h_high = h_high.detach()
        w_high = w_high.detach()

        lh = h - h_low  # N
        lw = w - w_low
        hh = 1 - lh
        hw = 1 - lw

        v1 = feat_map[:, h_low, w_low]  # C * N
        v2 = feat_map[:, h_low, w_high]
        v3 = feat_map[:, h_high, w_low]
        v4 = feat_map[:, h_high, w_high]

        w1 = hh * hw  # N
        w2 = hh * lw
        w3 = lh * hw
        w4 = lh * lw
        w1, w2, w3, w4 = [x.unsqueeze(0) for x in [w1, w2, w3, w4]]

        val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4  # C*N
        return val

    def apply_score_op(self, mlvl_pred):
        cls_pred = mlvl_pred[0]
        centerness_pred = mlvl_pred[2].sigmoid()
        _, _, h, w = cls_pred.shape
        c = cls_pred.shape[1] // self.num_anchors
        centerness_pred = centerness_pred.view(-1, self.num_anchors, 1, h, w).expand((-1, self.num_anchors, c, h, w))
        centerness_pred = centerness_pred.view(-1, cls_pred.shape[1], h, w)
        return centerness_pred * cls_pred

    def apply_activation(self, mlvl_preds, remove_background_channel_if_any):
        mlvl_activated_preds = []
        for lvl_idx, preds in enumerate(mlvl_preds):
            cls_pred = apply_class_activation(preds[0], self.class_activation)
            centerness_pred = apply_class_activation(preds[2], 'sigmoid')
            if self.with_background_channel and remove_background_channel_if_any:
                cls_pred = cls_pred[..., 1:]
            mlvl_activated_preds.append((cls_pred * centerness_pred, preds[1]))
        return mlvl_activated_preds

    def get_deform_feats(self, feat_map, anchors, offset, stride, num_point):
        B = len(feat_map)
        result_deform_feats = []
        centers_x = (anchors[:, 0] + anchors[:, 2]) / 2
        centers_y = (anchors[:, 1] + anchors[:, 3]) / 2
        w_ = anchors[:, 2] - anchors[:, 0] + 1
        h_ = anchors[:, 3] - anchors[:, 1] + 1
        centers = torch.stack((centers_x, centers_y), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
        wh_ = torch.stack((w_, h_), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
        for b_ix in range(B):
            batch_offset = offset[b_ix]
            batch_feat_map = feat_map[b_ix, ...]
            batch_index = centers + batch_offset * wh_ * 0.1
            batch_index = batch_index.reshape(-1, 2)
            batch_deform_feat = self.bilinear_gather(batch_feat_map, batch_index / stride)

            batch_deform_feat = rearrange(
                batch_deform_feat, 'c (nb np) -> nb np c', nb=len(anchors))
            result_deform_feats.append(batch_deform_feat)

        return torch.stack(result_deform_feats, dim=0)

    def forward(self, input):
        features = input['features']
        strides = input['strides']
        image_info = input['image_info']
        device = features[0].device

        mlvl_shapes = self.get_mlvl_shapes(features)
        mlvl_shapes = [(*shp, s) for shp, s in zip(mlvl_shapes, strides)]
        mlvl_anchors = self.anchor_generator.get_anchors(mlvl_shapes, device=device)
        self.mlvl_anchors = mlvl_anchors

        mlvl_preds = [self.forward_net(x, self.mlvl_anchors[lvl], strides[lvl])
                      for lvl, x in enumerate(features)]

        output = {}
        if self.training:
            targets = self.supervisor.get_targets(mlvl_anchors, input, mlvl_preds)
            losses = self.get_loss(targets, mlvl_preds, mlvl_shapes)
            output.update(losses)
        else:
            mlvl_preds = self.apply_activation(mlvl_preds, remove_background_channel_if_any=True)
            results = self.test_predictor.predict(mlvl_anchors, mlvl_preds, image_info)
            output.update(results)

        return output

    def forward_net(self, x, anchor, stride):
        # anchor: A
        offset_pred = self.offset_pred(x)  # B * (A*n_point*2) * H * W
        offset = rearrange(offset_pred, 'b (a n p) h w -> b (h w a) n p',
                           a=self.num_anchors, n=self.num_point, p=2)   # B * (H*W*A) * n_point * 2
        deform_feats = self.get_deform_feats(x, anchor, offset, stride, self.num_point)  # B * (H*W*A) * n_point * C
        B, K, _, C = deform_feats.shape
        deform_feats = deform_feats.reshape(-1, self.num_point, C)

        cls_tokens = repeat(
            self.cls_token, '() n d -> b n d', b=deform_feats.shape[0])
        cls_x = torch.cat((cls_tokens, deform_feats), dim=1)

        if self.checkpoint:
            cls_x = checkpoint(self.cls_transformer, cls_x)
        else:
            cls_x = self.cls_transformer(cls_x)

        cls_x = rearrange(cls_x, 'b n d -> b (n d)')
        cls_pred = self.cls_subnet_pred(cls_x)
        cls_pred = cls_pred.reshape(B, K, -1)

        centerness_tokens = repeat(
            self.centerness_token, '() n d -> b n d', b =deform_feats.shape[0])
        loc_x = torch.cat((centerness_tokens, deform_feats), dim=1)

        if self.checkpoint:
            loc_x = checkpoint(self.loc_transformer, loc_x)
        else:
            loc_x = self.loc_transformer(loc_x)

        centerness_x = loc_x[:, 0, :]

        loc_x = loc_x[:, 1:, :]
        loc_x = rearrange(loc_x, 'b n d -> b (n d)')

        loc_pred = self.loc_subnet_pred(loc_x)
        loc_pred = loc_pred.reshape(B, K, self.num_point, 2)

        centerness_pred = self.centerness_pred(centerness_x)
        centerness_pred = centerness_pred.reshape(B, K, 1)

        if FP16_FLAG.fp16:
            cls_pred = cls_pred.float()
            loc_pred = loc_pred.float()
            centerness_pred = centerness_pred.float()
            offset = offset.float()

        return cls_pred, (offset, loc_pred), centerness_pred

    def get_regression_boxes(self, rois, offset_pred, loc_pred):
        offset = offset_pred.reshape(rois.shape[0], -1, 2)
        centers_x = (rois[:, 0] + rois[:, 2]) / 2
        centers_y = (rois[:, 1] + rois[:, 3]) / 2
        w_, h_ = (rois[:, 2] - rois[:, 0] + 1), (rois[:, 3] - rois[:, 1] + 1)
        centers = torch.stack((centers_x, centers_y), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
        wh_ = torch.stack((w_, h_), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
        offset_point = centers + offset * wh_ * 0.1  # N_batch * num_point * 2
        loc_pred = loc_pred.reshape(rois.shape[0], self.num_point, 2)
        x_shift = loc_pred[..., 0] * w_.unsqueeze(1) * 0.5
        y_shift = loc_pred[..., 1] * h_.unsqueeze(1) * 0.5

        shifts = torch.stack((x_shift, y_shift), dim=-1)  # N_batch * num_point * 2
        shifted_offset_point = offset_point + shifts

        x_min, _ = torch.min(shifted_offset_point[..., 0], dim=-1)
        y_min, _ = torch.min(shifted_offset_point[..., 1], dim=-1)
        x_max, _ = torch.max(shifted_offset_point[..., 0], dim=-1)
        y_max, _ = torch.max(shifted_offset_point[..., 1], dim=-1)
        iou_boxes = torch.stack((x_min, y_min, x_max, y_max), dim=-1)

        return iou_boxes

    def get_loss(self, targets, mlvl_preds, mlvl_shapes=None):
        mlvl_shapes = [shape[:-1] for shape in mlvl_shapes]
        cls_target, loc_target, centerness_target, cls_mask, loc_mask = targets
        mlvl_cls_pred, mlvl_loc_pred, mlvl_centerness_pred = zip(*mlvl_preds)

        all_anchors = torch.cat(self.mlvl_anchors, dim=0)
        offsets = [x[0] for x in mlvl_loc_pred]
        loc_pred = [x[1] for x in mlvl_loc_pred]
        offsets = torch.cat(offsets, dim=1)
        loc_pred = torch.cat(loc_pred, dim=1)
        loc_pred = torch.stack(
            [self.get_regression_boxes(all_anchors, offsets[i], loc_pred[i]) for i in range(offsets.shape[0])],
            dim=0)

        cls_pred = torch.cat(mlvl_cls_pred, dim=1)
        centerness_pred = torch.cat(mlvl_centerness_pred, dim=1)

        del mlvl_cls_pred, mlvl_loc_pred, mlvl_centerness_pred

        normalizer = self.get_ave_normalizer(loc_mask)
        # cls loss
        cls_loss = self.cls_loss(cls_pred, cls_target, normalizer_override=normalizer)
        acc = self._get_acc(cls_pred, cls_target)

        # loc loss
        weights_normalizer = self.get_weights_normalizer(centerness_target)
        loc_target, loc_pred = self._mask_tensor([loc_target, loc_pred], loc_mask)

        if loc_mask.sum() == 0:
            loc_loss = loc_pred.sum()
        else:
            loc_loss = self.loc_loss(
                loc_pred, loc_target, normalizer_override=weights_normalizer) # noqa

        centerness_loss = self.centerness_loss(centerness_pred[loc_mask].view(-1), centerness_target, normalizer_override=normalizer) # noqa

        return {
            self.prefix + '.cls_loss': cls_loss,
            self.prefix + '.loc_loss': loc_loss,
            self.prefix + '.centerness_loss': centerness_loss,
            self.prefix + '.accuracy': acc
        }

    def get_mlvl_shapes(self, features):
        mlvl_shapes = []
        for lvl, x in enumerate(features):
            b, _, h, w = x.shape
            k = self.num_anchors * h * w
            mlvl_shapes.append((h, w, k))
        return mlvl_shapes

    def permute_preds(self, mlvl_preds):
        mlvl_permuted_preds, mlvl_shapes = [], []
        for lvl_idx, preds in enumerate(mlvl_preds):
            b, _, h, w = preds[0].shape
            k = self.num_anchors * h * w
            preds = [p.permute(0, 2, 3, 1).contiguous().view(b, k, -1) for p in preds]
            mlvl_permuted_preds.append(preds)
            mlvl_shapes.append((h, w, k))
        return mlvl_permuted_preds, mlvl_shapes

    def get_weights_normalizer(self, centerness_target):
        sum_centerness_targets = centerness_target.sum()
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
