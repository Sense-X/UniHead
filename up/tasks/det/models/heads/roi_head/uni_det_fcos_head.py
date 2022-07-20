# modified from https://github.com/ModelTC/United-Perception
import copy

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from up.utils.model.normalize import build_conv_norm
from up.utils.model import accuracy as A
from up.tasks.det.models.utils.anchor_generator import build_anchor_generator
from up.utils.model.initializer import init_bias_focal, initialize_from_cfg, init_weights_normal
from up.models.losses import build_loss
from up.tasks.det.models.losses.entropy_loss import apply_class_activation
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.utils.general.global_flag import FP16_FLAG
from einops import rearrange, repeat
from up.tasks.det.models.utils.transformer import TransformerBlock
from up.tasks.det.models.utils.bbox_helper import bbox_iou_overlaps

from up.tasks.det.models.postprocess.roi_supervisor import build_roi_supervisor
from up.tasks.det.models.postprocess.roi_predictor import build_roi_predictor


class UniFcosNet(nn.Module):
    def __init__(self, inplanes, num_classes, dense_points, loc_ranges, cfg):
        super(UniFcosNet, self).__init__()

        self.num_classes = num_classes
        self.dense_points = dense_points
        # self.dense_points = cfg.get('dense_points', 1)
        assert self.num_classes > 1
        # self.loc_ranges = cfg.get('loc_ranges', [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 100000]])
        self.loc_ranges = loc_ranges
        if isinstance(inplanes, list):
            assert len(inplanes) == 1, 'single input is expected, but found:{} '.format(inplanes)
            inplanes = inplanes[0]
        assert isinstance(inplanes, int)
        self.supervisor = build_roi_supervisor(cfg['roi_supervisor'])
        self.predictor = build_roi_predictor(cfg['roi_predictor'])
        self.center_generator = build_anchor_generator(cfg['center_generator'])

        self.cls_loss = build_loss(cfg['cls_loss'])
        self.loc_loss = build_loss(cfg['loc_loss'])
        self.center_loss = build_loss(cfg['center_loss'])
        self.prefix = 'UniFcosNet'
        self.cfg = copy.deepcopy(cfg)
        self.cls_loss_type = self.cls_loss.activation_type

    @property
    def class_activation(self):
        return self.cls_loss.activation_type

    @property
    def with_background_channel(self):
        return self.class_activation == 'softmax'

    def apply_activation_and_centerness(self, mlvl_preds, remove_background_channel_if_any):
        mlvl_activated_preds = []
        for lvl_idx, preds in enumerate(mlvl_preds):
            cls_pred = apply_class_activation(preds[0], self.class_activation)
            ctr_pred = preds[2].sigmoid()
            if self.with_background_channel and remove_background_channel_if_any:
                cls_pred = cls_pred[..., 1:]
            cls_pred *= ctr_pred
            mlvl_activated_preds.append((cls_pred, *preds[1:]))
        return mlvl_activated_preds

    def permute_preds(self, mlvl_preds):
        mlvl_permuted_preds, mlvl_shapes = [], []
        for lvl_idx, preds in enumerate(mlvl_preds):
            b, _, h, w = preds[0].shape
            k = self.dense_points * h * w
            preds = [p.permute(0, 2, 3, 1).contiguous().view(b, k, -1) for p in preds]
            mlvl_permuted_preds.append(preds)
            mlvl_shapes.append((h, w, k))
        return mlvl_permuted_preds, mlvl_shapes

    def forward_net(self, x, location, stride):
        raise NotImplementedError
    
    def get_mlvl_shapes(self, features):
        mlvl_shapes = []
        for lvl, x in enumerate(features):
            b, _, h, w = x.shape
            k = self.dense_points * h * w
            mlvl_shapes.append((h, w, k))
        return mlvl_shapes

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

    def get_deform_feats(self, feat_map, locations, offset, stride, num_point):
        B = len(feat_map)
        result_deform_feats = []
        centers_x = locations[:, 0]
        centers_y = locations[:, 1]
        scale_ = stride * self.base_scale
        centers = torch.stack((centers_x, centers_y), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
        for b_ix in range(B):
            batch_offset = offset[b_ix]
            batch_feat_map = feat_map[b_ix, ...]
            batch_index = centers + batch_offset * scale_ * 0.1
            batch_index = batch_index.reshape(-1, 2)
            batch_deform_feat = self.bilinear_gather(batch_feat_map, batch_index / stride)

            batch_deform_feat = rearrange(
                batch_deform_feat, 'c (nb np) -> nb np c', nb=len(locations))
            result_deform_feats.append(batch_deform_feat)

        return torch.stack(result_deform_feats, dim=0)

    def get_regression_boxes(self, locations, offset_pred, loc_pred):
        offset = offset_pred.reshape(locations.shape[0], -1, 2)
        centers_x = locations[:, 0]
        centers_y = locations[:, 1]
        scale_ = locations[:, 2] * self.base_scale

        centers = torch.stack((centers_x, centers_y), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
        offset_point = centers + offset * scale_.reshape(-1, 1, 1) * 0.1  # N_batch * num_point * 2
        loc_pred = loc_pred.reshape(locations.shape[0], self.num_point, 2)
        x_shift = loc_pred[..., 0] * scale_.unsqueeze(1) * 0.5
        y_shift = loc_pred[..., 1] * scale_.unsqueeze(1) * 0.5

        shifts = torch.stack((x_shift, y_shift), dim=-1)  # N_batch * num_point * 2
        shifted_offset_point = offset_point + shifts

        x_min, _ = torch.min(shifted_offset_point[..., 0], dim=-1)
        y_min, _ = torch.min(shifted_offset_point[..., 1], dim=-1)
        x_max, _ = torch.max(shifted_offset_point[..., 0], dim=-1)
        y_max, _ = torch.max(shifted_offset_point[..., 1], dim=-1)
        iou_boxes = torch.stack((x_min, y_min, x_max, y_max), dim=-1)

        return iou_boxes

    def forward(self, input):
        features = input['features']
        strides = input['strides']
        image_info = input['image_info']
        
        mlvl_shapes = self.get_mlvl_shapes(features)
        mlvl_shapes = [(*shp, s) for shp, s in zip(mlvl_shapes, strides)]
        mlvl_locations = self.center_generator.get_anchors(mlvl_shapes, device=features[0].device)

        location_with_stride = []
        for location_, stride in zip(mlvl_locations, strides):
            tmp_ = location_.new_full((location_.shape[0], ), fill_value=stride)
            location_with_stride.append(torch.cat((location_, tmp_.reshape(-1, 1)), dim=-1))
        self.mlvl_locations = location_with_stride

        mlvl_preds = [self.forward_net(x, self.mlvl_locations[lvl], strides[lvl])
                      for lvl, x in enumerate(features)]

        if self.training:
            targets = self.supervisor.get_targets(mlvl_locations, self.loc_ranges, input)
            losses = self.get_loss(targets, mlvl_preds)
            return losses
        else:
            with torch.no_grad():
                mlvl_preds = self.apply_activation_and_centerness(mlvl_preds, remove_background_channel_if_any=True)
                results = self.predictor.predict(self.mlvl_locations, mlvl_preds, image_info)
                return results

    def get_loss(self, targets, mlvl_preds):
        mlvl_cls_pred, mlvl_loc_pred, mlvl_ctr_pred = zip(*mlvl_preds)
        cls_pred = torch.cat(mlvl_cls_pred, dim=1)
        centerness_pred = torch.cat(mlvl_ctr_pred, dim=1)

        cls_target, loc_target, cls_mask, loc_mask = targets

        normalizer = max(1, torch.sum(loc_mask).item())
        cls_loss = self.cls_loss(cls_pred, cls_target, normalizer_override=normalizer)

        all_locations = torch.cat(self.mlvl_locations, dim=0)
        offsets = [x[0] for x in mlvl_loc_pred]
        loc_pred = [x[1] for x in mlvl_loc_pred]
        offsets = torch.cat(offsets, dim=1)
        loc_pred = torch.cat(loc_pred, dim=1)
        loc_pred = torch.stack(
            [self.get_regression_boxes(all_locations, offsets[i], loc_pred[i]) for i in range(offsets.shape[0])],
            dim=0)
        del mlvl_cls_pred, mlvl_loc_pred, mlvl_ctr_pred

        acc = A.accuracy_v2(cls_pred, cls_target, activation_type=self.cls_loss_type)

        sample_loc_mask = loc_mask.reshape(-1)
        loc_target = loc_target.reshape(-1, 4)[sample_loc_mask]
        loc_pred = loc_pred.reshape(-1, 4)[sample_loc_mask]
        centerness_pred = centerness_pred.reshape(-1)[sample_loc_mask]
        
        if loc_pred.numel() > 0:
            centerness_targets = bbox_iou_overlaps(loc_pred.detach(), loc_target, aligned=True)
            loc_loss = self.loc_loss(
                loc_pred, loc_target,
                weights=centerness_targets.clamp(1e-12),
                normalizer_override=centerness_targets.sum())
            centerness_loss = self.center_loss(centerness_pred, centerness_targets)
        else:
            loc_loss = loc_pred.sum()
            centerness_loss = centerness_pred.sum()
        return {
            self.prefix + '.cls_loss': cls_loss,
            self.prefix + '.loc_loss': loc_loss,
            self.prefix + '.centerness_loss': centerness_loss,
            self.prefix + '.accuracy': acc
        }


@MODULE_ZOO_REGISTRY.register('UniFcosHead')
class UniFcosHead(UniFcosNet):
    def __init__(self, inplanes, feat_planes, num_classes, dense_points, loc_ranges,
                 cfg, normalize=None, initializer=None,
                 num_point=16, dropout=0.1, cls_block_num=1, loc_block_num=1,
                 offset_bias_val=3.0, base_scale=8, checkpoint=True):
        super().__init__(inplanes, num_classes, dense_points, loc_ranges, cfg)
        init_prior = self.cls_loss.init_prior
        self.num_point = num_point
        self.cfg = cfg
        self.base_scale = base_scale
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

        self.cls_token = nn.Parameter(torch.randn(1, 1, inplanes))
        self.centerness_token = nn.Parameter(torch.randn(1, 1, inplanes))

    def _init_loc_bias_tensor(self, bias_val=1.0):
        assert self.num_point % 4 == 0, 'num_point must be divided by 4'
        bias_tensor = torch.tensor([[-bias_val, 0], [0, -bias_val], [bias_val, 0], [0, bias_val]]).reshape(4, 1, 2)
        bias_tensor = bias_tensor.repeat(1, self.num_point // 4, 1).reshape(self.num_point, 2)
        return bias_tensor.reshape(-1)

    def _init_offset_bias_tensor(self, bias_val=1.0):
        assert self.num_point % 4 == 0, 'num_point must be divided by 4'
        bias_tensor = torch.tensor([[-bias_val, 0], [0, -bias_val], [bias_val, 0], [0, bias_val]]).reshape(4, 1, 2)
        bias_tensor = bias_tensor.repeat(1, self.num_point // 4, 1).reshape(self.num_point, 2)
        bias_tensor = bias_tensor.repeat(self.dense_points, 1, 1)
        return bias_tensor.reshape(-1)

    def _build_offset_layers(self, inplanes, feat_planes, normalize):
        offset_layers = []
        module = build_conv_norm(inplanes, feat_planes,
                                 kernel_size=3, stride=1, padding=1,
                                 normalize=normalize, activation=True)
        for child in module.children():
            offset_layers.append(child)
        offset_layers.append(nn.Conv2d(feat_planes, self.dense_points * self.num_point * 2,
                                       kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*offset_layers)

    def forward_net(self, x, centers, stride):
        # dense point num: A
        offset_pred = self.offset_pred(x)  # B * (A*n_point*2) * H * W
        offset = rearrange(offset_pred, 'b (a n p) h w -> b (h w a) n p',
                           a=self.dense_points, n=self.num_point, p=2)   # B * (H*W*A) * n_point * 2
        deform_feats = self.get_deform_feats(x, centers, offset, stride, self.num_point)  # B * (H*W*A) * n_point * C
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
            self.centerness_token, '() n d -> b n d', b=deform_feats.shape[0])
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
