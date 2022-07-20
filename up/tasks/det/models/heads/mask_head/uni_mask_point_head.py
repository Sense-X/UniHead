# modified from https://github.com/ModelTC/United-Perception
import copy
import math

import torch
import torch.nn as nn

from up import extensions as E
from up.tasks.det.models.utils.assigner import map_rois_to_level
from up.utils.model.initializer import init_weights_normal, initialize_from_cfg
from up.models.losses import build_loss
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from einops import rearrange
from up.tasks.det.models.utils.transformer import TransformerBlock
from up.tasks.det.models.postprocess import build_mask_predictor, build_mask_supervisor


class MaskPointNet(nn.Module):
    def __init__(self, num_classes, cfg):
        super(MaskPointNet, self).__init__()

        self.cfg = copy.deepcopy(cfg)
        self.cfg['num_classes'] = num_classes
        self.share_location = self.cfg.get('share_location', False)
        self.num_classes = num_classes

        self.roipool = E.build_generic_roipool(self.cfg['roipooling'])
        self.pool_size = cfg['roipooling']['pool_size']
        self.mask_loss = build_loss(self.cfg['mask_loss'])

        self.mask_supervisor = build_mask_supervisor(cfg['mask_supervisor'])
        self.mask_predictor = build_mask_predictor(cfg['mask_predictor'])
        self.prefix = 'MaskNet'
        self.counter = 0

    def forward_net(self, x):
        raise NotImplementedError

    def forward(self, input):
        if self.training:
            return self.get_loss(input)
        else:
            return self.get_mask(input)

    def mlvl_predict(self, x_rois, x_features, x_strides, levels, input=None):
        mlvl_deform_feats = []
        ini_offsets = []
        mlvl_rois = []
        mlvl_strides = []

        for lvl_idx in levels:
            if x_rois[lvl_idx].numel() > 0:
                lvl_rois = x_rois[lvl_idx]  # K*5
                lvl_feature = x_features[lvl_idx]  # N*C*H*W
                lvl_stride = x_strides[lvl_idx]
                pooled_feat = self.single_level_roi_extractor(lvl_rois, lvl_feature, lvl_stride)  # K*C*H*W
                ini_offset = self.get_offsets(pooled_feat)  # K*(N_point*2)
                deform_feats = self.get_deform_feats(lvl_feature, lvl_rois, ini_offset, lvl_stride, self.num_point)
                mlvl_strides.append(ini_offset.new_full((ini_offset.shape[0], ), fill_value=lvl_stride))
                mlvl_rois.append(lvl_rois)
                mlvl_deform_feats.append(deform_feats)
                ini_offsets.append(ini_offset)
        assert len(mlvl_rois) > 0, "No rois provided for second stage"
        deform_feats = torch.cat(mlvl_deform_feats, dim=0)
        ini_offsets = torch.cat(ini_offsets, dim=0)
        mask_offsets = self.forward_net(deform_feats)
        mask_strides = torch.cat(mlvl_strides, dim=0)
        return ini_offsets, mask_offsets, mask_strides

    def get_logits(self, rois, features, strides, input=None):
        assert self.cfg.get('fpn', None) is not None
        fpn = self.cfg['fpn']
        mlvl_rois, recover_inds = map_rois_to_level(fpn['fpn_levels'], fpn['base_scale'], rois)
        ini_offset_pred, mask_offset_pred, mask_strides = self.mlvl_predict(mlvl_rois, features, strides, fpn['fpn_levels'], input)
        rois = torch.cat(mlvl_rois, dim=0)
        return rois, ini_offset_pred.float(), mask_offset_pred.float(), mask_strides, recover_inds

    def get_pred_contour(self, rois, ini_offset_pred, mask_offset_pred):
        ini_offset = ini_offset_pred.reshape(rois.shape[0], -1, 2)
        mask_offset = mask_offset_pred.reshape(rois.shape[0], -1, 2)
        centers_x = (rois[:, 0] + rois[:, 2]) / 2
        centers_y = (rois[:, 1] + rois[:, 3]) / 2
        w_, h_ = (rois[:, 2] - rois[:, 0] + 1), (rois[:, 3] - rois[:, 1] + 1)
        centers = torch.stack((centers_x, centers_y), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
        wh_ = torch.stack((w_, h_), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
        offset_points = centers + ini_offset * wh_ * 0.1
        contours = centers + ini_offset * wh_ * 0.1 + mask_offset * wh_ * 0.5  # N_batch * num_point * 2
        return contours, offset_points

    def get_loss(self, input):
        features = input['features']
        strides = input['strides']

        sampled_rois, contour_target, gt_target = self.mask_supervisor.get_targets(input)

        rois, ini_offset, mask_offset, mask_strides, recover_inds = self.get_logits(sampled_rois, features, strides, input)
        ini_offset = ini_offset[recover_inds]
        mask_offset = mask_offset[recover_inds]
        rois = rois[recover_inds]
        mask_strides = mask_strides[recover_inds]
        normalize_term = mask_strides.reshape(-1, 1, 1) * 8

        contour_target = contour_target.reshape(rois.shape[0], self.num_point, 2)
        contour_pred, _ = self.get_pred_contour(rois[:, 1:5], ini_offset, mask_offset)
        if self.use_cross_iou_loss:
            qvec_pred, _ = self.get_qvec(contour_pred, rois[:, 1:5])
            qvec_target, pos_inds = self.get_qvec(contour_target, rois[:, 1:5])
            mask_loss = self.cross_iou_loss(
                qvec_pred, qvec_target, contour_pred, gt_target, pos_inds)
            mask_loss = self.cross_iou_loss_weight * mask_loss
        else:
            mask_loss = self.mask_loss(contour_pred/normalize_term, contour_target/normalize_term)

        return {self.prefix + '.mask_loss': mask_loss}

    def polygons2bbox(self, polygons):
        x_min = polygons[..., 0].min(dim=-1)[0]
        y_min = polygons[..., 1].min(dim=-1)[0]
        x_max = polygons[..., 0].max(dim=-1)[0]
        y_max = polygons[..., 1].max(dim=-1)[0]
        return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    def get_qvec(self, pts, rois):
        pts = pts.reshape(pts.shape[0], -1)
        gt_reg = pts.new_zeros([pts.shape[0], pts.shape[1]*2])
        centers_x = (rois[:, 0] + rois[:, 2]) / 2
        centers_y = (rois[:, 1] + rois[:, 3]) / 2
        anchor_pts = torch.stack((centers_x, centers_y), dim=-1)

        anchor_pts_repeat = anchor_pts.repeat(1, self.num_point)
        offset_reg = pts - anchor_pts_repeat

        br_reg = offset_reg >= 0
        tl_reg = offset_reg < 0
        tlbr_inds = torch.stack([tl_reg, br_reg], -1).reshape(-1, pts.shape[1]*2)
        gt_reg[tlbr_inds] = torch.abs(offset_reg.reshape(-1))

        xl_reg = gt_reg[..., 0::4]
        xr_reg = gt_reg[..., 1::4]
        yt_reg = gt_reg[..., 2::4]
        yb_reg = gt_reg[..., 3::4]

        yx_gt_reg = torch.stack([yt_reg, yb_reg, xl_reg, xr_reg], -1).reshape(-1, pts.size(1)*2)

        xl_inds = tlbr_inds[..., 0::4]
        xr_inds = tlbr_inds[..., 1::4]
        yt_inds = tlbr_inds[..., 2::4]
        yb_inds = tlbr_inds[..., 3::4]
        yx_inds = torch.stack([yt_inds, yb_inds, xl_inds, xr_inds], -1).reshape(-1, pts.size(1)*2)

        return yx_gt_reg, yx_inds

    def cross_iou_loss(self, pred, target, poly_pred, bbox_gt, pos_inds, alpha=0.2, eps=1e-6):
        neg_inds = ~pos_inds
        target[neg_inds] = alpha * target[pos_inds]

        total = torch.stack([pred, target], -1)
        total_reshape = total.reshape(total.shape[0], -1, 4, total.shape[-1])

        l_max = total_reshape.max(dim=3)[0]
        l_min = total_reshape.min(dim=3)[0]
        overlaps = l_min.sum(dim=2) / l_max.sum(dim=2)
        overlaps = overlaps.mean(dim=-1)

        bbox_pred = self.polygons2bbox(poly_pred)

        enclose_x1y1 = torch.min(bbox_pred[:, :2], bbox_gt[:, :2])
        enclose_x2y2 = torch.max(bbox_pred[:, 2:], bbox_gt[:, 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

        cw = enclose_wh[:, 0]
        ch = enclose_wh[:, 1]

        c2 = cw ** 2 + ch ** 2 + eps

        b1_x1, b1_y1 = bbox_pred[:, 0], bbox_pred[:, 1]
        b1_x2, b1_y2 = bbox_pred[:, 2], bbox_pred[:, 3]
        b2_x1, b2_y1 = bbox_gt[:, 0], bbox_gt[:, 1]
        b2_x2, b2_y2 = bbox_gt[:, 2], bbox_gt[:, 3]

        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4
        right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
        rho2 = left + right

        factor = 4 / math.pi ** 2
        v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        loss = 1 - (overlaps - (rho2 / c2 + v ** 2 / (1 - overlaps + v)))

        return loss.mean()

    def get_mask(self, input):
        features = input['features']
        strides = input['strides']
        rois = input['dt_bboxes']

        bboxes, ini_offset, mask_offset, _, recover_inds = self.get_logits(rois, features, strides)
        contour_pred, offset_points = self.get_pred_contour(bboxes[:, 1:5], ini_offset, mask_offset)

        results = self.mask_predictor.predict(contour_pred, bboxes, input)
        results['dt_bboxes'] = bboxes

        return results

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

        if self.bilinear_detach:
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

    def get_deform_feats(self, feat_map, rois, offset, stride, num_point):
        B = len(feat_map)
        deform_feats = torch.zeros(rois.shape[0], num_point, feat_map.shape[1]).to(feat_map)

        for b_ix in range(B):
            batch_roi_idx = (rois[:, 0] == b_ix)
            batch_scatter_index = torch.nonzero(batch_roi_idx).reshape(-1)
            if len(batch_scatter_index) == 0:
                continue

            batch_rois = rois[batch_roi_idx]  # N_batch * 5
            batch_offset = offset[batch_roi_idx]  # N_batch * (num_points * 2)
            batch_offset = batch_offset.reshape(len(batch_scatter_index), -1, 2)  # N_batch * num_points * 2

            batch_feat_map = feat_map[b_ix, ...]
            centers_x = (batch_rois[:, 1] + batch_rois[:, 3]) / 2
            centers_y = (batch_rois[:, 2] + batch_rois[:, 4]) / 2
            w_ = batch_rois[:, 3] - batch_rois[:, 1] + 1
            h_ = batch_rois[:, 4] - batch_rois[:, 2] + 1
            centers = torch.stack((centers_x, centers_y), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
            wh_ = torch.stack((w_, h_), dim=-1).unsqueeze(1)  # N_batch * 1 * 2

            batch_index = centers + batch_offset * wh_ * 0.1
            batch_index = batch_index.reshape(-1, 2)
            batch_deform_feat = self.bilinear_gather(batch_feat_map, batch_index / stride)

            batch_deform_feat = rearrange(
                batch_deform_feat, 'c (nb np) -> nb np c', nb=len(batch_scatter_index))
            deform_feats[batch_scatter_index] = batch_deform_feat
            del batch_roi_idx

        return deform_feats


@MODULE_ZOO_REGISTRY.register('UniMaskHead')
class MaskPointHead(MaskPointNet):
    def __init__(self,
                 inplanes,
                 num_classes,
                 feat_planes,
                 cfg,
                 num_point=32,
                 initializer=None,
                 dropout=0.1,
                 mask_block_num=1,
                 offset_bias_val=3.0,
                 use_cross_iou_loss=False,
                 cross_iou_loss_weight=1.0):
        super(MaskPointHead, self).__init__(num_classes, cfg)
        self.num_point = num_point
        self.inplanes = inplanes
        self.bilinear_detach = True
        self.use_cross_iou = False
        self.use_cross_iou_loss = use_cross_iou_loss
        self.cross_iou_loss_weight = cross_iou_loss_weight

        if isinstance(inplanes, list):
            assert len(inplanes) == 1, 'single input is expected, but found:{} '.format(inplanes)
            inplanes = inplanes[0]
        assert isinstance(inplanes, int)

        planes = feat_planes
        self.fc_offset = nn.Sequential(
            nn.Linear(self.pool_size * self.pool_size * inplanes, inplanes),
            nn.ReLU(inplace=True),
            nn.Linear(inplanes, self.num_point*2))

        if mask_block_num > 1:
            self.mask_transformer = nn.Sequential(
                *[TransformerBlock(
                    self.inplanes, num_heads=8, activation='gelu', dropout=dropout, norm_style='pre_norm')
                  for _ in range(mask_block_num)])
        else:
            self.mask_transformer = TransformerBlock(
                self.inplanes, num_heads=8, activation='gelu', dropout=dropout, norm_style='pre_norm')

        self.fc_mask_offset = nn.Sequential(
            nn.Linear(self.num_point * inplanes, feat_planes),
            nn.ReLU(inplace=True),
            nn.Linear(feat_planes, self.num_point*2))
        
        initialize_from_cfg(self, initializer)

        init_weights_normal(self.fc_offset, 0.001)
        init_weights_normal(self.fc_mask_offset, 0.001)
        with torch.no_grad():
            loc_bias_val = 1 - 0.2 * offset_bias_val
            offset_bias_tensor = self._init_bias_tensor(offset_bias_val).to(self.fc_offset[-1].bias)
            self.fc_offset[-1].bias.copy_(offset_bias_tensor)
            loc_bias_tensor = self._init_bias_tensor(loc_bias_val).to(self.fc_mask_offset[-1].bias)
            self.fc_mask_offset[-1].bias.copy_(loc_bias_tensor)

    def _init_bias_tensor(self, bias_val=1.0):
        assert self.num_point % 4 == 0, 'num_point must be divided by 4'
        k = self.num_point // 4
        ascend_val = torch.arange(-1, 1, 2 / k) * bias_val
        dscend_val = torch.arange(1, -1, -2 / k) * bias_val
        bias_val = torch.full((k, ), fill_value=bias_val)

        l_val = torch.stack((bias_val.clone() * -1, dscend_val.clone()), dim=-1)
        t_val = torch.stack((ascend_val.clone(), bias_val.clone() * -1), dim=-1)
        r_val = torch.stack((bias_val.clone(), ascend_val.clone()), dim=-1)
        b_val = torch.stack((dscend_val.clone(), bias_val.clone()), dim=-1)

        combined = [l_val[k//2:], t_val, r_val, b_val, l_val[:k//2]]
        bias_tensor = torch.cat(combined, dim=0)

        return bias_tensor.reshape(-1)

    def single_level_roi_extractor(self, rois, feature, stride):
        pooled_feats = self.roipool(rois, feature, stride)
        return pooled_feats

    def get_offsets(self, pooled_feat):
        c = pooled_feat.numel() // pooled_feat.shape[0]
        x = pooled_feat.view(-1, c)
        return self.fc_offset(x)

    def forward_net(self, deform_feats):
        mask_x = self.mask_transformer(deform_feats)
        mask_x = rearrange(mask_x, 'b n d -> b (n d)')
        mask_offset_pred = self.fc_mask_offset(mask_x)
        return mask_offset_pred
