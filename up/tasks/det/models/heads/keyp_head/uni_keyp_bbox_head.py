# modified from https://github.com/ModelTC/United-Perception
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from up import extensions as E
from up.utils.model import accuracy as A
from up.tasks.det.models.utils.assigner import map_rois_to_level
from up.utils.model.initializer import init_weights_normal, initialize_from_cfg, init_bias_focal
from up.models.losses import build_loss
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from einops import rearrange, repeat
from up.tasks.det.models.utils.transformer import TransformerBlock

from up.tasks.det.models.utils.bbox_helper import bbox_iou_overlaps
from up.tasks.det.models.postprocess.bbox_supervisor import build_bbox_supervisor
from up.tasks.det.models.postprocess.bbox_predictor import build_bbox_predictor


class KeypPointBboxNet(nn.Module):
    """
    classify proposals and compute their bbox regression.
    """

    def __init__(self, inplanes, num_classes, cfg):
        super(KeypPointBboxNet, self).__init__()
        self.prefix = 'BboxNet'
        self.num_classes = num_classes
        self.means = cfg['bbox_normalize']['means']
        self.stds = cfg['bbox_normalize']['stds']

        if isinstance(inplanes, list):
            assert len(inplanes) == 1, 'single input is expected, but found:{} '.format(inplanes)
            inplanes = inplanes[0]
        assert isinstance(inplanes, int)
        self.inplanes = inplanes

        self.supervisor = build_bbox_supervisor(cfg['bbox_supervisor'])
        self.predictor = build_bbox_predictor(cfg['bbox_predictor'])

        self.roipool = E.build_generic_roipool(cfg['roipooling'])
        self.pool_size = cfg['roipooling']['pool_size']

        self.cls_loss = build_loss(cfg['cls_loss'])
        self.loc_loss_iou = build_loss(cfg['loc_loss_iou'])

        if 'centerness_loss' in cfg:
            self.centerness_loss = build_loss(cfg['centerness_loss'])
            self.use_centerness = True
        else:
            self.use_centerness = False

        self.cfg = copy.deepcopy(cfg)
        self.counter = 0

    def roi_extractor(self, mlvl_rois, mlvl_features, mlvl_strides):
        raise NotImplementedError

    def forward_net(self, rois, x, stride):
        raise NotImplementedError

    def keyp2bbox(self, keyp, bbox_cls):
        valid_inds = (keyp[:, 2] != 0)
        valid_keyps = keyp[valid_inds, :2]
        if len(valid_keyps) <= 3:
            return None
        else:
            x0 = valid_keyps[:, 0].min().item()
            y0 = valid_keyps[:, 1].min().item()
            x1 = valid_keyps[:, 0].max().item()
            y1 = valid_keyps[:, 1].max().item()
            return [x0, y0, x1, y1, bbox_cls]

    def get_one_img_gts(self, gt_bboxes, gt_keyps):
        new_gt_bboxes, new_gt_keyps = [], []
        for i in range(len(gt_bboxes)):
            if self.use_keyp_box:
                bbox_cls = int(gt_bboxes[i, 4])
                new_bbox = self.keyp2bbox(gt_keyps[i], bbox_cls)
            else:
                new_bbox = gt_bboxes[i]

            if new_bbox is not None:
                new_gt_bboxes.append(torch.tensor(new_bbox).to(gt_bboxes))
                new_gt_keyps.append(gt_keyps[i])
        if len(new_gt_bboxes) == 0:
            new_gt_bboxes = gt_bboxes.new_zeros((0, 5))
            new_gt_keyps = gt_bboxes.new_zeros((0, 17, 3))
        else:
            new_gt_bboxes = torch.stack(new_gt_bboxes)
            new_gt_keyps = torch.stack(new_gt_keyps)

        return new_gt_bboxes, new_gt_keyps

    def get_new_gts(self, input):
        gt_bboxes = input['gt_bboxes']
        gt_keyps = input['gt_keyps']
        new_gt_bboxes, new_gt_keyps = [], []
        for i in range(len(gt_bboxes)):
            b_new_gt_bbox, b_new_gt_keyp = self.get_one_img_gts(gt_bboxes[i], gt_keyps[i])
            new_gt_bboxes.append(b_new_gt_bbox)
            new_gt_keyps.append(b_new_gt_keyp)

        return new_gt_bboxes, new_gt_keyps

    def forward(self, input):
        output = {}
        if self.training:
            new_gt_bboxes, new_gt_keyps = self.get_new_gts(input)
            input['gt_bboxes'] = new_gt_bboxes
            input['gt_keyps'] = new_gt_keyps
            losses = self.get_loss(input)
            output.update(losses)
            output['gt_keyps'] = new_gt_keyps
            output['gt_bboxes'] = new_gt_bboxes
        else:
            results = self.get_bboxes(input)
            output.update(results)
        return output

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

        lh = h - h_low
        lw = w - w_low
        hh = 1 - lh
        hw = 1 - lw

        v1 = feat_map[:, h_low, w_low]
        v2 = feat_map[:, h_low, w_high]
        v3 = feat_map[:, h_high, w_low]
        v4 = feat_map[:, h_high, w_high]

        w1 = hh * hw
        w2 = hh * lw
        w3 = lh * hw
        w4 = lh * lw
        w1, w2, w3, w4 = [x.unsqueeze(0) for x in [w1, w2, w3, w4]]

        val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
        return val

    def get_deform_feats(self, feat_map, rois, offset, stride, num_point):
        B = len(feat_map)
        deform_feats = torch.zeros(rois.shape[0], num_point, feat_map.shape[1]).to(feat_map)

        for b_ix in range(B):
            batch_roi_idx = (rois[:, 0] == b_ix)
            batch_scatter_index = torch.nonzero(batch_roi_idx).reshape(-1)
            if len(batch_scatter_index) == 0:
                continue

            batch_rois = rois[batch_roi_idx]
            batch_offset = offset[batch_roi_idx]
            batch_offset = batch_offset.reshape(len(batch_scatter_index), -1, 2)

            batch_feat_map = feat_map[b_ix, ...]
            centers_x = (batch_rois[:, 1] + batch_rois[:, 3]) / 2
            centers_y = (batch_rois[:, 2] + batch_rois[:, 4]) / 2
            w_ = batch_rois[:, 3] - batch_rois[:, 1] + 1
            h_ = batch_rois[:, 4] - batch_rois[:, 2] + 1
            centers = torch.stack((centers_x, centers_y), dim=-1).unsqueeze(1)
            wh_ = torch.stack((w_, h_), dim=-1).unsqueeze(1)

            batch_index = centers + batch_offset * wh_ * 0.1
            batch_index = batch_index.reshape(-1, 2)
            batch_deform_feat = self.bilinear_gather(batch_feat_map, batch_index / stride)

            batch_deform_feat = rearrange(
                batch_deform_feat, 'c (nb np) -> nb np c', nb=len(batch_scatter_index))
            deform_feats[batch_scatter_index] = batch_deform_feat
            del batch_roi_idx

        return deform_feats

    def mlvl_predict(self, x_rois, x_features, x_strides, levels, input=None):
        """Predict results level by level"""
        mlvl_rois = []
        mlvl_pooled_feats = []
        mlvl_deform_feats = []
        offsets = []
        for lvl_idx in levels:
            if x_rois[lvl_idx].numel() > 0:
                lvl_rois = x_rois[lvl_idx]
                lvl_feature = x_features[lvl_idx]
                lvl_stride = x_strides[lvl_idx]
                pooled_feat = self.single_level_roi_extractor(lvl_rois, lvl_feature, lvl_stride)  # K*C*H*W
                offset = self.get_offsets(pooled_feat)
                deform_feats = self.get_deform_feats(lvl_feature, lvl_rois, offset, lvl_stride, self.num_point)
                mlvl_rois.append(lvl_rois)
                mlvl_pooled_feats.append(pooled_feat)
                mlvl_deform_feats.append(deform_feats)
                offset_stride = offset.new_full((offset.shape[0], 1), lvl_stride)
                offsets.append(torch.cat((offset_stride, offset), dim=-1))
        assert len(mlvl_rois) > 0, "No rois provided for second stage"
        pooled_feats = torch.cat(mlvl_pooled_feats, dim=0)
        deform_feats = torch.cat(mlvl_deform_feats, dim=0)
        offsets = torch.cat(offsets, dim=0)
        pred_cls, pred_loc, pred_centerness = self.forward_net(pooled_feats, deform_feats)

        return pred_cls, (offsets, pred_loc), pred_centerness

    def get_head_output(self, rois, features, strides, input=None):
        if self.cfg.get('fpn', None):
            fpn = self.cfg['fpn']
            mlvl_rois, recover_inds = map_rois_to_level(fpn['fpn_levels'], fpn['base_scale'], rois)
            cls_pred, loc_pred, centerness_pred = self.mlvl_predict(mlvl_rois, features, strides, fpn['fpn_levels'], input)
            rois = torch.cat(mlvl_rois, dim=0)
        else:
            raise NotImplementedError
        return rois, cls_pred.float(), (loc_pred[0].float(), loc_pred[1].float()), centerness_pred.float(), recover_inds

    def get_regression_boxes(self, rois, offset_pred, loc_pred):
        offset = offset_pred[:, 1:]
        offset = offset.reshape(rois.shape[0], -1, 2)
        centers_x = (rois[:, 0] + rois[:, 2]) / 2
        centers_y = (rois[:, 1] + rois[:, 3]) / 2
        w_, h_ = (rois[:, 2] - rois[:, 0] + 1), (rois[:, 3] - rois[:, 1] + 1)
        centers = torch.stack((centers_x, centers_y), dim=-1).unsqueeze(1)
        wh_ = torch.stack((w_, h_), dim=-1).unsqueeze(1)
        offset_point = centers + offset * wh_ * 0.1
        loc_pred = loc_pred.reshape(rois.shape[0], self.num_point, 2)
        x_shift = loc_pred[..., 0] * w_.unsqueeze(1) * 0.5
        y_shift = loc_pred[..., 1] * h_.unsqueeze(1) * 0.5

        shifts = torch.stack((x_shift, y_shift), dim=-1)
        shifted_offset_point = offset_point + shifts

        x_min, _ = torch.min(shifted_offset_point[..., 0], dim=-1)
        y_min, _ = torch.min(shifted_offset_point[..., 1], dim=-1)
        x_max, _ = torch.max(shifted_offset_point[..., 0], dim=-1)
        y_max, _ = torch.max(shifted_offset_point[..., 1], dim=-1)
        iou_boxes = torch.stack((x_min, y_min, x_max, y_max), dim=-1)

        return iou_boxes

    def get_loss(self, input):
        features = input['features']
        strides = input['strides']
        image_info = input['image_info']
        B = len(image_info)

        sample_record, sampled_rois, cls_target, loc_target, loc_weight, pos_gts = self.supervisor.get_targets(input)
        rois, cls_pred, (offset_pred, loc_pred), centerness_pred, recover_inds = self.get_head_output(sampled_rois, features, strides)
        cls_pred = cls_pred[recover_inds]
        offset_pred = offset_pred[recover_inds]
        loc_pred = loc_pred[recover_inds]
        centerness_pred = centerness_pred[recover_inds]
        rois = rois[recover_inds]

        cls_inds = cls_target
        if self.cfg.get('share_location', 'False'):
            cls_inds = cls_target.clamp(max=0)

        N = loc_pred.shape[0]
        loc_pred = loc_pred.reshape(N, -1, self.num_point*2)
        inds = torch.arange(N, dtype=torch.int64, device=loc_pred.device)
        if self.cls_loss.activation_type == 'sigmoid' and not self.cfg.get('share_location', 'False'):
            cls_inds = cls_inds - 1
        loc_pred = loc_pred[inds, cls_inds].reshape(-1, self.num_point*2)

        cls_loss = self.cls_loss(cls_pred, cls_target)
        if self.cls_loss.activation_type == 'sigmoid':
            pos_inds = torch.nonzero(cls_target >= 0).reshape(-1)
        else:
            pos_inds = torch.nonzero(cls_target > 0).reshape(-1)

        output = {'sample_record': sample_record,
                  self.prefix + '.cls_loss': cls_loss}

        if len(pos_inds) > 0:
            pos_iou_bbox_pred = self.get_regression_boxes(
                rois[pos_inds, 1:5], offset_pred[pos_inds], loc_pred[pos_inds])
            iou_target = bbox_iou_overlaps(pos_iou_bbox_pred.detach(), pos_gts[:, :4], aligned=True)
            loc_target_iou = loc_target[pos_inds]

            # pos_bbox_pred = pos_bbox_pred[iou_target > 0]
            # loc_target = loc_target[iou_target > 0]
            loc_loss_iou = self.loc_loss_iou(pos_iou_bbox_pred, loc_target_iou,
                                             weights=iou_target.clamp(1e-12),
                                             normalizer_override=iou_target.sum())
            output.update({self.prefix + '.loc_loss': loc_loss_iou})
        else:
            output.update({self.prefix + '.loc_loss': cls_loss * 0})

        if self.cls_loss.activation_type == 'softmax':
            acc = A.accuracy(cls_pred, cls_target)[0]
        else:
            try:
                acc = A.binary_accuracy(cls_pred, self.cls_loss.expand_target)[0]
            except: # noqa
                acc = cls_pred.new_zeros(1)
        output.update({self.prefix + '.accuracy': acc})

        if self.use_centerness:
            if len(pos_inds) > 0:
                centerness_loss = self.centerness_loss(centerness_pred[pos_inds], iou_target)
                output.update({self.prefix + '.centerness_loss': centerness_loss})
            else:
                output.update({self.prefix + '.centerness_loss': cls_loss * 0})

        return output

    def get_bboxes(self, input):
        features = input['features']
        strides = input['strides']
        image_info = input['image_info']
        rois = input['dt_bboxes']
        rois, cls_pred, loc_pred, centerness_pred, recover_inds = self.get_head_output(rois, features, strides, input)
        if self.cls_loss.activation_type == 'sigmoid':
            cls_pred = torch.sigmoid(cls_pred)
        elif self.cls_loss.activation_type == 'softmax':
            cls_pred = F.softmax(cls_pred, dim=1)
        else:
            cls_pred = self.cls_loss.get_activation(cls_pred)

        if self.use_centerness:
            centerness_pred = torch.sigmoid(centerness_pred)
            cls_pred = cls_pred * centerness_pred
        start_idx = 0 if self.cls_loss.activation_type == 'sigmoid' else 1
        output = self.predictor.predict(rois, (cls_pred, loc_pred), image_info,
                                        start_idx=start_idx)
        return output


@MODULE_ZOO_REGISTRY.register('UniKeypBboxHead')
class KeypPointBboxFC(KeypPointBboxNet):
    def __init__(self, inplanes, feat_planes, num_classes, cfg, num_point=17,
                 bilinear_detach=True,
                 cls_block_num=1, loc_block_num=1, offset_bias_val=3.0,
                 initializer=None, dropout=0.1, use_keyp_box=True):
        super(KeypPointBboxFC, self).__init__(inplanes, num_classes, cfg)

        inplanes = self.inplanes
        self.num_point = num_point
        self.bilinear_detach = bilinear_detach
        self.use_keyp_box = use_keyp_box

        self.fc_offset = nn.Sequential(
            nn.Linear(self.pool_size * self.pool_size * inplanes, inplanes),
            nn.ReLU(inplace=True),
            nn.Linear(inplanes, self.num_point * 2))

        if self.cls_loss.activation_type == 'sigmoid':
            cls_out_channel = num_classes - 1
        elif self.cls_loss.activation_type == 'softmax':
            cls_out_channel = num_classes
        else:
            cls_out_channel = self.cls_loss.get_channel_num(num_classes)

        if cls_block_num > 1:
            self.cls_transformer = nn.Sequential(
                *[TransformerBlock(
                    self.inplanes, num_heads=8, activation='gelu', dropout=dropout, norm_style='pre_norm')
                  for _ in range(cls_block_num)])
        else:
            self.cls_transformer = TransformerBlock(
                self.inplanes, num_heads=8, activation='gelu', dropout=dropout, norm_style='pre_norm')

        self.fc_rcnn_cls = nn.Sequential(
            nn.Linear(self.inplanes * (self.num_point + 1), feat_planes),
            nn.ReLU(inplace=True),
            nn.Linear(feat_planes, cls_out_channel))
        if loc_block_num > 1:
            self.loc_transformer = nn.Sequential(
                *[TransformerBlock(
                    self.inplanes, num_heads=8, activation='gelu', dropout=dropout, norm_style='pre_norm')
                  for _ in range(loc_block_num)])
        else:
            self.loc_transformer = TransformerBlock(
                self.inplanes, num_heads=8, activation='gelu', dropout=dropout, norm_style='pre_norm')

        self.fc_rcnn_loc = nn.Sequential(
            nn.Linear(self.inplanes * self.num_point, feat_planes // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_planes // 2, self.num_point * 2 * cls_out_channel))

        self.fc_rcnn_centerness = nn.Linear(inplanes, 1)

        initialize_from_cfg(self, initializer)
        init_weights_normal(self.fc_rcnn_cls, 0.01)
        init_weights_normal(self.fc_rcnn_centerness, 0.01)
        init_weights_normal(self.fc_offset, 0.001)
        init_weights_normal(self.fc_rcnn_loc, 0.001)

        with torch.no_grad():
            loc_bias_val = 1 - 0.2 * offset_bias_val
            offset_bias_tensor = self._init_offset_bias_tensor(offset_bias_val).to(self.fc_offset[-1].bias)
            self.fc_offset[-1].bias.copy_(offset_bias_tensor)
            loc_bias_tensor = self._init_loc_bias_tensor(cls_out_channel, loc_bias_val).to(self.fc_rcnn_cls[-1].bias)
            self.fc_rcnn_loc[-1].bias.copy_(loc_bias_tensor)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.inplanes))
        self.centerness_token = nn.Parameter(torch.randn(1, 1, self.inplanes))

        if 'sigmoid' in self.cls_loss.activation_type:
            init_prior = self.cls_loss.init_prior
            init_bias_focal(self.fc_rcnn_cls, 'sigmoid', init_prior, num_classes)

    def _init_loc_bias_tensor(self, n_cls, bias_val=1.0):
        assert self.num_point % 4 == 0, 'num_point must be divided by 4'
        bias_tensor = torch.tensor([[-bias_val, 0], [0, -bias_val], [bias_val, 0], [0, bias_val]]).reshape(4, 1, 2)  # 4 * 2
        bias_tensor = bias_tensor.repeat(1, self.num_point // 4, 1).reshape(1, self.num_point, 2)
        bias_tensor = bias_tensor.repeat(n_cls, 1, 1)
        return bias_tensor.reshape(-1)

    def _init_offset_bias_tensor(self, bias_val=1.0):
        assert self.num_point % 4 == 0, 'num_point must be divided by 4'
        bias_tensor = torch.tensor([[-bias_val, 0], [0, -bias_val], [bias_val, 0], [0, bias_val]]).reshape(4, 1, 2)  # 4 * 2
        bias_tensor = bias_tensor.repeat(1, self.num_point // 4, 1).reshape(self.num_point, 2)
        return bias_tensor.reshape(-1)

    def single_level_roi_extractor(self, rois, feature, stride):
        pooled_feats = self.roipool(rois, feature, stride)
        return pooled_feats

    def get_offsets(self, pooled_feat):
        c = pooled_feat.numel() // pooled_feat.shape[0]
        x = pooled_feat.view(-1, c)
        return self.fc_offset(x)

    def forward_net(self, pooled_feats, deform_feats):
        cls_tokens = repeat(
            self.cls_token, '() n d -> b n d', b=deform_feats.shape[0])
        cls_x = torch.cat((cls_tokens, deform_feats), dim=1)
        cls_x = self.cls_transformer(cls_x)
        cls_x = rearrange(cls_x, 'b n d -> b (n d)')
        cls_pred = self.fc_rcnn_cls(cls_x)

        centerness_tokens = repeat(
            self.centerness_token, '() n d -> b n d', b=deform_feats.shape[0])

        loc_x = torch.cat((centerness_tokens, deform_feats), dim=1)
        loc_x = self.loc_transformer(loc_x)
        centerness_x = loc_x[:, 0, :]
        centerness_pred = self.fc_rcnn_centerness(centerness_x)

        loc_x = rearrange(loc_x[:, 1:, :], 'b n d -> b (n d)')
        loc_pred = self.fc_rcnn_loc(loc_x)

        return cls_pred, loc_pred, centerness_pred
