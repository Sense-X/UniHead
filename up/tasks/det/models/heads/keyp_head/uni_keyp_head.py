# modified from https://github.com/ModelTC/United-Perception
import copy

import torch
import torch.nn as nn

from up import extensions as E
from up.tasks.det.models.utils.assigner import map_rois_to_level
from up.utils.model.initializer import init_weights_normal, initialize_from_cfg
from up.models.losses import build_loss
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from einops import rearrange, repeat
from up.tasks.det.models.utils.transformer import TransformerBlock
from up.tasks.det.models.postprocess import build_keypoint_predictor, build_keypoint_supervisor


class KeypPointNet(nn.Module):
    def __init__(self, num_classes, cfg):
        super(KeypPointNet, self).__init__()

        self.cfg = copy.deepcopy(cfg)
        self.cfg['num_classes'] = num_classes
        self.share_location = self.cfg.get('share_location', False)
        self.num_classes = num_classes

        self.roipool = E.build_generic_roipool(self.cfg['roipooling'])
        self.pool_size = cfg['roipooling']['pool_size']
        self.keyp_loss = build_loss(self.cfg['keyp_loss'])
        self.vis_loss = build_loss(self.cfg['vis_loss'])

        self.keyp_supervisor = build_keypoint_supervisor(cfg['keyp_supervisor'])
        self.keyp_predictor = build_keypoint_predictor(cfg['keyp_predictor'])
        self.prefix = 'KeypNet'

    def forward_net(self, x):
        raise NotImplementedError

    def forward(self, input):
        if self.training:
            return self.get_loss(input)
        else:
            return self.get_keyp(input)

    def mlvl_predict(self, x_rois, x_features, x_strides, levels, input=None):
        """Predict results level by level"""
        mlvl_deform_feats = []
        ini_offsets = []
        mlvl_rois = []
        mlvl_strides = []

        for lvl_idx in levels:
            if x_rois[lvl_idx].numel() > 0:
                lvl_rois = x_rois[lvl_idx]
                lvl_feature = x_features[lvl_idx]
                lvl_stride = x_strides[lvl_idx]
                pooled_feat = self.single_level_roi_extractor(lvl_rois, lvl_feature, lvl_stride)
                ini_offset = self.get_offsets(pooled_feat)
                deform_feats = self.get_deform_feats(lvl_feature, lvl_rois, ini_offset, lvl_stride, self.num_point)
                mlvl_strides.append(ini_offset.new_full((ini_offset.shape[0], ), fill_value=lvl_stride))
                mlvl_rois.append(lvl_rois)
                mlvl_deform_feats.append(deform_feats)
                ini_offsets.append(ini_offset)
        assert len(mlvl_rois) > 0, "No rois provided for second stage"
        deform_feats = torch.cat(mlvl_deform_feats, dim=0)
        ini_offsets = torch.cat(ini_offsets, dim=0)
        keyp_offsets, keyp_vis_pred = self.forward_net(deform_feats)
        keyp_strides = torch.cat(mlvl_strides, dim=0)
        return ini_offsets, keyp_offsets, keyp_vis_pred, keyp_strides

    def get_logits(self, rois, features, strides, input=None):
        assert self.cfg.get('fpn', None) is not None
        fpn = self.cfg['fpn']
        mlvl_rois, recover_inds = map_rois_to_level(fpn['fpn_levels'], fpn['base_scale'], rois)
        ini_offset_pred, keyp_offset_pred, keyp_vis_pred, keyp_strides = self.mlvl_predict(mlvl_rois, features, strides, fpn['fpn_levels'], input)
        rois = torch.cat(mlvl_rois, dim=0)
        return rois, ini_offset_pred.float(), keyp_offset_pred.float(), keyp_vis_pred.float(), keyp_strides, recover_inds

    def get_pred_keyp(self, rois, ini_offset_pred, mask_offset_pred):
        ini_offset = ini_offset_pred.reshape(rois.shape[0], -1, 2)
        mask_offset = mask_offset_pred.reshape(rois.shape[0], -1, 2)
        centers_x = (rois[:, 0] + rois[:, 2]) / 2
        centers_y = (rois[:, 1] + rois[:, 3]) / 2
        w_, h_ = (rois[:, 2] - rois[:, 0] + 1), (rois[:, 3] - rois[:, 1] + 1)
        centers = torch.stack((centers_x, centers_y), dim=-1).unsqueeze(1)
        wh_ = torch.stack((w_, h_), dim=-1).unsqueeze(1)
        keyp = centers + ini_offset * wh_ * 0.1 + mask_offset * wh_ * 0.5
        return keyp

    def get_loss(self, input):
        features = input['features']
        strides = input['strides']

        sampled_rois, keyp_target, gt_bbox_target = self.keyp_supervisor.get_targets(input)
        if sampled_rois[0].sum() == 0:
            return {self.prefix + '.keyp_loss': input['BboxNet.cls_loss'] * 0,
                  self.prefix + '.keyp_vis_loss': input['BboxNet.cls_loss'] * 0}
        rois, ini_offset, keyp_offset, keyp_vis, keyp_strides, recover_inds = self.get_logits(sampled_rois, features, strides, input)

        ini_offset = ini_offset[recover_inds]
        keyp_offset = keyp_offset[recover_inds]
        rois = rois[recover_inds]
        keyp_strides = keyp_strides[recover_inds]
        keyp_vis = keyp_vis[recover_inds]
        normalize_term = keyp_strides.reshape(-1, 1, 1) * 8

        vs = keyp_target[:, :, 2]
        vs_mask = (vs > 0)
        keyp_target_ = keyp_target[:, :, :2]
        keyp_pred = self.get_pred_keyp(rois[:, 1:5], ini_offset, keyp_offset)
        
        keyp_loss = self.keyp_loss(keyp_pred/normalize_term, keyp_target_/normalize_term,
                                    reduction_override='none')
        keyp_loss = vs_mask.unsqueeze(-1) * keyp_loss
        keyp_loss = keyp_loss.sum() / vs_mask.sum()

        vis_loss = self.vis_loss(keyp_vis, vs_mask.float())

        output = {self.prefix + '.keyp_loss': keyp_loss,
                  self.prefix + '.keyp_vis_loss': vis_loss}

        return output

    def get_keyp(self, input):
        features = input['features']
        strides = input['strides']
        rois = input['dt_bboxes']

        bboxes, ini_offset, mask_offset, keyp_vis, _, recover_inds = self.get_logits(rois, features, strides)
        keyp_pred = self.get_pred_keyp(bboxes[:, 1:5], ini_offset, mask_offset)

        results = self.keyp_predictor.predict(keyp_pred, keyp_vis, bboxes, input)
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


@MODULE_ZOO_REGISTRY.register('UniKeypHead')
class KeypPointHead(KeypPointNet):
    def __init__(self,
                 inplanes,
                 num_classes,
                 feat_planes,
                 cfg,
                 num_point=17,
                 initializer=None,
                 dropout=0.1,
                 keyp_block_num=1,
                 use_keyp_box=True):
        super(KeypPointHead, self).__init__(num_classes, cfg)
        self.num_point = num_point
        self.inplanes = inplanes
        self.bilinear_detach = True

        if isinstance(inplanes, list):
            assert len(inplanes) == 1, 'single input is expected, but found:{} '.format(inplanes)
            inplanes = inplanes[0]
        assert isinstance(inplanes, int)

        planes = feat_planes
        self.fc_offset = nn.Sequential(
            nn.Linear(self.pool_size * self.pool_size * inplanes, inplanes),
            nn.ReLU(inplace=True),
            nn.Linear(inplanes, self.num_point*2))

        if keyp_block_num > 1:
            self.keyp_transformer = nn.Sequential(
                *[TransformerBlock(
                    self.inplanes, num_heads=8, activation='gelu', dropout=dropout, norm_style='pre_norm')
                  for _ in range(keyp_block_num)])
        else:
            self.keyp_transformer = TransformerBlock(
                self.inplanes, num_heads=8, activation='gelu', dropout=dropout, norm_style='pre_norm')

        self.fc_keyp_offset = nn.Sequential(
            nn.Linear(self.num_point * inplanes, feat_planes),
            nn.ReLU(inplace=True),
            nn.Linear(feat_planes, self.num_point*2))

        self.fc_keyp_vis = nn.Linear(inplanes, num_point)

        if use_keyp_box:
            self.x_mean = [0.0009068, 0.02683756, -0.02802999, 0.07496057, -0.07512482, 0.13424154, -0.13591655, 0.18730339,
                           -0.18922948, 0.1123826, -0.11387086, 0.08209435, -0.07935154, 0.07428858, -0.06675689,
                           0.06871676, -0.06004739]
            self.y_mean = [-0.419478, -0.47120036, -0.47009499, -0.45973552, -0.45793597,
                           -0.26638743, -0.26399606, -0.06629673, -0.06602572, -0.02154654,
                           -0.02938734, 0.11578116, 0.11653638, 0.23129613, 0.23171223,
                            0.45934125, 0.45737722]
        else:
            ## use bounding box (original)
            self.x_mean = [0.00573894, 0.06648359, -0.05234604, 0.16543678, -0.16315118,
                             0.11272677, -0.11892816, 0.17986859, -0.19158753, 0.12022895,
                             -0.139075, 0.06282358, -0.07855126, 0.06115148, -0.07093605,
                             0.04971221, -0.06163098]
            self.y_mean = [-0.32313928, -0.36213202, -0.3616331, -0.3416236, -0.34096858,
                             -0.18487627, -0.18246655, -0.00791439, -0.00914987, 0.03151885,
                             0.01991513, 0.15773121, 0.1583971, 0.27798129, 0.277898,
                             0.47950379, 0.4757622]

        initialize_from_cfg(self, initializer)

        init_weights_normal(self.fc_offset, 0.001)
        init_weights_normal(self.fc_keyp_offset, 0.001)

        with torch.no_grad():
            offset_bias_tensor = self._init_ini_bias_tensor().to(self.fc_offset[-1].bias)
            self.fc_offset[-1].bias.copy_(offset_bias_tensor)
            keyp_bias_tensor = self._init_keyp_bias_tensor().to(self.fc_keyp_offset[-1].bias)
            self.fc_keyp_offset[-1].bias.copy_(keyp_bias_tensor)

        self.vis_token = nn.Parameter(torch.randn(1, 1, inplanes))

    def single_level_roi_extractor(self, rois, feature, stride):
        pooled_feats = self.roipool(rois, feature, stride)
        return pooled_feats

    def _init_ini_bias_tensor(self):
        x_all = torch.tensor(self.x_mean)
        y_all = torch.tensor(self.y_mean)
        xy_all = torch.stack((x_all, y_all), dim=-1)
        bias_tensor = xy_all / (2 * 0.1)
        return bias_tensor.reshape(-1)

    def _init_keyp_bias_tensor(self):
        x_all = torch.tensor(self.x_mean)
        y_all = torch.tensor(self.y_mean)
        bias_tensor = torch.stack((x_all, y_all), dim=-1)
        return bias_tensor.reshape(-1)

    def get_offsets(self, pooled_feat):
        c = pooled_feat.numel() // pooled_feat.shape[0]
        x = pooled_feat.view(-1, c)
        return self.fc_offset(x)

    def forward_net(self, deform_feats):
        vis_tokens = repeat(
            self.vis_token, '() n d -> b n d', b=deform_feats.shape[0])
        keyp_x = torch.cat((vis_tokens, deform_feats), dim=1)
        keyp_x = self.keyp_transformer(keyp_x)
        vis_pred = self.fc_keyp_vis(keyp_x[:, 0, :])

        keyp_x = rearrange(keyp_x[:, 1:, :], 'b n d -> b (n d)')
        keyp_offset_pred = self.fc_keyp_offset(keyp_x)
        return keyp_offset_pred, vis_pred
