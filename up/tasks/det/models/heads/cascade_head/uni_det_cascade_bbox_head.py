# Standard Library
import copy

# Import from third library
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from pod
from up import extensions as E
from up.utils.model import accuracy as A
from up.tasks.det.models.utils.assigner import map_rois_to_level
from up.utils.model.initializer import init_weights_normal, initialize_from_cfg
from up.models.losses import build_loss
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from einops import rearrange, repeat
from up.tasks.det.models.utils.transformer import TransformerBlock
from up.tasks.det.models.utils.bbox_helper import bbox_iou_overlaps
from up.tasks.det.models.postprocess.bbox_supervisor import build_bbox_supervisor
from up.tasks.det.models.postprocess.bbox_predictor import build_bbox_predictor


class UniCascadeBboxNet(nn.Module):
    """
    Cascade boxes prediction and refinement.
    """
    def __init__(self, inplanes, num_classes, cfg):
        """
        Arguments:
            - inplanes (:obj:`list` or :obj:`int`): input channel,
              which is a number or list contains a single element
            - num_classes (:obj:`int`): number of classes, including the background class
            - cfg (:obj:`dict`): config for training or test
        """
        super(UniCascadeBboxNet, self).__init__()

        self.cfg = copy.deepcopy(cfg)  # runtime configuration
        self.cfg['num_classes'] = num_classes
        self.num_stage = self.cfg.get('num_stage', 1)
        self.stage_weights = self.cfg.get('stage_weights', None)
        self.stage_loc_weights = self.cfg.get('stage_loc_weights', [1.0, 1.0, 1.0])
        self.test_ensemble = self.cfg.get('test_ensemble', True)

        self.supervisor = build_bbox_supervisor(cfg['bbox_supervisor'])
        self.predictor = build_bbox_predictor(cfg['bbox_predictor'])

        if isinstance(inplanes, list):
            assert len(inplanes) == 1, 'single input is expected, but found:{} '.format(inplanes)
            inplanes = inplanes[0]
        assert isinstance(inplanes, int)
        self.inplanes = inplanes

        roipool = E.build_generic_roipool(cfg['roipooling'])
        self.roipool_list = nn.ModuleList()
        for i in range(self.num_stage):
            self.roipool_list.append(roipool)
        self.pool_size = cfg['roipooling']['pool_size']

        self.cls_loss = build_loss(self.cfg['cls_loss'])
        self.loc_loss_iou = build_loss(self.cfg['loc_loss_iou'])
        
        if 'centerness_loss' in cfg:
            self.centerness_loss = build_loss(cfg['centerness_loss'])
            self.use_centerness = True
        else:
            self.use_centerness = False

        self.prefix = 'CascadeBboxNet'

    def forward_net(self, rois, x, stride, stage):
        raise NotImplementedError

    def forward(self, input):
        prefix = self.prefix

        output = {}
        if self.training:
            if self.use_centerness:
                stage_sample_record, stage_cls_loss, stage_loc_loss, stage_centerness_loss, stage_acc = self.get_loss(input)
                for i in range(self.num_stage):
                    # output['sample_record_' + str(i)] = stage_sample_record[i]
                    output['sample_record'] = stage_sample_record[i]
                    output[prefix + '.cls_loss_' + str(i)] = stage_cls_loss[i]
                    output[prefix + '.loc_loss_' + str(i)] = stage_loc_loss[i]
                    output[prefix + '.centerness_loss_' + str(i)] = stage_centerness_loss[i]
                    output[prefix + '.accuracy_' + str(i)] = stage_acc[i]
            else:
                stage_sample_record, stage_cls_loss, stage_loc_loss, stage_acc = self.get_loss(input)
                for i in range(self.num_stage):
                    # output['sample_record_' + str(i)] = stage_sample_record[i]
                    output['sample_record'] = stage_sample_record[i]
                    output[prefix + '.cls_loss_' + str(i)] = stage_cls_loss[i]
                    output[prefix + '.loc_loss_' + str(i)] = stage_loc_loss[i]
                    output[prefix + '.accuracy_' + str(i)] = stage_acc[i]
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

    def mlvl_predict(self, x_rois, x_features, x_strides, levels, stage):
        """Predict results level by level"""
        mlvl_rois = []
        mlvl_pooled_feats = []
        mlvl_deform_feats = []
        offsets = []
        for lvl_idx in levels:
            if x_rois[lvl_idx].numel() > 0:
                lvl_rois = x_rois[lvl_idx]       # K*5
                lvl_feature = x_features[lvl_idx]  # N*C*H*W
                lvl_stride = x_strides[lvl_idx]
                pooled_feat = self.single_level_roi_extractor(lvl_rois, lvl_feature, lvl_stride, stage)  # K*C*H*W
                offset = self.get_offsets(pooled_feat, stage)  # K*(N_point*2)
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
        pred_cls, pred_loc, pred_centerness = self.forward_net(pooled_feats, deform_feats, stage)

        return pred_cls, (offsets, pred_loc), pred_centerness

    def get_head_output(self, rois, features, strides, stage):
        """
        Assign rois to each level and predict
        Returns:
            rois (FloatTensor): assigned rois
            cls_pred (FloatTensor, fp32): prediction of classification of assigned rois
            loc_pred (FloatTensor, fp32): prediction of localization of assigned rois
            recover_inds (LongTensor): indices of recovering input rois from assigned rois
        """
        if self.cfg.get('fpn', None):
            # assign rois and targets to each level
            fpn = self.cfg['fpn']
            mlvl_rois, recover_inds = map_rois_to_level(fpn['fpn_levels'], fpn['base_scale'], rois)
            cls_pred, loc_pred, centerness_pred = self.mlvl_predict(mlvl_rois, features, strides, fpn['fpn_levels'], stage)
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

    def get_loss(self, input):
        """
        Arguments:
            input['features'] (list): input feature layers, for C4 from backbone, others from FPN
            input['strides'] (list): strides of input feature layers
            input['image_info'] (list of FloatTensor): [B, 5] (reiszed_h, resized_w, scale_factor, origin_h, origin_w)
            input['dt_bboxes'] (FloatTensor): [N, >=7] (batch_ix, x1, y1, x2, y2, score, cls)
            input['gt_bboxes'] (list of FloatTensor): [B, num_gts, 5] (x1, y1, x2, y2, label)
            input['gt_ignores'] (list of FloatTensor): [B, num_igs, 4] (x1, y1, x2, y2)

        Returns:
            sample_record (list of tuple): [B, (pos_inds, pos_target_gt_inds)], saved for mask/keypoint head
            cls_loss, loc_loss, acc (FloatTensor)
        """
        features = input['features']
        strides = input['strides']
        image_info = input['image_info']
        B = len(image_info)
        rois = input['dt_bboxes']

        stage_sample_record = []
        stage_cls_loss = []
        stage_loc_loss = []
        stage_centerness_loss = []
        stage_acc = []
        for cascade_i in range(self.num_stage):
            stage_weight = self.stage_weights[cascade_i]
            stage_loc_weight = self.stage_loc_weights[cascade_i]
            # cascade_i_cfg = self.get_cascade_stage_cfg(cascade_i)
            # cls_target (LongTensor): [R]
            # loc_target (FloatTensor): [R, 4]
            # loc_weight (FloatTensor): [R, 4]
            sample_record, sampled_rois, cls_target, loc_target, loc_weight, pos_gts, gt_flags = \
                self.supervisor.get_targets(cascade_i, rois, input)
            rois, cls_pred, (offset_pred, loc_pred), centerness_pred, recover_inds = self.get_head_output(
                sampled_rois, features, strides, cascade_i)
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

            pos_iou_bbox_pred = self.get_regression_boxes(
                rois[pos_inds, 1:5], offset_pred[pos_inds], loc_pred[pos_inds])
            iou_target = bbox_iou_overlaps(pos_iou_bbox_pred.detach(), pos_gts[:, :4], aligned=True)
            loc_target_iou = loc_target[pos_inds]

            loc_loss = self.loc_loss_iou(pos_iou_bbox_pred, loc_target_iou,
                                             weights=iou_target.clamp(1e-12),
                                             normalizer_override=iou_target.sum())
            loc_loss = loc_loss * stage_loc_weight

            if self.cls_loss.activation_type == 'softmax':
                acc = A.accuracy(cls_pred, cls_target)[0]
            else:
                try:
                    acc = A.binary_accuracy(cls_pred, self.cls_loss.expand_target)[0]
                except:  # noqa
                    acc = cls_pred.new_zeros(1)

            if self.use_centerness:
                centerness_loss = self.centerness_loss(centerness_pred[pos_inds], iou_target)
                centerness_loss = centerness_loss * stage_weight
                stage_centerness_loss.append(centerness_loss)
            # collect cascade stage loss and accuracy
            cls_loss = cls_loss * stage_weight
            loc_loss = loc_loss * stage_weight
            stage_sample_record.append(sample_record)
            stage_cls_loss.append(cls_loss)
            stage_loc_loss.append(loc_loss)
            stage_acc.append(acc)
            # refine bboxes before the last stage
            if cascade_i < self.num_stage - 1:
                with torch.no_grad():
                    rois = self.predictor.refine(cascade_i, rois, cls_target, (offset_pred, loc_pred), image_info, gt_flags)
        if self.use_centerness:
            return stage_sample_record, stage_cls_loss, stage_loc_loss, stage_centerness_loss, stage_acc
        else:
            return stage_sample_record, stage_cls_loss, stage_loc_loss, stage_acc

    def get_bboxes(self, input):
        features = input['features']
        strides = input['strides']
        image_info = input['image_info']
        rois = input['dt_bboxes']

        stage_scores = []
        for cascade_i in range(self.num_stage):
            rois, cls_pred, loc_pred, centerness_pred, recover_inds = self.get_head_output(rois, features, strides, cascade_i)
            rois = rois.detach()[recover_inds]
            cls_pred = cls_pred.detach()[recover_inds]
            loc_pred = (loc_pred[0].detach()[recover_inds], loc_pred[1].detach()[recover_inds])
            # cls_pred = F.softmax(cls_pred, dim=1)
            if self.cls_loss.activation_type == 'softmax':
                cls_pred = F.softmax(cls_pred, dim=1)
            else:
                cls_pred = torch.sigmoid(cls_pred)
            if self.use_centerness:
                centerness_pred = torch.sigmoid(centerness_pred)
                cls_pred = cls_pred * centerness_pred
            stage_scores.append(cls_pred)

            if cascade_i < self.num_stage - 1:
                rois = self.predictor.refine(
                    cascade_i, rois, cls_pred.argmax(dim=1), loc_pred, image_info)

        if self.test_ensemble:
            cls_pred = sum(stage_scores) / self.num_stage

        start_idx = 0 if self.cls_loss.activation_type == 'sigmoid' else 1
        results = self.predictor.predict(rois, (cls_pred, loc_pred), image_info, start_idx=start_idx)
        return results


@MODULE_ZOO_REGISTRY.register('UniCascadeBboxFC')
class UniCascadeBboxFC(UniCascadeBboxNet):
    """
    Use FC as head
    """
    def __init__(self, inplanes, feat_planes, num_classes, cfg,
                 num_point=16, cls_block_num=1, loc_block_num=1,
                 offset_bias_val=3.0,
                 initializer=None, dropout=0.1):
        """
        Arguments:
            - inplanes (:obj:`list` or :obj:`int`): input channel,
              which is a number or list contains a single element
            - feat_planes (:obj:`int`): channels of intermediate features
            - num_classes (:obj:`int`): number of classes, including the background class
            - cfg (:obj:`dict`): config for training or test
            - initializer (:obj:`dict`): config for module parameters initialization

        """
        super(UniCascadeBboxFC, self).__init__(inplanes, num_classes, cfg)

        inplanes = self.inplanes
        self.num_point = num_point
        self.bilinear_detach = True
        self.decays = self.cfg['stage_bbox_normalize'].get('decays', [1.0, 1.0, 1.0])

        if self.cls_loss.activation_type == 'sigmoid':
            cls_out_channel = num_classes - 1
        elif self.cls_loss.activation_type == 'softmax':
            cls_out_channel = num_classes
        else:
            cls_out_channel = self.cls_loss.get_channel_num(num_classes)

        self.fc_offset_list = nn.ModuleList()
        self.cls_transformer_list = nn.ModuleList()
        self.loc_transformer_list = nn.ModuleList()
        self.fc_rcnn_cls_list = nn.ModuleList()
        self.fc_rcnn_loc_list = nn.ModuleList()
        self.fc_rcnn_centerness_list = nn.ModuleList()
        self.cls_token_list = nn.ParameterList()
        self.centerness_token_list = nn.ParameterList()

        self.relu = nn.ReLU(inplace=True)
        cls_out_channel = num_classes if self.cls_loss.activation_type == 'softmax' else num_classes - 1
        for i in range(self.num_stage):
            fc_offset = nn.Sequential(
                nn.Linear(self.pool_size * self.pool_size * inplanes, inplanes),
                nn.ReLU(inplace=True),
                nn.Linear(inplanes, self.num_point * 2))
            self.fc_offset_list.append(fc_offset)

            if cls_block_num > 1:
                cls_transformer = nn.Sequential(
                    *[TransformerBlock(
                        self.inplanes, num_heads=8, activation='gelu', dropout=dropout, norm_style='pre_norm')
                        for _ in range(cls_block_num)])
            elif cls_block_num == 1:
                cls_transformer = TransformerBlock(
                    self.inplanes, num_heads=8, activation='gelu', dropout=dropout, norm_style='pre_norm')
            else:
                cls_transformer = nn.Identity()
            self.cls_transformer_list.append(cls_transformer)

            fc_rcnn_cls = nn.Sequential(
                nn.Linear(self.inplanes * (self.num_point + 1), feat_planes),
                nn.ReLU(inplace=True),
                nn.Linear(feat_planes, cls_out_channel))
            self.fc_rcnn_cls_list.append(fc_rcnn_cls)

            if loc_block_num > 1:
                loc_transformer = nn.Sequential(
                    *[TransformerBlock(
                        self.inplanes, num_heads=8, activation='gelu', dropout=dropout, norm_style='pre_norm')
                        for _ in range(loc_block_num)])
            elif loc_block_num == 1:
                loc_transformer = TransformerBlock(
                    self.inplanes, num_heads=8, activation='gelu', dropout=dropout, norm_style='pre_norm')
            else:
                loc_transformer = nn.Identity()
            self.loc_transformer_list.append(loc_transformer)

            if self.cfg.get('share_location', False):
                fc_rcnn_loc = nn.Sequential(
                    nn.Linear(self.inplanes * self.num_point, feat_planes // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(feat_planes // 2, self.num_point * 2))
            else:
                fc_rcnn_loc = nn.Sequential(
                    nn.Linear(self.inplanes * self.num_point, feat_planes // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(feat_planes // 2, self.num_point * 2 * cls_out_channel))
            self.fc_rcnn_loc_list.append(fc_rcnn_loc)

            self.fc_rcnn_centerness_list.append(nn.Linear(inplanes, 1))

        initialize_from_cfg(self, initializer)
        init_weights_normal(self.fc_rcnn_cls_list, 0.01)
        init_weights_normal(self.fc_rcnn_centerness_list, 0.01)
        init_weights_normal(self.fc_offset_list, 0.001)
        init_weights_normal(self.fc_rcnn_loc_list, 0.001)

        for i in range(self.num_stage):
            cls_token = nn.Parameter(torch.randn(1, 1, self.inplanes))
            centerness_token = nn.Parameter(torch.randn(1, 1, self.inplanes))
            self.cls_token_list.append(cls_token)
            self.centerness_token_list.append(centerness_token)
            n_cls = 1 if self.cfg.get('share_location', False) else cls_out_channel

            with torch.no_grad():
                loc_bias_val = (1 - 0.2 * offset_bias_val) / self.decays[i]
                offset_bias_tensor = self._init_offset_bias_tensor(offset_bias_val).to(self.fc_offset_list[i][-1].bias)
                self.fc_offset_list[i][-1].bias.copy_(offset_bias_tensor)
                loc_bias_tensor = self._init_loc_bias_tensor(n_cls, loc_bias_val).to(self.fc_rcnn_cls_list[i][-1].bias)
                self.fc_rcnn_loc_list[i][-1].bias.copy_(loc_bias_tensor)

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

    def single_level_roi_extractor(self, rois, feature, stride, stage):
        pooled_feats = self.roipool_list[stage](rois, feature, stride)
        return pooled_feats

    def get_offsets(self, pooled_feat, stage):
        c = pooled_feat.numel() // pooled_feat.shape[0]
        x = pooled_feat.view(-1, c)
        return self.fc_offset_list[stage](x)

    def forward_net(self, pooled_feats, deform_feats, stage):
        cls_tokens = repeat(
            self.cls_token_list[stage], '() n d -> b n d', b=deform_feats.shape[0])
        cls_x = torch.cat((cls_tokens, deform_feats), dim=1)
        cls_x = self.cls_transformer_list[stage](cls_x)
        cls_x = rearrange(cls_x, 'b n d -> b (n d)')
        cls_pred = self.fc_rcnn_cls_list[stage](cls_x)

        if self.use_centerness:
            centerness_tokens = repeat(
                self.centerness_token_list[stage], '() n d -> b n d', b=deform_feats.shape[0])

            loc_x = torch.cat((centerness_tokens, deform_feats), dim=1)
            loc_x = self.loc_transformer_list[stage](loc_x)
            centerness_x = loc_x[:, 0, :]
            centerness_pred = self.fc_rcnn_centerness_list[stage](centerness_x)

            loc_x = rearrange(loc_x[:, 1:, :], 'b n d -> b (n d)')
            loc_pred = self.fc_rcnn_loc_list[stage](loc_x)
        else:
            loc_x = self.loc_transformer_list[stage](deform_feats)
            loc_x = rearrange(loc_x, 'b n d -> b (n d)')
            loc_pred = self.fc_rcnn_loc_list[stage](loc_x)

            centerness_pred = cls_pred.new_zeros((cls_pred.shape[0], 1))

        return cls_pred, loc_pred, centerness_pred