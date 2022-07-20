# modified from https://github.com/ModelTC/United-Perception
import torch

from up import extensions as E
from up.tasks.det.models.utils.bbox_helper import (
    bbox_iou_overlaps, clip_bbox, filter_by_size)
from up.utils.general.fp16_helper import to_float32
from up.utils.general.registry_factory import ROI_SUPERVISOR_REGISTRY, ROI_PREDICTOR_REGISTRY
from up.tasks.det.models.postprocess.roi_predictor import build_merger


@ROI_SUPERVISOR_REGISTRY.register('atss_point')
class AtssPointSupervisor(object):
    def __init__(self, top_n=9, use_centerness=False, use_iou=False, gt_encode=True, return_gts=False,
                 num_point=16):
        self.top_n = top_n
        self.use_centerness = use_centerness
        self.use_iou = use_iou
        self.gt_encode = gt_encode
        self.return_gts = return_gts
        self.num_point = num_point

    def compute_centerness_targets(self, loc_target, anchors):
        gts = loc_target
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        _l = anchors_cx - gts[:, 0]
        t = anchors_cy - gts[:, 1]
        r = gts[:, 2] - anchors_cx
        b = gts[:, 3] - anchors_cy
        left_right = torch.stack([_l, r], dim=1)
        top_bottom = torch.stack([t, b], dim=1)
        centerness_target = torch.sqrt((left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))  # noqa E501
        assert not torch.isnan(centerness_target).any()
        return centerness_target

    def compute_iou_targets(self, loc_target, anchors, loc_pred):
        iou_target = bbox_iou_overlaps(loc_pred, loc_target, aligned=True)
        return iou_target

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

    @torch.no_grad()
    @to_float32
    def get_targets(self, mlvl_anchors, input, mlvl_preds=None):
        INF = 100000000
        gt_bboxes = input['gt_bboxes']
        ignore_regions = input.get('gt_ignores', None)
        num_anchors_per_level = [len(anchor) for anchor in mlvl_anchors]
        all_anchors = torch.cat(mlvl_anchors, dim=0)
        B = len(gt_bboxes)
        K = all_anchors.shape[0]
        neg_targets = input.get('neg_targets', None)
        if ignore_regions is None:
            ignore_regions = [None] * B
        if neg_targets is None:
            neg_targets = [0] * B
        cls_target = all_anchors.new_full((B, K), 0, dtype=torch.int64)
        loc_target = all_anchors.new_zeros((B, K, 4))
        sample_cls_mask = all_anchors.new_zeros((B, K), dtype=torch.bool)
        sample_loc_mask = all_anchors.new_zeros((B, K), dtype=torch.bool)

        anchors_cx = (all_anchors[:, 2] + all_anchors[:, 0]) / 2.0
        anchors_cy = (all_anchors[:, 3] + all_anchors[:, 1]) / 2.0
        anchor_num = anchors_cx.shape[0]
        anchor_points = torch.stack((anchors_cx, anchors_cy), dim=1)

        offsets = [x[1][0] for x in mlvl_preds]
        loc_pred = [x[1][1] for x in mlvl_preds]
        offsets = torch.cat(offsets, dim=1)
        loc_pred = torch.cat(loc_pred, dim=1)
        loc_pred = torch.stack(
            [self.get_regression_boxes(all_anchors, offsets[i], loc_pred[i]) for i in range(B)],
            dim=0)

        for b_ix in range(B):
            gt, _ = filter_by_size(gt_bboxes[b_ix], min_size=1)
            num_gt = gt.shape[0]
            if gt.shape[0] == 0:
                cls_target[b_ix][:] = neg_targets[b_ix]
                continue
            else:
                bbox = gt[:, :4]
                labels = gt[:, 4]
                ious = bbox_iou_overlaps(all_anchors, gt)
                gt_cx = (bbox[:, 2] + bbox[:, 0]) / 2.0
                gt_cy = (bbox[:, 3] + bbox[:, 1]) / 2.0
                gt_points = torch.stack((gt_cx, gt_cy), dim=1)
                distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
                # Selecting candidates based on the center distance between anchor box and object
                candidate_idxs = []
                star_idx = 0
                for num in num_anchors_per_level:
                    end_idx = star_idx + num
                    distances_per_level = distances[star_idx:end_idx]
                    topk = min(self.top_n, num)
                    _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)
                    candidate_idxs.append(topk_idxs_per_level + star_idx)
                    star_idx = end_idx
                candidate_idxs = torch.cat(candidate_idxs, dim=0)

                # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
                candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
                iou_mean_per_gt = candidate_ious.mean(0)
                iou_std_per_gt = candidate_ious.std(0)
                iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
                is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

                # Limiting the final positive samples’ center to object
                for ng in range(num_gt):
                    candidate_idxs[:, ng] += ng * anchor_num
                e_anchors_cx = anchors_cx.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
                e_anchors_cy = anchors_cy.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
                candidate_idxs = candidate_idxs.view(-1)
                _l = e_anchors_cx[candidate_idxs].view(-1, num_gt) - bbox[:, 0]
                t = e_anchors_cy[candidate_idxs].view(-1, num_gt) - bbox[:, 1]
                r = bbox[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
                b = bbox[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt)
                is_in_gts = torch.stack([_l, t, r, b], dim=1).min(dim=1)[0] > 0.01
                is_pos = is_pos & is_in_gts

                # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
                ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
                index = candidate_idxs.view(-1)[is_pos.view(-1)]
                ious_inf[index] = ious.t().contiguous().view(-1)[index]
                ious_inf = ious_inf.view(num_gt, -1).t()

                anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)

                labels = labels[anchors_to_gt_indexs]
                neg_index = anchors_to_gt_values == -INF
                pos_index = ~neg_index
                labels[anchors_to_gt_values == -INF] = neg_targets[b_ix]
                matched_gts = bbox[anchors_to_gt_indexs]

                reg_targets_per_im = matched_gts
                cls_target[b_ix] = labels
                loc_target[b_ix] = reg_targets_per_im
                sample_cls_mask[b_ix] = pos_index
                sample_loc_mask[b_ix] = pos_index

            if ignore_regions[b_ix] is not None and ignore_regions[b_ix].shape[0] > 0:
                ig_bbox = ignore_regions[b_ix]
                if ig_bbox.sum() > 0:
                    ig_left = anchors_cx[:, None] - ig_bbox[..., 0]
                    ig_right = ig_bbox[..., 2] - anchors_cx[:, None]
                    ig_top = anchors_cy[:, None] - ig_bbox[..., 1]
                    ig_bottom = ig_bbox[..., 3] - anchors_cy[:, None]
                    ig_targets = torch.stack((ig_left, ig_top, ig_right, ig_bottom), -1)
                    ig_inside_bbox_mask = (ig_targets.min(-1)[0] > 0).max(-1)[0]
                    cls_target[b_ix][ig_inside_bbox_mask] = -1
                    sample_cls_mask[b_ix][ig_inside_bbox_mask] = False
                    sample_loc_mask[b_ix][ig_inside_bbox_mask] = False
        if self.use_centerness or self.use_iou:
            batch_anchor = all_anchors.view(1, -1, 4).expand((sample_loc_mask.shape[0], sample_loc_mask.shape[1], 4))
            sample_anchor = batch_anchor[sample_loc_mask].contiguous().view(-1, 4)
            sample_loc_target = loc_target[sample_loc_mask].contiguous().view(-1, 4)
            if sample_loc_target.numel():
                if self.use_iou:
                    centerness_target = self.compute_iou_targets(sample_loc_target, sample_anchor, loc_pred[sample_loc_mask].view(-1, 4))  # noqa
                else:
                    centerness_target = self.compute_centerness_targets(
                        sample_loc_target, sample_anchor)
            else:
                centerness_target = sample_loc_target.new_zeros(sample_loc_target.shape[0])
            return cls_target, loc_target, centerness_target, sample_cls_mask, sample_loc_mask

        if self.return_gts:
            return cls_target, loc_target, sample_cls_mask, sample_loc_mask, gt_bboxes
        return cls_target, loc_target, sample_cls_mask, sample_loc_mask


@ROI_PREDICTOR_REGISTRY.register('atss_point')
class AtssPointPredictor(object):
    def __init__(self, pre_nms_score_thresh, pre_nms_top_n, post_nms_top_n, roi_min_size, merger=None, nms=None,
                 num_point=16):
        self.pre_nms_score_thresh = pre_nms_score_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_cfg = nms
        self.roi_min_size = roi_min_size
        self.num_point = num_point
        if merger is not None:
            self.merger = build_merger(merger)
        else:
            self.merger = None

    @torch.no_grad()
    @to_float32
    def predict(self, mlvl_anchors, mlvl_preds, image_info):
        mlvl_resutls = []
        for anchors, preds in zip(mlvl_anchors, mlvl_preds):
            results = self.single_level_predict(anchors, preds, image_info)
            mlvl_resutls.append(results)
        if len(mlvl_resutls) > 0:
            results = torch.cat(mlvl_resutls, dim=0)
        else:
            results = mlvl_anchors[0].new_zeros((1, 7))
        if self.merger is not None:
            results = self.merger.merge(results)
        return {'dt_bboxes': results}

    def get_regression_boxes(self, rois, offset_pred, loc_pred):
        offset = offset_pred.reshape(rois.shape[0], -1, 2)
        centers_x = (rois[:, 0] + rois[:, 2]) / 2
        centers_y = (rois[:, 1] + rois[:, 3]) / 2
        w_, h_ = (rois[:, 2] - rois[:, 0] + 1), (rois[:, 3] - rois[:, 1] + 1)
        centers = torch.stack((centers_x, centers_y), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
        wh_ = torch.stack((w_, h_), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
        offset_point = centers + offset * wh_ * 0.1  # N_batch * num_point * 2
        # loc_pred = loc_pred.reshape(rois.shape[0], self.num_point, 2)
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

    def regression(self, anchors, preds, image_info):
        cls_pred, loc_pred = preds[:2]
        offset, loc_pred = loc_pred
        B, K = cls_pred.shape[:2]
        concat_anchors = torch.stack([anchors.clone() for _ in range(B)])
        rois = self.get_regression_boxes(
            concat_anchors.reshape(B * K, 4), offset.reshape(B * K, self.num_point, 2),
            loc_pred.view(B * K, self.num_point, 2)).view(B, K, 4)
        return rois

    def single_level_predict(self, anchors, preds, image_info):
        rois = self.regression(anchors, preds, image_info)

        cls_pred = preds[0]
        B, K, C = cls_pred.shape
        roi_min_size = self.roi_min_size
        pre_nms_top_n = self.pre_nms_top_n
        pre_nms_top_n = pre_nms_top_n if pre_nms_top_n > 0 else K
        post_nms_top_n = self.post_nms_top_n
        # if featmap size is too large, filter by score thresh to reduce computation
        if K > 120:
            score_thresh = self.pre_nms_score_thresh
        else:
            score_thresh = 0

        batch_rois = []
        for b_ix in range(B):
            # clip rois and filter rois which are too small
            image_rois = rois[b_ix]
            image_rois = clip_bbox(image_rois, image_info[b_ix])
            image_rois, filter_inds = filter_by_size(image_rois, roi_min_size)
            image_cls_pred = cls_pred[b_ix][filter_inds]
            if image_rois.numel() == 0:
                continue  # noqa E701

            for cls in range(C):
                cls_rois = image_rois
                scores = image_cls_pred[:, cls]
                assert not torch.isnan(scores).any()
                if score_thresh > 0:
                    # to reduce computation
                    keep_idx = torch.nonzero(scores > score_thresh).reshape(-1)
                    if keep_idx.numel() == 0:
                        continue  # noqa E701
                    cls_rois = cls_rois[keep_idx]
                    scores = scores[keep_idx]

                # do nms per image, only one class
                _pre_nms_top_n = min(pre_nms_top_n, scores.shape[0])
                scores, order = scores.topk(_pre_nms_top_n, sorted=True)
                cls_rois = cls_rois[order, :]
                cls_rois = torch.cat([cls_rois, scores[:, None]], dim=1)

                if self.nms_cfg is not None:
                    cls_rois, keep_idx = E.nms(cls_rois, self.nms_cfg)
                if post_nms_top_n > 0:
                    cls_rois = cls_rois[:post_nms_top_n]

                ix = cls_rois.new_full((cls_rois.shape[0], 1), b_ix)
                c = cls_rois.new_full((cls_rois.shape[0], 1), cls + 1)
                cls_rois = torch.cat([ix, cls_rois, c], dim=1)
                batch_rois.append(cls_rois)

        if len(batch_rois) == 0:
            return anchors.new_zeros((1, 7))
        return torch.cat(batch_rois, dim=0)
