# modified from https://github.com/ModelTC/United-Perception
import torch

from up.extensions import nms
from up.tasks.det.models.utils.bbox_helper import (
    box_voting, clip_bbox, filter_by_size)
from up.tasks.det.models.utils.box_sampler import build_roi_sampler
from up.tasks.det.models.utils.matcher import build_matcher
from up.utils.general.fp16_helper import to_float32
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.registry_factory import BBOX_SUPERVISOR_REGISTRY, BBOX_PREDICTOR_REGISTRY


@BBOX_SUPERVISOR_REGISTRY.register('uni_bbox_det')
class UniheadBboxSupervisor(object):
    def __init__(self, matcher, sampler, bbox_encode=False):
        self.matcher = build_matcher(matcher)
        self.sampler = build_roi_sampler(sampler)
        self.bbox_encode = bbox_encode

    @torch.no_grad()
    @to_float32
    def get_targets(self, input):
        proposals = input['dt_bboxes']
        gt_bboxes = input['gt_bboxes']
        ignore_regions = input.get('gt_ignores', None)
        image_info = input['image_info']

        T = proposals
        B = len(gt_bboxes)
        if ignore_regions is None:
            ignore_regions = [None] * B

        batch_rois = []
        batch_cls_target = []
        batch_loc_target = []
        batch_loc_weight = []
        batch_pos_gts = []
        batch_sample_record = [None] * B
        for b_ix in range(B):
            rois = proposals[proposals[:, 0] == b_ix]
            if rois.numel() > 0:
                # remove batch idx, score & label
                rois = rois[:, 1:1 + 4]
            else:
                rois = rois.view(-1, 4)

            # filter gt_bboxes which are too small
            gt, _ = filter_by_size(gt_bboxes[b_ix], min_size=1)

            # add gts into rois
            if gt.numel() > 0:
                rois = torch.cat([rois, gt[:, :4]], dim=0)

            # clip rois which are out of bounds
            rois = clip_bbox(rois, image_info[b_ix])

            rois_target_gt, overlaps = self.matcher.match(
                rois, gt, ignore_regions[b_ix], return_max_overlaps=True)
            pos_inds, neg_inds = self.sampler.sample(rois_target_gt, overlaps=overlaps)
            P = pos_inds.numel()
            N = neg_inds.numel()
            if P + N == 0: continue  # noqa E701

            # acquire target cls and target loc for sampled rois
            pos_rois = rois[pos_inds]
            neg_rois = rois[neg_inds]
            pos_target_gt = gt[rois_target_gt[pos_inds]]
            if pos_target_gt.numel() > 0:
                pos_cls_target = pos_target_gt[:, 4].to(dtype=torch.int64)
            else:
                # give en empty tensor if no positive
                pos_cls_target = T.new_zeros((0,), dtype=torch.int64)
            sample_record = (pos_inds, rois_target_gt[pos_inds], pos_rois, pos_cls_target)
            batch_pos_gts.append(pos_target_gt)

            batch_sample_record[b_ix] = sample_record
            neg_cls_target = T.new_zeros((N,), dtype=torch.int64)

            pos_loc_target = pos_target_gt[:, :4]

            rois = torch.cat([pos_rois, neg_rois], dim=0)
            b = T.new_full((rois.shape[0], 1), b_ix)
            rois = torch.cat([b, rois], dim=1)
            batch_rois += [rois]
            batch_cls_target += [pos_cls_target, neg_cls_target]
            batch_loc_target += [pos_loc_target, T.new_zeros(N, 4)]

            loc_weight = torch.cat([T.new_ones(P, 4), T.new_zeros(N, 4)])
            batch_loc_weight.append(loc_weight)

        if len(batch_rois) == 0:
            num_rois = 1
            rois = T.new_zeros((num_rois, 5))
            cls_target = T.new_zeros(num_rois).long() - 1  # target cls must be `long` type
            loc_target = T.new_zeros((num_rois, 4))
            loc_weight = T.new_zeros((num_rois, 4))
            pos_gts = T.new_zeros((num_rois, 5))
            logger.warning('no valid proposals, set cls_target to {}'.format(cls_target))
        else:
            rois = torch.cat(batch_rois, dim=0)
            cls_target = torch.cat(batch_cls_target, dim=0).long()
            loc_target = torch.cat(batch_loc_target, dim=0)
            loc_weight = torch.cat(batch_loc_weight, dim=0)
            pos_gts = torch.cat(batch_pos_gts, dim=0)

        return batch_sample_record, rois, cls_target, loc_target, loc_weight, pos_gts


class BaseBboxPredictor(object):
    def __init__(self, bbox_normalize, bbox_score_thresh, top_n, share_location,
                 nms, bbox_vote=None, use_l1_box=True):
        self.bbox_score_thresh = bbox_score_thresh
        self.top_n = top_n
        self.share_location = share_location
        self.nms_cfg = nms
        self.bbox_vote_cfg = bbox_vote
        self.use_l1_box = use_l1_box

    def get_l1_regression_boxes(self, rois, offset_pred, loc_pred):
        offset = offset_pred[:, 1:]
        offset = offset.reshape(rois.shape[0], -1, 2)
        centers_x = (rois[:, 0] + rois[:, 2]) / 2
        centers_y = (rois[:, 1] + rois[:, 3]) / 2
        w_, h_ = (rois[:, 2] - rois[:, 0] + 1), (rois[:, 3] - rois[:, 1] + 1)
        centers = torch.stack((centers_x, centers_y), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
        wh_ = torch.stack((w_, h_), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
        offset_point = centers + offset * wh_ * 0.1  # N_batch * 4 * 2

        loc_pred = loc_pred.reshape(loc_pred.shape[0], -1, 4)
        l, t, r, b = loc_pred[..., 0], loc_pred[..., 1], loc_pred[..., 2], loc_pred[..., 3]
        w_ = w_.reshape(-1, 1)
        h_ = h_.reshape(-1, 1)
        offset_point = offset_point.unsqueeze(1)
        x0 = offset_point[:, :, 0, 0] - l * w_ * 0.5
        y0 = offset_point[:, :, 1, 1] - t * h_ * 0.5
        x1 = offset_point[:, :, 2, 0] + r * w_ * 0.5
        y1 = offset_point[:, :, 3, 1] + b * h_ * 0.5
        boxes = torch.stack((x0, y0, x1, y1), dim=-1)
        return boxes

    def get_iou_regression_boxes(self, rois, offset_pred, loc_pred):
        offset = offset_pred[:, 1:]
        offset = offset.reshape(rois.shape[0], -1, 2)
        centers_x = (rois[:, 0] + rois[:, 2]) / 2
        centers_y = (rois[:, 1] + rois[:, 3]) / 2
        w_, h_ = (rois[:, 2] - rois[:, 0] + 1), (rois[:, 3] - rois[:, 1] + 1)
        centers = torch.stack((centers_x, centers_y), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
        wh_ = torch.stack((w_, h_), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
        offset_point = centers + offset * wh_ * 0.1  # N_batch * num_point * 2

        loc_pred = loc_pred.reshape(loc_pred.shape[0], -1, self.num_loc_point, 2)
        w_ = w_.reshape(-1, 1, 1)
        h_ = h_.reshape(-1, 1, 1)
        offset_point = offset_point.unsqueeze(1)  # N_batch * 1 * num_point * 2
        x_shift = loc_pred[..., 0] * w_ * 0.5 * self.decay
        y_shift = loc_pred[..., 1] * h_ * 0.5 * self.decay

        shifts = torch.stack((x_shift, y_shift), dim=-1)  # N_batch * num_cls * num_point * 2
        shifted_offset_point = offset_point + shifts

        x_min, _ = torch.min(shifted_offset_point[..., 0], dim=-1)
        y_min, _ = torch.min(shifted_offset_point[..., 1], dim=-1)
        x_max, _ = torch.max(shifted_offset_point[..., 0], dim=-1)
        y_max, _ = torch.max(shifted_offset_point[..., 1], dim=-1)
        iou_boxes = torch.stack((x_min, y_min, x_max, y_max), dim=-1)
        return iou_boxes
    
    @torch.no_grad()
    @to_float32
    def get_bboxes(self, proposals, preds, image_info, start_idx):
        cls_pred, (loc_offset_pred, loc_pred) = preds

        # acquire all predicted bboxes
        B, (R, C) = len(image_info), cls_pred.shape
        if self.use_l1_box:
            bboxes = self.get_l1_regression_boxes(proposals[:, 1:1 + 4], loc_offset_pred, loc_pred.view(R, -1))
        else:
            bboxes = self.get_iou_regression_boxes(proposals[:, 1:1 + 4], loc_offset_pred, loc_pred.view(R, -1))
        bboxes = torch.cat([bboxes for _ in range(C)], dim=1) if self.share_location else bboxes
        bboxes = torch.cat([bboxes.view(-1, 4), cls_pred.view(-1, 1)], dim=1).view(R, -1)

        detected_bboxes = []
        for b_ix in range(B):
            img_inds = torch.nonzero(proposals[:, 0] == b_ix).reshape(-1)
            if len(img_inds) == 0: continue  # noqa E701
            img_bboxes = bboxes[img_inds]
            img_scores = cls_pred[img_inds]

            # clip bboxes which are out of bounds
            img_bboxes = img_bboxes.view(-1, 5)
            img_bboxes[:, :4] = clip_bbox(img_bboxes[:, :4], image_info[b_ix])
            img_bboxes = img_bboxes.view(-1, C * 5)

            inds_all = img_scores > self.bbox_score_thresh
            result_bboxes = []
            nms_cfg = self.nms_cfg.copy()
            for cls in range(start_idx, C):
                # keep well-predicted bbox
                inds_cls = inds_all[:, cls].nonzero().reshape(-1)
                img_bboxes_cls = img_bboxes[inds_cls, cls * 5:(cls + 1) * 5]
                if len(img_bboxes_cls) == 0: continue  # noqa E701

                # do nms, can choose the nms type, naive or softnms
                order = img_bboxes_cls[:, 4].sort(descending=True)[1]
                img_bboxes_cls = img_bboxes_cls[order]

                img_bboxes_cls_nms, keep_inds = nms(img_bboxes_cls, nms_cfg)
                ix = img_bboxes_cls.new_full((img_bboxes_cls_nms.shape[0], 1), b_ix)
                c = img_bboxes_cls.new_full((img_bboxes_cls_nms.shape[0], 1), (cls + 1 - start_idx))
                result_bboxes.append(torch.cat([ix, img_bboxes_cls_nms, c], dim=1))
            if len(result_bboxes) == 0: continue  # noqa E701

            # keep the top_n well-predicted bboxes per image
            result_bboxes = torch.cat(result_bboxes, dim=0)
            # detected_bboxes.append(result_bboxes)

            _, keep_inds = nms(result_bboxes[:, 1:6], nms_cfg)
            detected_bboxes.append(result_bboxes[keep_inds])

        if len(detected_bboxes) == 0:
            dt_bboxes = proposals.new_zeros((1, 7))
        else:
            dt_bboxes = torch.cat(detected_bboxes, dim=0)

        return dt_bboxes

    @torch.no_grad()
    @to_float32
    def predict(self, proposals, preds, image_info, start_idx):
        cls_pred, (loc_offset_pred, loc_pred) = preds

        # acquire all predicted bboxes
        B, (R, C) = len(image_info), cls_pred.shape
        if self.use_l1_box:
            bboxes = self.get_l1_regression_boxes(proposals[:, 1:1+4], loc_offset_pred, loc_pred.view(R, -1))
        else:
            bboxes = self.get_iou_regression_boxes(proposals[:, 1:1+4], loc_offset_pred, loc_pred.view(R, -1))
        bboxes = torch.cat([bboxes for _ in range(C)], dim=1) if self.share_location else bboxes
        bboxes = torch.cat([bboxes.view(-1, 4), cls_pred.view(-1, 1)], dim=1).view(R, -1)

        detected_bboxes = []
        for b_ix in range(B):
            img_inds = torch.nonzero(proposals[:, 0] == b_ix).reshape(-1)
            if len(img_inds) == 0: continue  # noqa E701
            img_bboxes = bboxes[img_inds]
            img_scores = cls_pred[img_inds]

            # clip bboxes which are out of bounds
            img_bboxes = img_bboxes.view(-1, 5)
            img_bboxes[:, :4] = clip_bbox(img_bboxes[:, :4], image_info[b_ix])
            img_bboxes = img_bboxes.view(-1, C * 5)

            inds_all = img_scores > self.bbox_score_thresh
            result_bboxes = []
            for cls in range(start_idx, C):
                # keep well-predicted bbox
                inds_cls = inds_all[:, cls].nonzero().reshape(-1)
                img_bboxes_cls = img_bboxes[inds_cls, cls * 5:(cls + 1) * 5]
                if len(img_bboxes_cls) == 0: continue  # noqa E701

                # do nms, can choose the nms type, naive or softnms
                order = img_bboxes_cls[:, 4].sort(descending=True)[1]
                img_bboxes_cls = img_bboxes_cls[order]
                img_bboxes_cls_nms, keep_inds = nms(img_bboxes_cls, self.nms_cfg)

                if self.bbox_vote_cfg is not None:
                    img_bboxes_cls_nms = box_voting(img_bboxes_cls_nms,
                                                    img_bboxes_cls,
                                                    self.bbox_vote_cfg.get('vote_th', 0.9),
                                                    scoring_method=self.bbox_vote_cfg.get('scoring_method', 'id'))

                ix = img_bboxes_cls.new_full((img_bboxes_cls_nms.shape[0], 1), b_ix)
                c = img_bboxes_cls.new_full((img_bboxes_cls_nms.shape[0], 1), (cls + 1 - start_idx))
                result_bboxes.append(torch.cat([ix, img_bboxes_cls_nms, c], dim=1))
            if len(result_bboxes) == 0: continue  # noqa E701

            # keep the top_n well-predicted bboxes per image
            result_bboxes = torch.cat(result_bboxes, dim=0)
            if 0 <= self.top_n < result_bboxes.shape[0]:
                num_base = result_bboxes.shape[0] - self.top_n + 1
                thresh = torch.kthvalue(result_bboxes[:, 5].cpu(), num_base)[0]
                keep_inds = result_bboxes[:, 5] >= thresh.item()
                result_bboxes = result_bboxes[keep_inds][:self.top_n]
            detected_bboxes.append(result_bboxes)

        if len(detected_bboxes) == 0:
            dt_bboxes = proposals.new_zeros((1, 7))
        else:
            dt_bboxes = torch.cat(detected_bboxes, dim=0)

        return {'dt_bboxes': dt_bboxes}


@BBOX_PREDICTOR_REGISTRY.register('uni_bbox_det')
class UniheadBboxPredictor(BaseBboxPredictor):
    def __init__(self, bbox_normalize, bbox_score_thresh, top_n, share_location,
                 nms, num_loc_point=16, bbox_vote=None):
        super(UniheadBboxPredictor, self).__init__(
            bbox_normalize, bbox_score_thresh, top_n, share_location, nms, bbox_vote,
            use_l1_box=False)
        self.num_loc_point = num_loc_point
        self.decay = bbox_normalize.get('decay', 1.0)

    def get_iou_regression_boxes(self, rois, offset_pred, loc_pred):
        offset = offset_pred[:, 1:]
        offset = offset.reshape(rois.shape[0], -1, 2)
        centers_x = (rois[:, 0] + rois[:, 2]) / 2
        centers_y = (rois[:, 1] + rois[:, 3]) / 2
        w_, h_ = (rois[:, 2] - rois[:, 0] + 1), (rois[:, 3] - rois[:, 1] + 1)
        centers = torch.stack((centers_x, centers_y), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
        wh_ = torch.stack((w_, h_), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
        offset_point = centers + offset * wh_ * 0.1  # N_batch * num_point * 2

        loc_pred = loc_pred.reshape(loc_pred.shape[0], -1, self.num_loc_point, 2)
        w_ = w_.reshape(-1, 1, 1)
        h_ = h_.reshape(-1, 1, 1)
        offset_point = offset_point.unsqueeze(1)  # N_batch * 1 * num_point * 2
        x_shift = loc_pred[..., 0] * w_ * 0.5 * self.decay
        y_shift = loc_pred[..., 1] * h_ * 0.5 * self.decay

        shifts = torch.stack((x_shift, y_shift), dim=-1)  # N_batch * num_cls * num_point * 2
        shifted_offset_point = offset_point + shifts

        x_min, _ = torch.min(shifted_offset_point[..., 0], dim=-1)
        y_min, _ = torch.min(shifted_offset_point[..., 1], dim=-1)
        x_max, _ = torch.max(shifted_offset_point[..., 0], dim=-1)
        y_max, _ = torch.max(shifted_offset_point[..., 1], dim=-1)
        iou_boxes = torch.stack((x_min, y_min, x_max, y_max), dim=-1)

        return iou_boxes
