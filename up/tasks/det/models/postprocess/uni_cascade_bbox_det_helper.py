# modified from https://github.com/ModelTC/United-Perception
import copy
import torch

from up.tasks.det.models.utils.bbox_helper import clip_bbox, filter_by_size
from up.tasks.det.models.utils.box_sampler import build_roi_sampler
from up.tasks.det.models.utils.matcher import build_matcher
from up.utils.general.fp16_helper import to_float32
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.registry_factory import BBOX_SUPERVISOR_REGISTRY, BBOX_PREDICTOR_REGISTRY

from .uni_bbox_det_helper import UniheadBboxPredictor as BboxPredictor


class SingleSupervisor(object):
    def __init__(self, bbox_normalize, matcher, sampler):
        self.matcher = build_matcher(matcher)
        self.sampler = build_roi_sampler(sampler)

    @torch.no_grad()
    @to_float32
    def get_targets(self, proposals, input):
        gt_bboxes = input['gt_bboxes']
        ignore_regions = input.get('gt_ignores', None)
        image_info = input['image_info']

        T = proposals
        B = len(gt_bboxes)
        if ignore_regions is None:
            ignore_regions = [None] * B

        batch_rois = []
        batch_gt_flags = []
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
            gt_flags = rois.new_zeros((rois.shape[0], ), dtype=torch.bool)
            if gt.numel() > 0:
                rois = torch.cat([rois, gt[:, :4]], dim=0)
                gt_ones = rois.new_ones(gt.shape[0], dtype=torch.bool)
                gt_flags = torch.cat([gt_flags, gt_ones])

            # clip rois which are out of bounds
            rois = clip_bbox(rois, image_info[b_ix])

            rois_target_gt, overlaps = self.matcher.match(
                rois, gt, ignore_regions[b_ix], return_max_overlaps=True)
            pos_inds, neg_inds = self.sampler.sample(rois_target_gt, overlaps=overlaps)
            P = pos_inds.numel()
            N = neg_inds.numel()
            if P + N == 0: continue  # noqa E701

            # save pos inds and related gts' inds for mask/keypoint head
            sample_record = (pos_inds, rois_target_gt[pos_inds])
            batch_sample_record[b_ix] = sample_record

            # acquire target cls and target loc for sampled rois
            pos_rois = rois[pos_inds]
            neg_rois = rois[neg_inds]
            pos_target_gt = gt[rois_target_gt[pos_inds]]
            gt_flags = gt_flags[pos_inds]
            if pos_target_gt.numel() > 0:
                pos_cls_target = pos_target_gt[:, 4].to(dtype=torch.int64)
            else:
                # give en empty tensor if no positive
                pos_cls_target = T.new_zeros((0, ), dtype=torch.int64)
            neg_cls_target = T.new_zeros((N, ), dtype=torch.int64)

            batch_pos_gts.append(pos_target_gt)  ##
            pos_loc_target = pos_target_gt[:, :4]

            rois = torch.cat([pos_rois, neg_rois], dim=0)
            # for cascade rcnn, add by buxingyuan
            gt_zeros = rois.new_zeros(neg_rois.shape[0], dtype=torch.bool)
            gt_flags = torch.cat([gt_flags, gt_zeros])
            # end
            b = T.new_full((rois.shape[0], 1), b_ix)
            rois = torch.cat([b, rois], dim=1)
            batch_rois += [rois]
            batch_gt_flags += [gt_flags]
            batch_cls_target += [pos_cls_target, neg_cls_target]
            batch_loc_target += [pos_loc_target, T.new_zeros(N, 4)]

            loc_weight = torch.cat([T.new_ones(P, 4), T.new_zeros(N, 4)])
            batch_loc_weight.append(loc_weight)

        if len(batch_rois) == 0:
            num_rois = 1
            rois = T.new_zeros((num_rois, 5))
            gt_flags = T.new_zeros(num_rois).to(dtype=torch.bool)
            cls_target = T.new_zeros(num_rois).long() - 1  # target cls must be `long` type
            loc_target = T.new_zeros((num_rois, 4))
            loc_weight = T.new_zeros((num_rois, 4))
            pos_gts = T.new_zeros((num_rois, 5))
            logger.warning('no valid proposals, set cls_target to {}'.format(cls_target))
        else:
            rois = torch.cat(batch_rois, dim=0)
            gt_flags = torch.cat(batch_gt_flags, dim=0)
            cls_target = torch.cat(batch_cls_target, dim=0).long()
            loc_target = torch.cat(batch_loc_target, dim=0)
            loc_weight = torch.cat(batch_loc_weight, dim=0)
            pos_gts = torch.cat(batch_pos_gts, dim=0)

        return batch_sample_record, rois, cls_target, loc_target, loc_weight, pos_gts, gt_flags


@BBOX_SUPERVISOR_REGISTRY.register('uni_cascade_bbox_det')
class CascadeSupervisor(object):
    def __init__(self, num_stage, stage_bbox_normalize, stage_matcher, sampler):
        self.supervisor_list = []
        for i in range(num_stage):
            stage_config = self.get_stage_config(i, stage_bbox_normalize, stage_matcher, sampler)
            supervisor = SingleSupervisor(*stage_config)
            self.supervisor_list.append(supervisor)

    def get_stage_config(self, stage, stage_bbox_normalize, stage_matcher, sampler):
        bbox_normalize = {
            'decay': stage_bbox_normalize.get('decays', [1.0, 1.0, 1.0])[stage]
        }
        matcher = copy.deepcopy(stage_matcher)
        kwargs = matcher['kwargs']
        for k in ['positive_iou_thresh', 'negative_iou_thresh']:
            kwargs[k] = kwargs[k][stage]
        return bbox_normalize, matcher, sampler

    def get_targets(self, stage, proposals, input):
        return self.supervisor_list[stage].get_targets(proposals, input)


@BBOX_PREDICTOR_REGISTRY.register('uni_cascade_bbox_det')
class CascadePredictor(object):
    def __init__(self, num_stage, stage_bbox_normalize, bbox_score_thresh, top_n, share_location,
                 nms, bbox_vote=None, num_loc_point=16):
        bbox_normalize = self.get_stage_config(num_stage - 1, stage_bbox_normalize)
        self.num_loc_point = num_loc_point
        self.final_predictor = BboxPredictor(
            bbox_normalize, bbox_score_thresh, top_n, share_location, nms, self.num_loc_point,
            bbox_vote)
        self.refiner_list = []
        for i in range(num_stage - 1):
            bbox_normalize = self.get_stage_config(i, stage_bbox_normalize)
            self.refiner_list.append(Refiner(bbox_normalize, share_location, self.num_loc_point))

    def get_stage_config(self, stage, stage_bbox_normalize):
        bbox_normalize = {
            'decay': stage_bbox_normalize.get('decays', [1.0, 1.0, 1.0])[stage]
        }
        return bbox_normalize

    def refine(self, i, *args, **kwargs):
        return self.refiner_list[i].refine(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.final_predictor.predict(*args, **kwargs)


class Refiner(object):
    def __init__(self, bbox_normalize, share_location, num_point):
        self.decay = bbox_normalize.get('decay', 1.0)
        self.share_location = share_location
        self.num_point = num_point

    def get_iou_regression_boxes(self, rois, offset_pred, loc_pred):
        offset = offset_pred[:, 1:]
        offset = offset.reshape(rois.shape[0], -1, 2)
        centers_x = (rois[:, 0] + rois[:, 2]) / 2
        centers_y = (rois[:, 1] + rois[:, 3]) / 2
        w_, h_ = (rois[:, 2] - rois[:, 0] + 1), (rois[:, 3] - rois[:, 1] + 1)
        centers = torch.stack((centers_x, centers_y), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
        wh_ = torch.stack((w_, h_), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
        offset_point = centers + offset * wh_ * 0.1  # N_batch * num_point * 2
        loc_pred = loc_pred.reshape(rois.shape[0], self.num_point, 2)
        x_shift = loc_pred[..., 0] * w_.unsqueeze(1) * 0.5 * self.decay
        y_shift = loc_pred[..., 1] * h_.unsqueeze(1) * 0.5 * self.decay

        shifts = torch.stack((x_shift, y_shift), dim=-1)  # N_batch * num_point * 2
        shifted_offset_point = offset_point + shifts

        x_min, _ = torch.min(shifted_offset_point[..., 0], dim=-1)
        y_min, _ = torch.min(shifted_offset_point[..., 1], dim=-1)
        x_max, _ = torch.max(shifted_offset_point[..., 0], dim=-1)
        y_max, _ = torch.max(shifted_offset_point[..., 1], dim=-1)
        iou_boxes = torch.stack((x_min, y_min, x_max, y_max), dim=-1)

        return iou_boxes

    def refine(self, rois, labels, loc_pred, image_info, gt_flags=None):
        B = len(image_info)
        (loc_offset_pred, loc_pred) = loc_pred

        # double check loc_pred shape because loc_pred is reduced to [N, 4] in get loss func
        if self.share_location or loc_pred.shape[1] == self.num_point * 2:
            loc_pred = loc_pred
        else:
            N = loc_pred.shape[0]
            inds = torch.arange(N, dtype=torch.int64, device=loc_pred.device)
            loc_pred = loc_pred.reshape(N, -1, self.num_point * 2)
            loc_pred = loc_pred[inds, labels.long()].reshape(-1, self.num_point*2)

        assert loc_pred.size(1) == self.num_point*2

        bboxes = self.get_iou_regression_boxes(rois[:, 1:1 + 4], loc_offset_pred, loc_pred)

        detected_bboxes = []
        for b_ix in range(B):
            rois_ix = torch.nonzero(rois[:, 0] == b_ix).reshape(-1)
            if rois_ix.numel() == 0: continue
            pre_bboxes = bboxes[rois_ix]

            # clip bboxes which are out of bounds
            pre_bboxes[:, :4] = clip_bbox(pre_bboxes[:, :4], image_info[b_ix])

            ix = pre_bboxes.new_full((pre_bboxes.shape[0], 1), b_ix)
            post_bboxes = torch.cat([ix, pre_bboxes], dim=1)
            detected_bboxes.append(post_bboxes)
        detected_bboxes = torch.cat(detected_bboxes, dim=0)

        if gt_flags is not None:
            # filter gt bboxes
            assert gt_flags.dtype == torch.bool, gt_flags.dtype
            pos_keep = ~gt_flags
            keep_inds = torch.nonzero(pos_keep).reshape(-1)
            new_rois = detected_bboxes[keep_inds]
            return new_rois
        else:
            return detected_bboxes
