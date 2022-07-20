# modified from https://github.com/ModelTC/United-Perception
import torch
import torch.nn as nn

from .roi_predictor import RoiPredictor
from up.tasks.det.models.utils.matcher import build_matcher
from up.utils.general.registry_factory import (
    MATCHER_REGISTRY, ROI_SUPERVISOR_REGISTRY, ROI_PREDICTOR_REGISTRY)


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


@ROI_PREDICTOR_REGISTRY.register('fcos_point')
class FcosPointPredictor(RoiPredictor):
    def __init__(self, pre_nms_score_thresh, pre_nms_top_n, post_nms_top_n,
                 roi_min_size, merger, nms=None, all_in_one=False, base_scale=8,
                 num_point=16):

        super().__init__(pre_nms_score_thresh, pre_nms_top_n,
                         post_nms_top_n, roi_min_size, merger, nms)
        self.all_in_one = all_in_one
        self.base_scale = base_scale
        self.num_point = num_point

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

    def regression(self, locations, preds, image_info):
        offset_pred, loc_pred = preds[1]
        B = loc_pred.shape[0]
        boxes = torch.stack([
            self.get_regression_boxes(locations, offset_pred[i], loc_pred[i]) for i in range(B)],
            dim=0)
        return boxes

    def predict(self, locations, preds, image_info):
        bboxes = super().predict(locations, preds, image_info)['dt_bboxes']

        if self.all_in_one:
            bboxes[:, -1] = torch.clamp(bboxes[:, -1], max=1)
        return {'dt_bboxes': bboxes}


@ROI_SUPERVISOR_REGISTRY.register('fcos_point')
class FcosPointSupervisor(object):
    def __init__(self, matcher, norm_on_bbox=False, return_gts=False):
        self.matcher = build_matcher(matcher)
        self.norm_on_bbox = norm_on_bbox
        self.return_gts = return_gts

    def get_targets(self, locations, loc_ranges, input):
        gt_bboxes = input['gt_bboxes']
        strides = input['strides']
        igs = input.get('gt_ignores', None)
        expanded_loc_ranges = []
        num_points_per = [len(p) for p in locations]
        for i in range(len(locations)):
            expanded_loc_ranges.append(locations[i].new_tensor(loc_ranges[i])[None].expand(len(locations[i]), -1))
        points = torch.cat(locations, dim=0)
        loc_ranges = torch.cat(expanded_loc_ranges, dim=0)

        K = points.size(0)
        B = len(gt_bboxes)
        cls_target = points.new_full((B, K), -1, dtype=torch.int64)
        loc_target = points.new_zeros((B, K, 4))
        sample_cls_mask = points.new_zeros((B, K), dtype=torch.bool)
        sample_loc_mask = points.new_zeros((B, K), dtype=torch.bool)
        for b_ix in range(B):
            gt = gt_bboxes[b_ix]
            if gt.shape[0] > 0:
                ig = None
                if igs is not None:
                    ig = igs[b_ix]
                labels, bbox_targets = self.matcher.match(points, gt, loc_ranges, num_points_per, strides, ig)
                cls_target[b_ix] = labels
                loc_target[b_ix] = bbox_targets
                sample_cls_mask[b_ix] = labels != -1
                sample_loc_mask[b_ix] = labels > 0

        if self.return_gts:
            return cls_target, loc_target, sample_cls_mask, sample_loc_mask, gt_bboxes
        return cls_target, loc_target, sample_cls_mask, sample_loc_mask


@MATCHER_REGISTRY.register('fcos_point')
class FcosPointMatcher(object):
    def __init__(self, center_sample=None, pos_radius=1):
        self.center_sample = center_sample
        self.pos_radius = pos_radius

    def match(self, points, gt, loc_ranges, num_points_per,
              strides=[8, 16, 32, 64, 128], ig=None):

        INF = 1e10
        num_gts = gt.shape[0]
        K = points.shape[0]
        gt_labels = gt[:, 4]
        xs, ys = points[:, 0], points[:, 1]
        gt_bboxes = gt[:, :4]
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)

        areas = areas[None].repeat(K, 1)
        loc_ranges = loc_ranges[:, None, :].expand(K, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(K, num_gts, 4)
        gt_xs = xs[:, None].expand(K, num_gts)
        gt_ys = ys[:, None].expand(K, num_gts)

        left = gt_xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - gt_xs
        top = gt_ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - gt_ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sample:
            sample_mask = self.get_sample_region(
                gt_bboxes, strides, num_points_per, gt_xs, gt_ys)
        else:
            sample_mask = bbox_targets.min(-1)[0] > 0

        max_loc_distance = bbox_targets.max(-1)[0]
        inside_loc_range = (max_loc_distance >= loc_ranges[..., 0]) & (max_loc_distance <= loc_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[~sample_mask] = INF

        areas[inside_loc_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0
        bbox_targets = gt_bboxes[range(K), min_area_inds]

        # ignore
        if ig is not None:
            num_igs = ig.shape[0]
            ig_xs = xs[:, None].expand(K, num_igs)
            ig_ys = ys[:, None].expand(K, num_igs)
            ig_left = ig_xs - ig[..., 0]
            ig_right = ig[..., 2] - ig_xs
            ig_top = ig_ys - ig[..., 1]
            ig_bottom = ig[..., 3] - ig_ys
            ig_targets = torch.stack((ig_left, ig_top, ig_right, ig_bottom), -1)
            ig_inside_gt_bbox_mask = (ig_targets.min(-1)[0] > 0).max(-1)[0]
            labels[ig_inside_gt_bbox_mask] = -1
        return labels, bbox_targets

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys):
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].float().sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.bool)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * self.pos_radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)
            beg = end
        left = gt_xs - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs
        top = gt_ys - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask
