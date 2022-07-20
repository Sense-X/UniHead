# modified from https://github.com/ModelTC/United-Perception
import cv2
import torch

from up.tasks.det.models.utils.bbox_helper import clip_bbox, filter_by_size
from up.tasks.det.models.utils.box_sampler import build_roi_sampler
from up.tasks.det.models.utils.matcher import build_matcher
from up.utils.general.fp16_helper import to_float32
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.registry_factory import KEYPOINT_SUPERVISOR_REGISTRY, KEYPOINT_PREDICTOR_REGISTRY

cv2.ocl.setUseOpenCL(False)

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit


@KEYPOINT_SUPERVISOR_REGISTRY.register('keyp_point')
class KeypPointSupervisor(object):
    def __init__(self, matcher, sampler, resample=False, num_point=17):
        self.matcher = build_matcher(matcher)
        self.sampler = build_roi_sampler(sampler)
        self.resample = resample
        self.num_point = num_point

    @torch.no_grad()
    @to_float32
    def get_targets(self, input):
        proposals = input['dt_bboxes']
        sample_record = input['sample_record']
        gt_bboxes = input['gt_bboxes']
        gt_keyps = input['gt_keyps']
        image_info = input['image_info']

        B = len(gt_keyps)
        K = gt_keyps[0].shape[1]

        batch_rois = []
        batch_keyps_target = []
        batch_bbox_target = []

        for b_ix in range(B):
            rois = proposals[proposals[:, 0] == b_ix][:, 1:1 + 4]

            # filter bboxes and keyps which are too small
            _gt_bboxes, filter_inds = filter_by_size(gt_bboxes[b_ix], min_size=1)
            if _gt_bboxes.numel() == 0: continue
            filter_inds = filter_inds.nonzero().squeeze(1).cpu().numpy()
            _gt_keyps = gt_keyps[b_ix][filter_inds]

            # resample or not, if use ohem loss, supposed to be True
            # although with a litte difference, resample or not almost has no effect here
            if self.resample:
                keep_inds = _gt_keyps[:, :, 2].max(dim=1)[0] > 0
                _gt_bboxes = _gt_bboxes[keep_inds]
                _gt_keyps = _gt_keyps[keep_inds]

                if _gt_bboxes.shape[0] == 0: continue
                rois = torch.cat([rois, _gt_bboxes[:, :4]], dim=0)
                rois = clip_bbox(rois.floor(), image_info[b_ix])

                rois_target_gt, overlaps = self.matcher.match(rois, _gt_bboxes, return_max_overlaps=True)
                pos_inds, _ = self.sampler.sample(rois_target_gt, overlaps=overlaps)
                pos_target_gt = rois_target_gt[pos_inds]
            else:
                if not sample_record[b_ix]: continue
                pos_inds, pos_target_gt = sample_record[b_ix][:2]
                rois = torch.cat([rois, _gt_bboxes[:, :4]], dim=0)
                rois = clip_bbox(rois.floor(), image_info[b_ix])
            if pos_inds.numel() == 0: continue

            # acquire target keyps for sampled rois
            pos_rois = rois[pos_inds]
            pos_keyps_target = _gt_keyps[pos_target_gt]
            pos_bbox_target = _gt_bboxes[pos_target_gt, :4]

            ix = pos_rois.new_full((pos_rois.shape[0], 1), b_ix)
            pos_rois = torch.cat([ix, pos_rois], dim=1)
            batch_rois.append(pos_rois)
            batch_keyps_target.append(pos_keyps_target)
            batch_bbox_target.append(pos_bbox_target)

        if len(batch_rois) == 0:
            logger.warning('no positive rois found for keypoint')
            rois = proposals.new_zeros((1, 5))
            keyps_target = rois.new_full((1, K, 3), -1, dtype=torch.int64)
            bbox_target = rois.new_zeros((1, 4))
        else:
            rois = torch.cat(batch_rois, dim=0)
            keyps_target = torch.cat(batch_keyps_target, dim=0)
            bbox_target = torch.cat(batch_bbox_target, dim=0)

        return rois, keyps_target, bbox_target


@KEYPOINT_PREDICTOR_REGISTRY.register('keyp_point')
class KeypPointPredictor(object):
    def __init__(self):
        pass

    @torch.no_grad()
    @to_float32
    def predict(self, keyp_pred, keyp_vis, bboxes, input):
        pred = torch.cat((keyp_pred, keyp_vis.sigmoid().unsqueeze(-1)), dim=-1)

        return {'dt_keyps': pred}


def build_keypoint_supervisor(supervisor_cfg):
    return KEYPOINT_SUPERVISOR_REGISTRY.build(supervisor_cfg)


def build_keypoint_predictor(predictor_cfg):
    return KEYPOINT_PREDICTOR_REGISTRY.build(predictor_cfg)
