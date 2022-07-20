# modified from https://github.com/ModelTC/United-Perception
import cv2
import numpy as np
import torch
from torch.nn.modules.utils import _pair
from shapely.geometry import Polygon
from pycocotools import mask as mask_utils
import functools

from up.tasks.det.models.utils.bbox_helper import clip_bbox, filter_by_size
from up.tasks.det.models.utils.box_sampler import build_roi_sampler
from up.tasks.det.models.utils.matcher import build_matcher
from up.utils.general.fp16_helper import to_float32
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.registry_factory import MASK_PREDICTOR_REGISTRY, MASK_SUPERVISOR_REGISTRY
from torch.nn import functional as F

cv2.ocl.setUseOpenCL(False)

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit


def poly_to_mask(polygons, height, width):
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle).astype(np.bool)
    mask = torch.from_numpy(mask)
    return mask


@MASK_SUPERVISOR_REGISTRY.register('mask_point')
class MaskSupervisor(object):
    def __init__(self, matcher, sampler, resample=False, num_point=64):
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
        gt_masks = input['gt_masks']
        image_info = input['image_info']

        B = len(gt_bboxes)
        assert len(gt_bboxes) > 0, "gt_bboxes is empty"

        batch_rois = []
        batch_cls_target = []
        batch_mask_target = []
        batch_gt_target = []
        for b_ix in range(B):
            rois = proposals[proposals[:, 0] == b_ix]
            if rois.numel() > 0:
                rois = rois[:, 1:1 + 4]
            else:
                rois = rois.view(-1, 4)

            # filter bboxes and masks which are too small
            _gt_boxes, filter_inds = filter_by_size(gt_bboxes[b_ix], min_size=1)
            if _gt_boxes.numel() == 0:
                continue
            filter_inds = filter_inds.nonzero().squeeze(1).cpu().numpy()
            _gt_masks = np.array(gt_masks[b_ix])[filter_inds]  # list of polygons, can not be a tensor
            # add gt_bboxes into rois
            rois = torch.cat([rois, _gt_boxes[:, :4]], dim=0)

            # clip bboxes which are out of bounds
            rois = clip_bbox(rois, image_info[b_ix])

            # resample or not, if use ohem loss, supposed to be True
            if self.resample:
                rois_target_gt, overlaps = self.matcher.match(rois, _gt_boxes, return_max_overlaps=True)
                pos_inds, _ = self.sampler.sample(rois_target_gt, overlaps=overlaps)
                pos_target_gt = rois_target_gt[pos_inds]
            else:
                if not sample_record[b_ix]:
                    continue
                pos_inds, pos_target_gt = sample_record[b_ix][:2]
            if pos_inds.numel() == 0:
                continue

            # acquire target cls and target mask for sampled rois
            pos_rois = rois[pos_inds]
            pos_cls_target = _gt_boxes[pos_target_gt, 4]
            pos_gt_target = _gt_boxes[pos_target_gt, :4]

            pos_target_gt = pos_target_gt.cpu().numpy()
            mask_contour_target = _gt_masks[pos_target_gt]
            mask_contour_target = torch.from_numpy(mask_contour_target).to(_gt_boxes)

            ix = pos_rois.new_full((pos_rois.shape[0], 1), b_ix)
            pos_rois = torch.cat([ix, pos_rois], dim=1)
            batch_rois.append(pos_rois)
            batch_cls_target.append(pos_cls_target)
            batch_mask_target.append(mask_contour_target)
            batch_gt_target.append(pos_gt_target)

        if len(batch_rois) == 0:
            logger.warning('no positive sampled for mask')
            rois = proposals.new_zeros((1, 5))
            mask_target = rois.new_full((1, self.num_point*2), -1)
            gt_target = rois.new_zeros((1, 4))
        else:
            rois = torch.cat(batch_rois, dim=0)
            mask_target = torch.cat(batch_mask_target, dim=0)
            gt_target = torch.cat(batch_gt_target, dim=0)

        return rois, mask_target, gt_target

    def mask_to_poly(self, mask):
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            contour = contour.flatten().tolist()
            if len(contour) > 4:
                polygons.append(contour)
        return polygons


@MASK_PREDICTOR_REGISTRY.register('mask_point')
class MaskPointPredictor(object):
    def __init__(self, num_classes, mask_thresh):
        self.num_classes = num_classes
        self.mask_thresh = mask_thresh
        self.counter = 0

    @torch.no_grad()
    @to_float32
    def predict(self, contour_pred, rois, input):
        image_info = input['image_info']
        R = len(rois)
        B = len(image_info)

        rois = rois.detach().cpu().numpy()

        masks = [None] * R
        for b_ix in range(B):
            scale_h, scale_w = _pair(image_info[b_ix][2])
            img_h, img_w = map(int, image_info[b_ix][3:5])
            keep_inds = np.where(rois[:, 0] == b_ix)[0]
            img_rois = rois[keep_inds].copy()
            contours = contour_pred[keep_inds]
            img_rois[:, 1] /= scale_w
            img_rois[:, 2] /= scale_h
            img_rois[:, 3] /= scale_w
            img_rois[:, 4] /= scale_h
            contours[..., 0] /= scale_w
            contours[..., 1] /= scale_h

            img_masks = self.get_seg_masks(contours, img_rois, img_h, img_w)
            for idx in range(len(img_rois)):
                masks[keep_inds[idx]] = img_masks[idx]

        return {'dt_masks': masks}

    def get_seg_masks(self, det_pts, det_bboxes, img_h, img_w):
        bboxes = det_bboxes

        im_masks = []
        for i in range(bboxes.shape[0]):
            det_pt = det_pts[i]
            im_mask = poly_to_mask([det_pt.reshape(-1).detach().cpu().numpy()], img_h, img_w)
            im_masks.append(im_mask)
        return im_masks


class MaskContourHelper(object):
    # partially borrowed from https://github.com/Duankaiwen/LSNet
    def __init__(self, spline_num=10, num_contour_points=32):
        self.spline_num = spline_num
        self.num_points = num_contour_points
        self.spline_poly_num = self.num_points * self.spline_num

    def process(self, polygons, gt_bbox):
        polygons = [np.array(p).reshape(-1, 2) for p in polygons]
        filtered_polygons = self.filter_tiny_polys(polygons)
        if len(filtered_polygons) == 0:
            xmin, ymin, xmax, ymax = gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]
            tl = np.stack([xmin, ymin])
            bl = np.stack([xmin, ymax])
            br = np.stack([xmax, ymax])
            tr = np.stack([xmax, ymin])
            filtered_polygons = [np.stack([tl, bl, br, tr])]

        valid_polygons = []
        for polygon in filtered_polygons:
            sampled_polygon = self.uniformsample(polygon, self.spline_poly_num)
            tt_idx = np.argmin(np.power(sampled_polygon - sampled_polygon[0], 2).sum(axis=1))
            valid_polygon = np.roll(sampled_polygon, -tt_idx, axis=0)[::self.spline_num]
            cw_valid_polygon = self.get_cw_poly(valid_polygon)
            unify_origin_polygon = self.unify_origin_polygon(cw_valid_polygon).reshape(-1)
            unify_origin_polygon = self.sort_polygon(unify_origin_polygon)[::-1]
            valid_polygons.append(unify_origin_polygon.reshape(-1))
        return valid_polygons

    def sort_polygon(self, polygon):
        polygon = polygon.reshape(-1, 2)
        xs = polygon[:, 0]
        ys = polygon[:, 1]
        center = [xs.mean(), ys.mean()]
        ref_vec = [-1, 0]
        sort_func = functools.partial(self.clockwiseangle_and_distance, origin=center, ref_vec=ref_vec)
        sorted_polygon = sorted(polygon.tolist(), key=sort_func)
        return np.array(sorted_polygon)

    def clockwiseangle_and_distance(self, point, origin=[0, 0], ref_vec=[1, 0]):
        import math
        vector = [point[0] - origin[0], point[1] - origin[1]]
        lenvector = math.hypot(vector[0], vector[1])
        if lenvector == 0:
            return -math.pi, 0
        normalized = [vector[0] / lenvector, vector[1] / lenvector]
        dotprod = normalized[0] * ref_vec[0] + normalized[1] * ref_vec[1]  # x1*x2 + y1*y2
        diffprod = ref_vec[1] * normalized[0] - ref_vec[0] * normalized[1]  # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        if angle < 0:
            return 2 * math.pi + angle, lenvector
        return angle, lenvector

    def polygons2bbox(self, polygons, bbox_cls=None):
        bboxes = []
        for poly in polygons:
            x_min = poly[0::2].min()
            x_max = poly[0::2].max()
            y_min = poly[1::2].min()
            y_max = poly[1::2].max()
            if bbox_cls is None:
                bboxes.append([x_min, y_min, x_max, y_max])
            else:
                bboxes.append([x_min, y_min, x_max, y_max, bbox_cls])
        return bboxes

    def _polygon_area(self, poly):
        # borrowed from https://github.com/open-mmlab/mmdetection
        """ Compute the area of a component of a polygon.
        Using the shoelace formula:
        https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
        Args:
            x (ndarray): x coordinates of the component
            y (ndarray): y coordinates of the component
        Return:
            float: the are of the component
        """
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * np.abs(
            np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def filter_tiny_polys(self, polys):
        polys_ = []
        for poly in polys:
            x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
            x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
            if x_max - x_min >= 1 and y_max - y_min >= 1:
                polys_.append(poly)
        return [poly for poly in polys_ if self._polygon_area(poly) > 5]

    def get_cw_poly(self, poly):
        return poly[::-1] if Polygon(poly).exterior.is_ccw else poly

    def unify_origin_polygon(self, poly):
        new_poly = np.zeros_like(poly)
        xmin = poly[:, 0].min()
        xmax = poly[:, 0].max()
        ymin = poly[:, 1].min()
        ymax = poly[:, 1].max()
        tcx = (xmin + xmax) / 2
        tcy = ymin
        dist = (poly[:, 0] - tcx) ** 2 + (poly[:, 1] - tcy) ** 2
        min_dist_idx = dist.argmin()
        new_poly[:(poly.shape[0] - min_dist_idx)] = poly[min_dist_idx:]
        new_poly[(poly.shape[0] - min_dist_idx):] = poly[:min_dist_idx]
        return new_poly

    def uniformsample(self, pgtnp_px2, newpnum):  
        # borrowed from https://github.com/zju3dv/snake
        pnum, cnum = pgtnp_px2.shape
        assert cnum == 2

        idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
        pgtnext_px2 = pgtnp_px2[idxnext_p]
        edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
        edgeidxsort_p = np.argsort(edgelen_p)

        # two cases
        # we need to remove gt points
        # we simply remove shortest paths
        if pnum > newpnum:
            edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
            edgeidxsort_k = np.sort(edgeidxkeep_k)
            pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
            assert pgtnp_kx2.shape[0] == newpnum
            return pgtnp_kx2
        # we need to add gt points
        # we simply add it uniformly
        else:
            edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
            for i in range(pnum):
                if edgenum[i] == 0:
                    edgenum[i] = 1

            # after round, it may has 1 or 2 mismatch
            edgenumsum = np.sum(edgenum)
            if edgenumsum != newpnum:

                if edgenumsum > newpnum:

                    id = -1
                    passnum = edgenumsum - newpnum
                    while passnum > 0:
                        edgeid = edgeidxsort_p[id]
                        if edgenum[edgeid] > passnum:
                            edgenum[edgeid] -= passnum
                            passnum -= passnum
                        else:
                            passnum -= edgenum[edgeid] - 1
                            edgenum[edgeid] -= edgenum[edgeid] - 1
                            id -= 1
                else:
                    id = -1
                    edgeid = edgeidxsort_p[id]
                    edgenum[edgeid] += newpnum - edgenumsum

            assert np.sum(edgenum) == newpnum

            psample = []
            for i in range(pnum):
                pb_1x2 = pgtnp_px2[i:i + 1]
                pe_1x2 = pgtnext_px2[i:i + 1]

                pnewnum = edgenum[i]
                wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

                pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
                psample.append(pmids)

            psamplenp = np.concatenate(psample, axis=0)
            return psamplenp


def build_mask_supervisor(supervisor_cfg):
    return MASK_SUPERVISOR_REGISTRY.build(supervisor_cfg)


def build_mask_predictor(predictor_cfg):
    return MASK_PREDICTOR_REGISTRY.build(predictor_cfg)