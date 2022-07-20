# Import from third library
import torch
from up.utils.general.global_flag import ALIGNED_FLAG


def get_rois_target_levels(levels, base_scale, rois, base_level=0, minus_min_lvl=False):
    """
    Assign proposals to different level feature map to roi pooling

    Arguments:
        rois (FloatTensor): [R, 5] (batch_ix, x1, y1, x2, y2)
        levels (list of int): [L], levels. e.g.[2, 3, 4, 5, 6]
        base_scale: scale of the minimum level
    """
    w = rois[:, 3] - rois[:, 1] + ALIGNED_FLAG.offset
    h = rois[:, 4] - rois[:, 2] + ALIGNED_FLAG.offset
    scale = (w * h)**0.5
    eps = 1e-6
    target_levels = (base_level + (scale / base_scale + eps).log2()).floor()
    target_levels = target_levels.to(dtype=torch.int64)
    min_level, max_level = min(levels), max(levels)
    minus_level = 0
    if minus_min_lvl:
        minus_level = min_level
    return torch.clamp(target_levels, min=min_level, max=max_level) - minus_level


def map_rois_to_level(levels, base_scale, rois, base_level=0, minus_min_lvl=False):
    target_lvls = get_rois_target_levels(levels, base_scale, rois, base_level, minus_min_lvl)
    rois_by_level, rois_ix_by_level = [], []
    if minus_min_lvl:
        min_lvl = min(levels)
        levels = [lvl - min_lvl for lvl in levels]
    for lvl in levels:
        ix = torch.nonzero(target_lvls == lvl).reshape(-1)
        rois_by_level.append(rois[ix])
        rois_ix_by_level.append(ix)
    map_from_inds = torch.cat(rois_ix_by_level)
    map_back_inds = torch.zeros((rois.shape[0], ), dtype=torch.int64, device=rois.device)
    seq_inds = torch.arange(rois.shape[0], device=rois.device)
    map_back_inds[map_from_inds] = seq_inds
    return rois_by_level, map_back_inds
