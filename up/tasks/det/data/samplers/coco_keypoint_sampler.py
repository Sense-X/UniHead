import math
import numpy as np
import torch
from torch.utils.data.sampler import Sampler

from up.utils.env.dist_helper import get_rank, get_world_size
from up.utils.general.registry_factory import SAMPLER_REGISTRY


@SAMPLER_REGISTRY.register('coco_keypoint')
class CocoKeypointDistributedSampler(Sampler):
    """ Make sure that number of keypoints within a batch_size > 20
    """
    def __init__(self,
                 dataset,
                 batch_size=2,
                 num_replicas=None,
                 rank=None):
        """
        Arguments:
             - dataset (:obj:`dataset`): instance of dataset object
             - batch_Size (:obj:`int`)
        """
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()

        self.dataset = dataset
        assert (hasattr(dataset, 'aspect_ratios'))
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self.batch_size = batch_size
        self.coco = dataset.coco
        # It's important to keep img_ids consistent with dataset.img_ids
        # so that the dataset fetch the correct data by image indices sampled here
        self.img_ids = dataset.img_ids
        self.valid_kpt_num = np.zeros((len(self.dataset)))
        for i in range(len(self.dataset)):
            self.valid_kpt_num[i] = self._get_valid_kpt_num(self.img_ids[i])

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = list(torch.randperm(len(self.dataset), generator=g))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]

        # sort samples by image aspect ratios
        indices = sorted(indices, key=lambda ix: int(self.dataset.aspect_ratios[ix] * 10))

        # check indices
        for i in range(0, len(indices), self.batch_size):
            if i + self.batch_size >= len(indices):
                break
            idx = indices[i:i + self.batch_size]
            while True:
                sum_valid_kpt = 0
                for cur_idx in idx:
                    sum_valid_kpt += self.valid_kpt_num[cur_idx]

                if sum_valid_kpt >= 20:
                    break

                # skip batch, resample
                idx = list(torch.randperm(len(indices), generator=g))[:self.batch_size]
                idx = [indices[j] for j in idx]
            indices[i:i + self.batch_size] = idx[:]

        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _get_valid_kpt_num(self, img_id):
        res = 0
        meta_annos = self.coco.imgToAnns[img_id]
        for ann in meta_annos:
            if ann['iscrowd']:
                continue
            kpt = np.array(ann['keypoints']).reshape(-1, 3)
            res += np.sum(kpt[:, -1] > 0)
        return res