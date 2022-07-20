Segmentation
============

UP supports the whole pipline of training, deploying, and interfering;

`Codes <https://github.com/ModelTC/EOD/tree/main/up/tasks/seg>`_

Configs
-------

It contains the illustration of common configs.

`Repos <https://github.com/ModelTC/EOD/tree/main/configs/seg>`_

Dataset related modules
-----------------------

1. Dataset types:

  * cityscapes

2. The type of datasets can be chosen by setting 'seg_type' in SegDataset (default is cityscapes). The config is as followed.

  .. code-block:: yaml

    dataset:
      type: cls
      kwargs:
        seg_type: cityscapes    # Default is cityscapes. Options: [cityscapes]
        meta_file: cityscapes/fine_train.txt
        image_reader:
           type: fs_opencv
           kwargs:
             image_dir: cityscapes
             color_mode: RGB
        seg_label_reader:
           type: fs_opencv
           kwargs:
             image_dir: cityscapes
             color_mode: GRAY
        transformer: [*seg_rand_resize, *flip, *seg_crop_train, *to_tensor, *normalize]
        num_classes: *num_classes
        ignore_label: 255
