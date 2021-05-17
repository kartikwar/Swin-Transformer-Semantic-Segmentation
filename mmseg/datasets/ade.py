from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ADE20KDataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = ('foreground')

    PALETTE = [[255, 255, 255]]

    def __init__(self, **kwargs):
        super(ADE20KDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.jpg',
            reduce_zero_label=True,
            **kwargs)
