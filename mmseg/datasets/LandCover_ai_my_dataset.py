
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

@DATASETS.register_module()
class LandCover_ai_my_dataset(BaseSegDataset):
    METAINFO = dict(
        classes=('buildings', 'woodlands', 'water', 'roads'),
        palette=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
