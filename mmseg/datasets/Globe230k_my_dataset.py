
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

@DATASETS.register_module()
class Globe230k_my_dataset(BaseSegDataset):
    METAINFO = dict(
        classes=('Cropland', 'Forest', 'Grass', 'Shrub', 'Wetland', 'Water', 'Tundra', 'Impervious surface',
                 'Bareland', 'Ice/snow'),
        palette=[[252, 250, 205], [0, 123, 79], [157, 221, 106], [77, 208, 159], [111, 208, 242],
                 [10, 78, 151], [92, 106, 55], [155, 36, 22], [205, 205, 205], [211, 242, 255]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
