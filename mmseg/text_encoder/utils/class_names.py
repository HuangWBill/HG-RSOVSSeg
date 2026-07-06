
from mmengine.utils import is_str

def potsdam_classes():
    """Potsdam class names for external use."""
    return [
        'impervious surface', 'building', 'low vegetation', 'tree', 'car'
    ]

def loveda_classes():
    """LoveDA class names for external use."""
    return [
        'building', 'road', 'water', 'barren', 'forest','agricultural'
    ]

def globe230k_classes():
    """Globe230k class names for external use."""
    return [
        'cropland', 'forest', 'grass', 'shrub', 'wetland', 'water', 'tundra', 'impervious surface',
        'bareland', 'ice'
    ]

def flair1_classes():
    """flair1 class names for external use."""
    return [
        'building', 'pervious surface', 'impervious surface', 'bare soil', 'water','coniferous','deciduous',
        'brushwood','vineyard','herbaceous vegetation','agricultural land','plowed land'
    ]

def gid_large_classes():
    """gid_large class names for external use."""
    return [
        'built up', 'farmland', 'forest', 'meadow', 'water'
    ]

def landcover_ai_classes():
    """flair1 class names for external use."""
    return [
        'buildings', 'woodlands', 'water', 'roads'
    ]

def openearthmap_classes():
    """openearthmap class names for external use."""
    return [
        'bareland', 'rangeland', 'developed space', 'road', 'tree', 'water', 'agriculture land', 'building'
    ]




def potsdam_palette():
    """Potsdam palette for external use."""
    return [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
            [255, 255, 0]]

def loveda_palette():
    """LoveDA palette for external use."""
    return [[255, 0, 0], [255, 255, 0], [0, 0, 255],
            [159, 129, 183], [0, 255, 0], [255, 195, 128]]

def globe230k_palette():
    """Globe230k palette for external use."""
    return [[252, 250, 205], [0, 123, 79], [157, 221, 106], [77, 208, 159], [111, 208, 242],
            [10, 78, 151], [92, 106, 55], [155, 36, 22], [205, 205, 205], [211, 242, 255]]

def flair1_palette():
    """FLAIR1 palette for external use."""
    return [[219, 14, 154], [147, 142, 123], [248, 12, 0],[169, 113, 1], [21, 83, 174], [25, 74, 38], [70, 228, 131],
            [243, 166, 13], [102, 0, 130], [85, 255, 0], [255, 243, 13], [228, 223, 124]]

def gid_large_palette():
    """gid_large palette for external use."""
    return [[255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 255, 0], [0, 0, 255]]

def landcover_ai_palette():
    """landcover_ai palette for external use."""
    return [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]

def openearthmap_palette():
    """openearthmap palette for external use."""
    return [[128, 0, 0], [0, 255, 36], [148, 148, 148], [255, 255, 255], [34, 97, 38], [0, 69, 255], [75, 181, 73], [222, 31, 7]]






dataset_aliases = {
    'potsdam': ['potsdam'],
    'loveda': ['loveda'],
    'gid_large':['gid_large'],
    'globe230k': ['globe230k'],
    'flair1': ['flair1'],
    'landcover_ai':['landcover_ai'],
    'openearthmap': ['openearthmap']
}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels


def get_palette(dataset):
    """Get class palette (RGB) of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_palette()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels
