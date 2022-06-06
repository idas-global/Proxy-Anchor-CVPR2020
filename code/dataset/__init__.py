from .cars import Cars
from .cub import CUBirds
from .SOP import SOP
from .note_styles import Notes
from .note_families import Families
from .Paper import Paper
from .base import BaseDataset

_type = {
    'cars': Cars,
    'cub': CUBirds,
    'SOP': SOP,
    'note_styles': Notes,
    'note_families_front': Families,
    'note_families_back': Families,
    'note_families_seal': Families,
    'paper' : Paper,
}


def load(name, root, mode, seed, le=None, transform=None):
    if name.startswith('note_families'):
        return _type[name](root=root, mode=mode, seed=seed, le=le, transform=transform, plate=name.split('_')[-1])
    return _type[name](root=root, mode=mode, seed=seed, le=le, transform=transform)
