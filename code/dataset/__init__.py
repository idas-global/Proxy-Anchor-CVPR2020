from .cars import Cars
from .cub import CUBirds
from .SOP import SOP
from .note_styles import Notes
from .note_families import NoteFamilies
from .import utils
from .base import BaseDataset


_type = {
    'cars': Cars,
    'cub': CUBirds,
    'SOP': SOP,
    'note_styles': Notes,
    'note_families': NoteFamilies

}

def load(name, root, mode, args, seed, le=None, transform = None):
    return _type[name](root = root, mode = mode, args=args, seed=seed, le=le, transform = transform)
    
