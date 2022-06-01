from .base import *

class CUBirds(BaseDataset):
    def __init__(self, root, mode, seed, le, transform=None):
        self.root = root + '/CUB_200_2011'
        self.mode = mode
        self.name = 'CUB'
        self.transform = transform
        self.perplex = 40

        if self.mode == 'train':
            self.classes = range(0,100)
        elif self.mode == 'eval':
            self.classes = range(100,200)

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        index = 0

        for i in torchvision.datasets.ImageFolder(root = 
                os.path.join(self.root, 'images')).imgs:
            # i[1]: label, i[0]: the full path to an image
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                self.ys += [y]
                self.I += [index]
                self.im_paths.append(i[0])
                index += 1

        self.class_names = self.im_paths
        self.class_names_coarse = [parse_im_name(specific_species, exclude_trailing_consonants=False)
                                   for specific_species in self.class_names]
        self.class_names_fine = [parse_im_name(specific_species, exclude_trailing_consonants=False, fine=True)
                                   for specific_species in self.class_names]
        
def parse_im_name(specific_species, exclude_trailing_consonants=False, fine=False):
    if fine:
        filter = os.path.split(os.path.split(specific_species)[0])[1].split('.')[-1].lower()
    else:
        coarse_filter = os.path.split(os.path.split(specific_species)[0])[1].split('_')[-1].lower()
        if '.' in coarse_filter:
            coarse_filter = coarse_filter.split('.')[-1]
        filter = coarse_filter

    if exclude_trailing_consonants:
        if filter[-1].isalpha():
            filter = filter[0:-1]
    return filter
