import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None, load_attr=False):
        """
        :param data_folder: folder where data files are stored - /Users/skye/docs/image_dataset/dataset
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # optional attribute labels (multi-label vector per image)
        self.load_attr = load_attr
        if self.load_attr:
            attr_clean_path = os.path.join(data_folder, "attr", "attr_labels_clean.npy")
            attr_path = os.path.join(data_folder, "attr", "attr_labels.npy")
            if os.path.exists(attr_clean_path) or os.path.exists(attr_path):
                import numpy as np
                self.attr_labels = np.load(attr_clean_path if os.path.exists(attr_clean_path) else attr_path)
            else:
                self.attr_labels = None

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':
            if self.load_attr and getattr(self, "attr_labels", None) is not None:
                y_attr = torch.FloatTensor(self.attr_labels[i // self.cpi])
                return img, caption, caplen, y_attr
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            if self.load_attr and getattr(self, "attr_labels", None) is not None:
                y_attr = torch.FloatTensor(self.attr_labels[i // self.cpi])
                return img, caption, caplen, all_captions, y_attr
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
