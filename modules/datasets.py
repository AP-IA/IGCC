from torch.utils.data.sampler import BatchSampler
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from collections import defaultdict


class BatchDataset(Dataset):
    def __init__(self, root_dir, stage, txt_dir, transform=None, namelist=None):
        self.transform = transform

        if stage == "train":
            self.txt_path = os.path.join(txt_dir, "train.txt")
        elif stage == "val":
            self.txt_path = os.path.join(txt_dir, "val.txt")
        self.bird_names_path = os.path.join(txt_dir, "classes.txt")
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))

        if namelist:
            self.imglist = namelist
        else:
            with open(self.txt_path, 'r', encoding="utf-8") as fid:
                self.imglist = fid.readlines()

        self.label_ids = []
        cnt = 0
        with open(self.bird_names_path, "r", encoding="utf-8") as f:
            for line in f:
                self.label_ids.append(cnt)
                cnt += 1
        self.n_classes = len(self.label_ids)

        self.sample_labels = []
        self.sample_categories = []

        for line in self.imglist:
            parts = line.strip().split(',')
            image_path = parts[0]
            label = int(parts[1])
            self.sample_labels.append(label)

            category = "default"
            self.sample_categories.append(category)

        sorted_indices = sorted(range(len(self.imglist)), key=lambda i: self.sample_labels[i])

        self.imglist = [self.imglist[i] for i in sorted_indices]
        self.sample_labels = [self.sample_labels[i] for i in sorted_indices]
        self.sample_categories = [self.sample_categories[i] for i in sorted_indices]

        self.all_labels = np.array(self.sample_labels)

    def __getitem__(self, index):
        line_parts = self.imglist[index].strip().split(",")
        image_path = os.path.join(self.root_dir, line_parts[0])
        filename = os.path.basename(image_path)

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {image_path}, {e}")
            raise e

        if self.transform:
            image = self.transform(image)

        label = self.sample_labels[index]
        category = self.sample_categories[index]

        return image, label, filename, category

    def __len__(self):
        return len(self.imglist)


class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, n_classes, n_samples):
        self.dataset = dataset
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = n_classes * n_samples

        self.all_labels = dataset.all_labels
        self.label_ids = dataset.label_ids

        self.global_indices = {l: np.where(self.all_labels == l)[0] for l in self.label_ids}
        for l in self.label_ids:
            np.random.shuffle(self.global_indices[l])
        self.global_count = {l: 0 for l in self.label_ids}

        self.total_batches = len(dataset) // self.batch_size

    def _sample_global(self):
        labels = np.random.choice(self.label_ids, self.n_classes, replace=False)
        indices = []
        for label in labels:
            arr = self.global_indices[label]
            start = self.global_count[label]
            end = start + self.n_samples

            if end > len(arr):
                indices.extend(arr[start:])
                np.random.shuffle(arr)
                remain = self.n_samples - (len(arr) - start)
                indices.extend(arr[:remain])
                self.global_count[label] = remain
            else:
                indices.extend(arr[start:end])
                self.global_count[label] = end

                if self.global_count[label] == len(arr):
                    self.global_count[label] = 0
                    np.random.shuffle(arr)

        return indices

    def __iter__(self):
        for _ in range(self.total_batches):
            yield self._sample_global()

    def __len__(self):
        return self.total_batches