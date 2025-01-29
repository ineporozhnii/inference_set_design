from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from inference_set_design.tasks.base_task import BaseTask
from inference_set_design.utils.metrics import get_classification_uncertainty


class CorruptedMNISTAcquisition(BaseTask):
    @property
    def input_size(self):
        return self.explorable_set[0][0].shape[0]

    @property
    def n_classes(self):
        return len(self.explorable_set.classes)

    def load_data(self):

        # Get (corrupted) MNIST data
        class FlattenTransform:
            def __call__(self, img):
                return torch.flatten(img)

        class HideBottomTransform:
            def __call__(self, x):
                x[:, 10:, :] = 0
                return x

        if self.cfg.corruption_type in ["bottom", "all"]:
            transf_list = [transforms.ToTensor(), HideBottomTransform(), FlattenTransform()]
        else:
            transf_list = [transforms.ToTensor(), FlattenTransform()]

        self.explorable_set = datasets.MNIST(
            self.cfg.data_path, train=True, download=True, transform=transforms.Compose(transf_list)
        )
        self.test_set = datasets.MNIST(
            self.cfg.data_path, train=False, download=True, transform=transforms.Compose(transf_list)
        )

        if self.cfg.corruption_type in ["689_random", "all"]:
            # Set the labels of any input that is 6, 8, or 9 to a random label among 6, 8, 9
            self.explorable_set.targets = np.where(
                np.isin(self.explorable_set.targets, [6, 8, 9]),
                np.random.choice([6, 8, 9], size=len(self.explorable_set.targets)),
                self.explorable_set.targets,
            )
            self.test_set.targets = np.where(
                np.isin(self.test_set.targets, [6, 8, 9]),
                np.random.choice([6, 8, 9], size=len(self.test_set.targets)),
                self.test_set.targets,
            )

        if self.cfg.corruption_type is not None:
            if self.cfg.corruption_type.startswith("random_up_to"):
                # Set random labels of the first n classes among 1-n
                n_classes = int(self.cfg.corruption_type.split("_")[-1]) + 1
                self.explorable_set.targets = np.where(
                    np.isin(self.explorable_set.targets, np.arange(n_classes)),
                    np.random.choice(np.arange(n_classes), size=len(self.explorable_set.targets)),
                    self.explorable_set.targets,
                )
                self.test_set.targets = np.where(
                    np.isin(self.test_set.targets, np.arange(n_classes)),
                    np.random.choice(np.arange(n_classes), size=len(self.test_set.targets)),
                    self.test_set.targets,
                )

        if self.cfg.shuffle_datasets:
            # Shuffle explorable and test sets with config random seed
            explorable_set_shuffle = np.random.permutation(len(self.explorable_set))
            self.explorable_set.data = self.explorable_set.data[explorable_set_shuffle]
            self.explorable_set.targets = self.explorable_set.targets[explorable_set_shuffle]

            test_set_shuffle = np.random.permutation(len(self.test_set))
            self.test_set.data = self.test_set.data[test_set_shuffle]
            self.test_set.targets = self.test_set.targets[test_set_shuffle]

        # Update the acquisition_mask
        if self.reveal_all:
            self.revealed_imgs = np.ones(
                len(
                    self.explorable_set,
                )
            )
        else:
            self.revealed_imgs = np.zeros(
                len(
                    self.explorable_set,
                )
            )
            self.revealed_imgs[: self.cfg.n_init_train_imgs] = 1

        self.acquisition_mask = self.get_acquisition_mask()

    def get_acquisition_mask(self):
        return 1.0 - self.revealed_imgs

    def label_batch(self, acquisition_idxs: List[int]):
        for i in acquisition_idxs:
            self.revealed_imgs[i] = 1
        self.acquisition_mask = self.get_acquisition_mask()

    def build_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool = False,
    ):
        class IndexAugmentedDataloader(DataLoader):
            def __iter__(self):
                # Create an iterator for the original DataLoader
                self.iterator = super(IndexAugmentedDataloader, self).__iter__()

                for batch in self.iterator:
                    # Assuming the batch is a tuple of (data, target)
                    data, target = batch

                    # Fake indices for the batch
                    indices = torch.full((len(data),), fill_value=-1, dtype=torch.long)

                    yield data, target, indices

        return IndexAugmentedDataloader(
            dataset=dataset,
            batch_size=self.model_cfg.train_batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_data_workers,
        )

    def get_explorable_dataloader(self):
        explorable_loader = self.build_dataloader(self.explorable_set, shuffle=False)
        return explorable_loader

    def get_hidden_dataloader(self, acquisition_mask: np.ndarray):
        available_acq_idxs = np.argwhere(acquisition_mask == 1).flatten()

        if len(available_acq_idxs) > 0:
            explorable_hidden_loader = self.build_dataloader(
                dataset=torch.utils.data.Subset(self.explorable_set, available_acq_idxs), shuffle=False
            )
        else:
            explorable_hidden_loader = None

        return explorable_hidden_loader

    def get_train_dataloader(self, acquisition_mask: np.ndarray):
        revealed_acq_idxs = np.argwhere(acquisition_mask == 0).flatten()

        if len(revealed_acq_idxs) > 0:
            train_loader = self.build_dataloader(
                dataset=torch.utils.data.Subset(self.explorable_set, revealed_acq_idxs), shuffle=False
            )
        else:
            train_loader = None

        return train_loader

    def get_valid_dataloader(self):
        # validation loader gets the first half of the test compounds
        validation_idxs = np.arange(0, len(self.test_set) // 2)

        if len(validation_idxs) > 0:
            validation_loader = self.build_dataloader(
                dataset=torch.utils.data.Subset(self.test_set, validation_idxs), shuffle=False
            )
        else:
            validation_loader = None

        return validation_loader

    def get_test_dataloader(self):
        # test loader gets the second half of the test compounds
        test_idxs = np.arange(len(self.test_set) // 2, len(self.test_set))

        if len(test_idxs) > 0:
            test_loader = self.build_dataloader(
                dataset=torch.utils.data.Subset(self.test_set, test_idxs), shuffle=False
            )
        else:
            test_loader = None

        return test_loader

    def get_batch_dataloader(self, batch: np.ndarray):
        if len(batch) > 0:
            batch_loader = self.build_dataloader(
                dataset=torch.utils.data.Subset(self.explorable_set, batch), shuffle=True
            )
        else:
            batch_loader = None

        return batch_loader

    def get_acquisition_scores(self, class_probs: np.ndarray, x_idxs: np.ndarray):
        n_samples, n_classes = class_probs.shape
        assert n_classes == self.n_classes
        # TODO: MAKE SURE THE ORDERING FROM THE DATALOADER THAT PRODUCES class_probs
        # IS THE SAME AS THE ORDERING OF COMPOUNDS IN THE ACQUISITION MASKS

        # Get the mean uncertainty for compounds
        uncertainty = get_classification_uncertainty(class_probs, n_classes=n_classes)

        # Prepare the acquisition scores
        acquisition_scores = {
            "uncertainty": uncertainty,
        }
        acquisition_metrics = {
            "max_uncertainty": float(np.mean(uncertainty)),
            "mean_uncertainty": float(np.max(uncertainty)),
        }

        return acquisition_metrics, acquisition_scores
