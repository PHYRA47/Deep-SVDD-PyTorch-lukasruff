import os
import random
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

class SL1HSDB(Dataset):
    def __init__(self, 
                 num_subjects=None,
                 randomize=True,
                 patch_size=32, 
                 interpolate=True, 
                 patches_per_file=1, 
                 overlapping=False,
                 
                 verbose=False):    
        """
        Initialize the dataset.

        Args:
            patch_size (int): Size of the square patch (height and width).
            interpolate (bool): Whether to interpolate the reflectance data to target wavelengths.
            patches_per_file (int): Number of patches to generate per .mat file.
            overlapping (bool): Whether to allow overlapping patches.
        """
        self.path = '/cig/common02nb/HS_skin_DB/RGB2HSI_dataset/2_Dataset/SL1DB_RGB2HSI_raw_HSI/HS_cubes/'
        self.patch_size = patch_size  # Single integer for square patch
        self.interpolate = interpolate
        self.patches_per_file = patches_per_file
        self.num_subjects = num_subjects
        self.randomize = randomize
        self.overlapping = overlapping

        self.verbose = verbose

        # Load all .mat files in the specified path
        self.mat_files = [f for f in os.listdir(self.path) if f.endswith('.mat')]

        # Define target wavelengths for interpolation if needed
        if self.interpolate:
            self.target_wavelengths = np.linspace(400, 700, 31)

    def _load_reflectance(self, num_subjects=1, randomize=True):
        """
        Load reflectance data from the .mat files. If idx is not provided, randomly pick files.

        Args:
            num_subjects (int): Number of subjects to process.
            randomize (bool): Whether to randomly select the files.

        Returns:
            tuple: (list of np.ndarray, list of str)
                - List of reflectance data arrays.
                - List of corresponding file names.
        """
        if randomize:
            selected_indices = random.sample(range(len(self.mat_files)), num_subjects)
        else:
            selected_indices = list(range(min(num_subjects, len(self.mat_files))))

        reflectance_list = []
        file_names = []

        for idx in selected_indices:
            file_name = self.mat_files[idx]
            file_path = os.path.join(self.path, file_name)
            mat_data = scipy.io.loadmat(file_path)
            hsi_data = mat_data['HS_skin_data']
            reflectance = hsi_data['reflectance'][0][0]

            if self.verbose:
                pass  # print(f"Loading {file_name} with shape {reflectance.shape}")

            # Interpolate reflectance if required
            if self.interpolate:
                original_wavelengths = hsi_data['wavelengths'][0][0][0]
                reflectance_interpolated = np.zeros((31, reflectance.shape[1], reflectance.shape[2]))
                for i in range(reflectance.shape[1]):  # Iterate over rows
                    for j in range(reflectance.shape[2]):  # Iterate over columns
                        reflectance_interpolated[:, i, j] = np.interp(
                            self.target_wavelengths, original_wavelengths, reflectance[:, i, j]
                        )
                reflectance = reflectance_interpolated

            reflectance_list.append(reflectance)
            file_names.append(file_name)

        return reflectance_list, file_names

    def _generate_patches(self):
        """
        Generate patches from the reflectance data.

        Args:
            idx (int): Index of the .mat file.

        Returns:
            list of tuples: [(patch_tensor, label, global_idx), ...]
        """
        # Load reflectance data
        reflectance, file_name = self._load_reflectance(self.num_subjects, self.randomize)

        # Determine stride based on overlapping
        random_stride = self.patch_size // 2 if self.overlapping else self.patch_size

        h, w = reflectance.shape[1], reflectance.shape[2]
        patches = []

        # Generate patches with the determined stride
        possible_positions = [
            (i, j)
            for i in range(0, h - self.patch_size + 1, random_stride)
            for j in range(0, w - self.patch_size + 1, random_stride)
        ]

        # Randomly select one position
        top, left = random.choice(possible_positions)

        # Extract the patch
        patch = reflectance[:, top:top + self.patch_size, left:left + self.patch_size]

        # Convert the patch to a PyTorch tensor
        patch_tensor = torch.tensor(patch, dtype=torch.float32)

        return patch_tensor, file_name

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        """
        Get patches from the dataset.

        Args:
            idx (int): Index of the .mat file.

        Returns:
            list of tuples: [(patch_tensor, label, global_idx), ...]
                - patch_tensor (torch.Tensor): Patch of reflectance data.
                - label (int): Label indicating real skin (0).
                - global_idx (int): Global index of the patch.
        """
        # Assign label based on skin type
        label = 0

        # Generate patches
        patch_tensor, file_name = self._generate_patches()

        if hasattr(self, 'global_offset'):
            global_idx = idx + self.global_offset
        else:
            global_idx = idx

        if self.verbose:
            print(f"Generating Patch".center(32, '-'))
            print(f"{'From':<20}: {file_name:>10}")
            print(f"{'Patch size':<20}: {f'{self.patch_size} x {self.patch_size}':>10}")
            print(f"{'Label':<20}: {label:>10}")
            print("-"*32)

        return patch_tensor, label, global_idx