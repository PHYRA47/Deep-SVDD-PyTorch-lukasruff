import os
import random
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

class SL1HSDB(Dataset):
    def __init__(self, 
                 idx=None,
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
        self.overlapping = overlapping
        self.idx = idx

        self.verbose = verbose

        # Load all .mat files in the specified path
        self.mat_files = [f for f in os.listdir(self.path) if f.endswith('.mat')]

        # Define target wavelengths for interpolation if needed
        if self.interpolate:
            self.target_wavelengths = np.linspace(400, 700, 31)

    def _load_reflectance(self):
        """
        Load reflectance data from the .mat file. If idx is not provided, randomly pick one.

        Args:
            idx (int, optional): Index of the .mat file. If None, a random index is chosen.

        Returns:
            tuple: (np.ndarray, str)
                - Reflectance data.
                - Name of the picked file.
        """
        if self.idx is not None:
            idx = self.idx
        else:
            idx = random.randint(0, len(self.mat_files) - 1)

        file_name = self.mat_files[idx]
        file_path = os.path.join(self.path, file_name)
        mat_data = scipy.io.loadmat(file_path)
        hsi_data = mat_data['HS_skin_data']
        reflectance = hsi_data['reflectance'][0][0]

        if self.verbose:
            pass # print(f"Loading {file_name} with shape {reflectance.shape}")

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

        return reflectance, file_name

    def _generate_patches(self, idx):
        """
        Generate patches from the reflectance data.

        Args:
            idx (int): Index of the .mat file.

        Returns:
            list of tuples: [(patch_tensor, label, global_idx), ...]
        """
        # Load reflectance data
        reflectance, file_name = self._load_reflectance()

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

        # Shuffle positions to introduce randomness
        random.shuffle(possible_positions)

        # Select up to patches_per_file positions
        for i in range(min(self.patches_per_file, len(possible_positions))):
            top, left = possible_positions[i]
            patch = reflectance[:, top:top + self.patch_size, left:left + self.patch_size]
            patches.append((torch.tensor(patch, dtype=torch.float32), 0, idx))

        return patches

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
        patch_tensor = self._generate_patches(idx)

        if hasattr(self, 'global_offset'):
            global_idx = idx + self.global_offset
        else:
            global_idx = idx

        return patch_tensor, label, global_idx