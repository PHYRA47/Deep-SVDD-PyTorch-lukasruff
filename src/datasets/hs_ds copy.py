import torch
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from base.torchvision_dataset import TorchvisionDataset
from utils.preprocessing import get_target_label_idx, global_contrast_normalization


class HS_Dataset(TorchvisionDataset):
    def __init__(self, num_subjects=10, patches_per_subject=10, patch_size=32, noise_scale=0.025):
        super().__init__(root=None)  # No root directory needed for on-the-fly generation

        self.n_classes = 2 # 0: normal, 1: outlier  
        self.normal_classes = tuple([0])    # Label 0: normal (real skin patches)
        self.outlier_classes = tuple([1])   # Label 1: outlier (anything not real)

        # Pre-computed min and max values (after applying GCN)  
        min_value, max_value = (-2.0743157863616943, 3.0839202404022217) # data from 10 sub 10 patches/sub
        # min_value, max_value = (-2.3745954036712646, 3.5976827144622803) # data from 100 sub 50 patches/sub
        
        # Preprocessing: GCN (with L1 norm) and min-max feature scaling 
        transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
            transforms.Normalize([min_value] * 31, [max_value - min_value] * 31)
        ])

        # Target transform: Change all subject IDs to 0
        # ---------------------------------------------------------------------------------
        # Here, everything is considered normal (real skin patches).
        # ---------------------------------------------------------------------------------
        target_transform = transforms.Lambda(lambda x: 0)

        # Create the PatchDataset
        self.dataset = SkinPatchDataset(
            num_subjects=num_subjects,
            patches_per_subject=patches_per_subject,
            patch_size=patch_size,
            noise_scale=noise_scale,
            transform=transform,
            target_transform=target_transform,
        )

        # Use the entire dataset for training (no explicit test set for anomalies)
        # self.train_set = self.dataset
        # self.test_set = self.dataset

        self.train_set = SkinPatchDataset(
            num_subjects=100,
            patches_per_subject=10,
            patch_size=patch_size,
            noise_scale=noise_scale,
            transform=transform,
            target_transform=target_transform,
        )
        self.test_set = SkinPatchDataset(
            num_subjects=2,
            patches_per_subject=200,
            patch_size=patch_size,
            noise_scale=noise_scale,
            transform=transform,
            target_transform=None
        )

class SkinPatchDataset(Dataset):

    def __init__(self, num_subjects=10, patches_per_subject=10, patch_size=16, noise_scale=0.025, 
                 transform=None, target_transform = None, verbose=False):

        # File paths
        reflectance_csv_path = '/cig/common05nb/students/denegasf/datasets/1832_Data_JResNIST_skinrefl_v3.csv'
        std_path = '/cig/common05nb/students/denegasf/datasets/UMINHO-HSFD/grid_files_v2/combined_grid_stats.mat'

        # Load reflectance data
        r_data = pd.read_csv(reflectance_csv_path, skiprows=7, encoding='latin1')  # Reflectance data
        std_data = loadmat(std_path)['std_of_mean_reflectance'][:31]              # Standard deviation data

        # Extract wavelength and reflectance
        wavelength = r_data['Wavelength (nm)'].to_numpy()
        reflectance = r_data.filter(like='Average').to_numpy()

        # Interpolate reflectance to a new wavelength range (400 to 720 nm, 10 nm step)
        new_wavelength = np.arange(400, 721, 10)[:31]
        interpolated_reflectance = np.zeros((len(new_wavelength), reflectance.shape[1]))
        for i in range(reflectance.shape[1]):
            interp_func = interp1d(wavelength, reflectance[:, i], kind='cubic', bounds_error=False, fill_value="extrapolate")
            interpolated_reflectance[:, i] = interp_func(new_wavelength)

        self.R_spectra = interpolated_reflectance[:31, :num_subjects]   # Use the first `num_subjects`
        self.sr_spectra = std_data[4][:31]                              # Use the std for the first 31 wavelengths
        self.sensor_sens = np.eye(len(new_wavelength))               # identity matrix for simplicity

        self.num_subjects = num_subjects
        self.patches_per_subject = patches_per_subject

        self.patch_size = patch_size
        self.noise_scale = noise_scale
        self.total_patches = num_subjects * patches_per_subject

        self.transform = transform
        self.target_transform = target_transform
        self.verbose = verbose

    def _generate_patch(self, subject_id):

        # Sample reflectance values with noise
        reflectance_cube = np.random.normal(
            loc=self.R_spectra[:, subject_id],
            scale=self.sr_spectra**2, # The square term is used here to reduce std to smaller values
            size=(self.patch_size, self.patch_size, self.R_spectra.shape[0])
        )

        # Integrate with sensor sensitivity
        intensity_cube = np.einsum('hwl,cl->hwc', reflectance_cube, self.sensor_sens)

        # Add shot-like sensor noise
        alpha = self.noise_scale
        std_dev = alpha * np.sqrt(np.clip(intensity_cube, 1e-10, None))
        noise = np.random.normal(loc=0.0, scale=std_dev)
        intensity_cube_noisy = intensity_cube + noise
        intensity_cube_noisy = np.clip(intensity_cube_noisy, 0.0, 1e3)  # Clip to a valid range

        # Convert to PyTorch tensor and permute to (bands, H, W)
        patch_tensor = torch.tensor(intensity_cube_noisy, dtype=torch.float32)
        patch_tensor = patch_tensor.permute(2, 0, 1)  # (bands, H, W)

        return patch_tensor
    
    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        # Determine the subject ID
        subject_id = idx // self.patches_per_subject

        # Generate a patch for the subject
        patch_tensor = self._generate_patch(subject_id)

        if self.transform:
            patch_tensor = self.transform(patch_tensor)

        if self.target_transform:
            label = self.target_transform(subject_id)
        else:
            label = subject_id
        
        return patch_tensor, label, idx   # label 0: normal, 1: outlier
    

    

    