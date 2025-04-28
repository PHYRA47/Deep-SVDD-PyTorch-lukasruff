import torch
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
from prettytable import PrettyTable


class HS_Dataset(TorchvisionDataset):
   
    def __init__(self, 
                 num_subjects=10, 
                 patches_per_subject=10, 
                 patch_size=32, 
                 noise_scale=0.025):
        
        super().__init__(root=None)  # No root directory needed for on-the-fly generation

        # Pre-computed min and max values (after applying GCN)  
        min_value, max_value = (-2.0743157863616943, 3.0839202404022217) # data from 10 sub 10 patches/sub
        # (-2.3745954036712646, 3.5976827144622803) # data from 100 sub 50 patches/sub
        
        # Preprocessing: GCN (with L1 norm) and min-max feature scaling 
        transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
            transforms.Normalize([min_value] * 31, [max_value - min_value] * 31)
        ])

        # Train set: Only real patches
        self.train_set = SkinPatchDataset(
            num_subjects=100,
            patches_per_subject=patches_per_subject,
            patch_size=patch_size,
            noise_scale=noise_scale,
            isRealSkin=True,  # Only real patches
            transform=transform,
        )

        # Test set: Mix of real and fake patches
        fake_percentage = 0.6 # 60% fake patches

        num_fake_patches = int((fake_percentage) * num_subjects * patches_per_subject)
        num_real_patches = (num_subjects * patches_per_subject) - num_fake_patches

        # Real patches for the test set
        self.test_real_set = SkinPatchDataset(
            num_subjects=100,
            patches_per_subject= 1, # num_real_patches // num_subjects,
            patch_size=patch_size,
            noise_scale=noise_scale,
            isRealSkin=True,  # Real patches
            transform=transform,
        )

        # Fake patches for the test set
        self.test_fake_set = SkinPatchDataset(
            num_subjects=100,
            patches_per_subject= 1, # num_fake_patches // num_subjects,
            patch_size=patch_size,
            noise_scale=noise_scale,
            isRealSkin=False,  # Fake patches
            transform=transform,
        )

        # Set global offsets for test sets
        self.test_real_set.global_offset = 0
        self.test_fake_set.global_offset = len(self.test_real_set)

        # Combine real and fake test sets and adjust indices
        self.test_set = torch.utils.data.ConcatDataset([
            self.test_real_set,
            self.test_fake_set
        ])

class SkinPatchDataset(Dataset):
    
    def __init__(self, 
                 num_subjects=10, 
                 randomize=False,
                 patches_per_subject=10, 
                 patch_size=16, 
                 noise_scale=0.025, 
                 isRealSkin=True,
                 transform=None, 
                 target_transform=None, 
                 verbose=False):
        """
        Initialize the SkinPatchDataset.

        Args:
            num_subjects (int): Number of subjects to load.
            patches_per_subject (int): Number of patches per subject.
            patch_size (int): Size of each patch (square).
            noise_scale (float): Scale of the noise to add.
            isRealSkin (bool): Whether to load real or fake skin data.
            transform (callable, optional): Transform to apply to the patches.
            target_transform (callable, optional): Transform to apply to the labels.
            verbose (bool): If True, print debug information.
        """
        self.verbose = verbose

        if self.verbose:
            print("=========================================")
            print("----- Initializing SkinPatchDataset -----")
            print(f"{'Skin type':<30}: {'Real' if isRealSkin else 'Fake'}")
            print(f"{'Number of subjects':<30}: {num_subjects}")
            print(f"{'Patches per subject':<30}: {patches_per_subject}")
            print(f"{'Total patches':<30}: {num_subjects * patches_per_subject}")
            print(f"{'Patch size':<30}: {patch_size}x{patch_size}")
            print(f"{'Noise scale':<30}: {noise_scale}")
            print("-----------------------------------------")

        # Dataset parameters
        self.num_subjects = num_subjects
        self.randomize = randomize
        self.patches_per_subject = patches_per_subject
        self.patch_size = patch_size
        self.noise_scale = noise_scale
        self.isRealSkin = isRealSkin
        self.total_patches = num_subjects * patches_per_subject

        # File paths
        R_real_csv_path = [
            '/cig/common05nb/students/denegasf/datasets/1832_Data_JResNIST_skinrefl_v3_average_only.csv'
        ]
        R_fake_csv_path = [
            "/cig/common05nb/students/denegasf/datasets/UMINHO-HSFD/reconstructed/mst-plus-plus/reconstruction_reflectance_data.csv",
            "/cig/common05nb/students/denegasf/datasets/UMINHO-HSFD/reconstructed/restormer/reconstruction_reflectance_data.csv"
        ]
        sr_mat_path = '/cig/common05nb/students/denegasf/datasets/UMINHO-HSFD/grid_files_v2/combined_grid_stats.mat'

        # Load and process reflectance data based on isRealSkin
        if isRealSkin:
            if self.verbose:
                print("-- Loading Real Skin Reflectance Data  --")
            self.reflectance_data = self._load_reflectance(R_real_csv_path, interpolate=True, num_subjects=num_subjects)
        else:
            if self.verbose:
                print("-- Loading Fake Skin Reflectance Data  --")
            self.reflectance_data = self._load_reflectance(R_fake_csv_path, interpolate=False, num_subjects=num_subjects)

        # Load standard deviation data
        if self.verbose:
            print("---- Loading Standard Deviation Data ----")
        self.sr = loadmat(sr_mat_path)['std_of_mean_reflectance'][4][:31]

        # Sensor sensitivity (identity matrix for simplicity)
        self.sensor_sens = np.eye(31)

        # Defined wavelength range (400–700 nm in 10 nm steps)
        self.wavelength = np.arange(400, 701, 10)

        # Transformations
        self.transform = transform
        self.target_transform = target_transform

        if self.verbose:
            print("-------- Initialization Complete --------")
            print("=========================================")

    def _load_reflectance(self, csv_paths, interpolate=False, num_subjects=None):
        """
        Load reflectance data from multiple CSV files.
        If interpolate=True, interpolate the data to match the wavelength range (400–700 nm in 10 nm steps).
        If num_subjects is specified, randomly or sequentially select that many subjects based on `randomize`.
        """
        reflectance_list = []
        for csv_path in csv_paths:
            # Load data from CSV
            data = pd.read_csv(csv_path, skiprows=7 if interpolate else 0, encoding='latin1')
            wavelength = data.iloc[:, 0].to_numpy()  # First column is wavelength
            reflectance = data.iloc[:, 1:].to_numpy()  # Remaining columns are reflectance

            wvl = np.arange(400, 701, 10)  # New wavelength range 
            
            if interpolate:
                if self.verbose:
                    print(f"{'Interpolation done':<30}: {'True' if interpolate else 'False'}")
                    
                # Interpolate reflectance to a new wavelength range (400 to 720 nm, 10 nm step)
                interpolated_reflectance = np.zeros((len(wvl), reflectance.shape[1]))
                for i in range(reflectance.shape[1]):
                    interp_func = interp1d(wavelength, reflectance[:, i], kind='cubic', bounds_error=False, fill_value="extrapolate")
                    interpolated_reflectance[:, i] = interp_func(wvl)
                reflectance_list.append(interpolated_reflectance)
            else:
                # Ensure the wavelength matches the expected range
                if not np.array_equal(wavelength, wvl):
                    raise ValueError(f"Wavelengths in {csv_path} do not match the expected range.")
                reflectance_list.append(reflectance)

        # Combine all reflectance data
        combined_reflectance = np.concatenate(reflectance_list, axis=1)

        # If num_subjects is specified, select that many subjects
        if num_subjects:
            total_subjects = combined_reflectance.shape[1]
            if self.verbose:
                print(f"{'Total ' + ('real' if self.isRealSkin else 'fake') + ' skin subjects':<30}: {total_subjects}")
                print(f"{'Selected subjects':<30}: {num_subjects}")
                print(f"{'Randomize':<30}: {self.randomize}")
            if num_subjects > total_subjects:
                raise ValueError(f"num_subjects ({num_subjects}) exceeds the number of available subjects ({total_subjects}).")
            if num_subjects < 1:
                raise ValueError("num_subjects must be at least 1.")

            if self.randomize:
                selected_indices = np.random.choice(total_subjects, num_subjects, replace=False)
                combined_reflectance = combined_reflectance[:, selected_indices]
            else:
                combined_reflectance = combined_reflectance[:, :num_subjects]

        if self.verbose:
            print("--- Reflectance data loading complete ---")
            print("-----------------------------------------")

        return combined_reflectance

    def _generate_patch(self, subject_id=None):
        """
        Generate a patch for the given data type ('real' or 'fake') and subject ID.
        """
        # Use the loaded reflectance data
        reflectance_data = self.reflectance_data

        # If subject_id is not provided, randomly select one
        if subject_id is None:
            subject_id = np.random.randint(0, reflectance_data.shape[1])

        # Sample reflectance values with noise
        reflectance_cube = np.random.normal(
            loc=reflectance_data[:, subject_id],
            scale=self.sr**2,
            size=(self.patch_size, self.patch_size, reflectance_data.shape[0])
        )

        # Apply sensor sensitivity
        intensity_cube = self._apply_sensor_sensitivity(reflectance_cube)

        # Add sensor noise
        intensity_cube_noisy = self._add_sensor_noise(intensity_cube)

        # Convert to PyTorch tensor and permute to (bands, H, W)
        patch_tensor = torch.tensor(intensity_cube_noisy, dtype=torch.float32)
        patch_tensor = patch_tensor.permute(2, 0, 1)  # (bands, H, W)

        return patch_tensor
    
    def __getitem__(self, idx):
        """
        Get a patch and its label by index.
        """
        subject_id = idx // self.patches_per_subject

        if self.verbose:
            print(f"------------ Generating Patch  ---------")
            print(f"Index: {idx}")
            print(f"Subject ID: {subject_id}")
            print(f"Skin Type: {'Real' if self.isRealSkin else 'Fake'}")

        patch_tensor = self._generate_patch(subject_id=subject_id)

        if self.transform:
            patch_tensor = self.transform(patch_tensor)

        label = 0 if self.isRealSkin else 1

        # Return the patch, label, and global index
        if hasattr(self, 'global_offset'):
            global_idx = idx + self.global_offset
        else:
            global_idx = idx

        if self.verbose:
            print(f"Patch generated successfully.")
            print(f"Label: {label}")
            print("-----------------------------------------")

        return patch_tensor, label, global_idx

    def __len__(self):
        return self.total_patches
    def _apply_sensor_sensitivity(self, reflectance_cube):
        """
        Apply sensor sensitivity to the reflectance cube.
        """
        # Integrate with sensor sensitivity
        intensity_cube = np.einsum('hwl,cl->hwc', reflectance_cube, self.sensor_sens)
        return intensity_cube

    def _add_sensor_noise(self, intensity_cube):
        """
        Add shot-like sensor noise to the intensity cube.
        """
        alpha = self.noise_scale
        std_dev = alpha * np.sqrt(np.clip(intensity_cube, 1e-10, None))
        noise = np.random.normal(loc=0.0, scale=std_dev)
        intensity_cube_noisy = intensity_cube + noise
        intensity_cube_noisy = np.clip(intensity_cube_noisy, 0.0, 1e3)  # Clip to a valid range
        return intensity_cube_noisy
