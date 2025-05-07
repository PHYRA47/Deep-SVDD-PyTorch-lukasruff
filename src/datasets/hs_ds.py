import torch
import torchvision.transforms as transforms
from base.torchvision_dataset import TorchvisionDataset
from datasets.SL1HSDataset import MultiSubjectSL1HSDBDataset
from datasets.SkinPatchDataset import SkinPatchDataset
from utils.preprocessing import get_target_label_idx, global_contrast_normalization

class HS_Dataset(TorchvisionDataset):
   
    def __init__(self, 
                 patch_size=32, 
                 transform=None):
        
        super().__init__(root=None)  # No root directory needed for on-the-fly generation

        # Pre-computed min and max values (after applying GCN)  
        min_value, max_value = (-2.0743157863616943, 3.0839202404022217) # data from 10 sub 10 patches/sub
        # (-2.3745954036712646, 3.5976827144622803) # data from 100 sub 50 patches/sub
        
        # If transform is None, do not apply any transformations
        if transform is None:
            self.transform = None
        else:
            # Default transform is to apply GCN and min-max scaling
            self.transform = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                transforms.Normalize([min_value] * 31, [max_value - min_value] * 31)
            ])

        # Noise scale for the dataset
        noise_scale = 0.025

        # -------------------------------------------------
        # Train set: Only real patches
        # -------------------------------------------------

        self.train_set_1 = SkinPatchDataset(
            num_subjects=100,
            patches_per_subject=100,
            patch_size=patch_size,
            noise_scale=noise_scale,
            applyRandomIllumination=True,
            isRealSkin=True,  # Only real patches
            transform=transform,
        )

        self.train_set_2 = MultiSubjectSL1HSDBDataset(
            num_subjects=50,
            patches_per_file= 1, # num_real_patches // num_subjects,
            patch_size=patch_size,
            transform=transform,
        )
            
        
        # -------------------------------------------------
        # Test set: Real and fake patches
        # -------------------------------------------------

        self.test_real_set_1 = SkinPatchDataset(
            num_subjects=100,
            patches_per_subject= 1, # num_real_patches // num_subjects,
            patch_size=patch_size,
            noise_scale=noise_scale,
            applyRandomIllumination=False,
            isRealSkin=True,  # Real patches
            transform=transform,
        )

        self.test_real_set_2 = MultiSubjectSL1HSDBDataset(
            num_subjects=50,
            patches_per_file= 2, # num_real_patches // num_subjects,
            patch_size=patch_size,
            transform=transform,
        )

        # Fake patches for the test set
        self.test_fake_set = SkinPatchDataset(
            num_subjects=10,
            patches_per_subject= 1, # num_fake_patches // num_subjects,
            patch_size=patch_size,
            noise_scale=noise_scale,
            isRealSkin=False,  # Fake patches
            applyRandomIllumination=True,
            transform=transform,
        )

        # Set global offsets for train sets
        self.train_set_1.global_offset = 0
        self.train_set_2.global_offset = len(self.train_set_1)

        # Combine train sets and adjust indices
        self.train_set = torch.utils.data.ConcatDataset([
            self.train_set_1
            # self.train_set_2
        ])

        # Set global offsets for test sets
        self.test_real_set_1.global_offset = 0
        self.test_real_set_2.global_offset = len(self.test_real_set_1)
        self.test_fake_set.global_offset = len(self.test_real_set_1) + len(self.test_real_set_2)

        # Combine real and fake test sets and adjust indices
        self.test_set = torch.utils.data.ConcatDataset([
            self.test_real_set_1,
            self.test_real_set_2,
            self.test_fake_set
        ])