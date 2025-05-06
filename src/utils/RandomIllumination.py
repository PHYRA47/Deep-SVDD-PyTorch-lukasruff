
import random
import torch
import numpy as np
from colour import SDS_ILLUMINANTS

class RandomIllumination:
    """
    A data augmentation class that applies random illumination changes to input tensors.
    Attributes:
        p (float): Probability of applying the random illumination. Default is 0.5.
        illum_sources (list): A list of normalized spectral distributions for various illumination sources.
    Methods:
        __call__(x):
            Applies a random illumination to the input tensor `x` and returns the index of the illumination source.
    Args:
        p (float, optional): Probability of applying the random illumination. Default is 0.5.
    Example:
        >>> random_illum = RandomIllumination(p=0.7)
        >>> augmented_tensor, illuminant = random_illum(input_tensor)
    Notes:
        - The illumination sources are predefined and interpolated to match a specific wavelength range.
        - The input tensor `x` is expected to have a shape of [C, H, W] or [bands, H, W].
        - The output tensor values are clamped between 0.0 and 1.0 to ensure valid intensity ranges.
    """
    def __init__(self, p=1.0):
        self.p = p
        # Target illumination
        self.sources = ['A', 'D50', 'D55', 'E', 'HP1', 'ISO 7589 Photographic Daylight', 'ID65', 'D65', 'FL1', 'FL10', 'LED-B1', 'LED-RGB1']
        # sources = ['D65']
        self.illum_sources = []
        for illum in self.sources:
            target_sd_values = SDS_ILLUMINANTS[illum].values
            target_sd_domain = SDS_ILLUMINANTS[illum].domain
            target_sd = np.stack([target_sd_domain, target_sd_values], 1)
            target_sd_interpolated = np.interp(np.linspace(400., 700., 31), target_sd[:, 0], target_sd[:, 1])
            self.illum_sources.append(target_sd_interpolated / np.max(target_sd_interpolated))
        
    def __call__(self, x): 
        
        if random.random() > self.p:
            return x, "None"
        
        # x is a torch.Tensor, shape [C, H, W] or [bands, H, W]
        illum_index = random.randint(0, len(self.illum_sources) - 1)
        target_illum = torch.from_numpy(self.illum_sources[illum_index]).float()

        power_factor = random.uniform(0.5, 1.5)  # Random illumination power factor
        offset = random.uniform(-0.1, 0.1)  # Random offset
        illumination_tensor = target_illum.unsqueeze(-1).unsqueeze(-1) # unsqueeze(0)

        illuminated_tensor = x * illumination_tensor * power_factor + offset
        return torch.clamp(illuminated_tensor , 0.0, 1.0), self.sources[illum_index]