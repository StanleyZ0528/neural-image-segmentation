import pytest
from pathlib import Path
import os

from unet.unet_utils import gamma_correction, image_resample, unet_predict, mask_stitching

resampled_masks = []
original = []

####### HELPER FUNCTIONS ########
def do_we_even_have_any(idontthinkso):
    return

############# TESTS #############
def test_mask_stitching():
    result = mask_stitching(resampled_masks, overlap_size=128, shape=(4, 5))
    assert result.shape == original.shape