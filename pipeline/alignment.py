import numpy as np
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as ndi_shift

class Align:
    def __init__(self):
        pass

    def register_translate(reference, target, upsample_factor=100):
        # strip units if using astropy Quantity
        ref = getattr(reference, 'value', reference)
        tgt = getattr(target, 'value', target)

        # optionally mask bad pixels / use a high-pass to emphasize stars
        shift, error, diffphase = phase_cross_correlation(ref, tgt, upsample_factor=upsample_factor)
        # shift is (y_shift, x_shift) to apply to 'target' to align with 'reference'
        aligned = ndi_shift(tgt, shift=shift, order=3)  # cubic interpolation
        return aligned, shift

if __name__ == '__main__': 
    align = Align()
    aligned, shift = align.register_translate(reference_cutout.data, other_cutout.data)
    print("Applied shift (y,x):", shift)
