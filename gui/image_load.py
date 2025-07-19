from PIL import Image
from astropy.io import fits
import numpy as np

class ImageLoader:
    def __init__(self):
        pass

    def fits_to_image(self, path) -> Image:
        hdul = fits.open(path)
        data = hdul[0].data
        hdul.close()

        # Normalize and convert to 8-bit image
        norm_data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
        img = Image.fromarray(norm_data.astype(np.uint8))
        return img