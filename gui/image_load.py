from PIL import Image
from astropy.io import fits
import numpy as np

class ImageLoader:
    def __init__(self):
        pass

    #filter for star visibility, only on preview images, does not affect the actual frames.
    def fits_to_image(self, path) -> Image:
        hdul = fits.open(path)
        data = hdul[0].data
        hdul.close()
        
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        #percentile clipping (adjust to preference)
        vmin, vmax = np.percentile(data, (1, 99.9))
        data = np.clip(data, vmin, vmax)

        #normalize to 0–255 for display
        data -= data.min()
        data /= data.max()
        data *= 255

        img = Image.fromarray(data.astype(np.uint8))
        return img

