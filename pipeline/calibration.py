from astropy.stats import mad_std
from astropy import units as u
from astropy.nddata import CCDData
from pathlib import Path
from astropy.coordinates import SkyCoord
import ccdproc as ccdp
import numpy as np
import datetime
import shutil

class Calibration:
    def __init__(self, science_frames:list, 
                 flip: bool, 
                 transit_name: str, 
                 bias_indication: str, 
                 flat_indication: str, 
                 light_indication: str,
                 data_directory: str):
        
        self.science_frames = science_frames
        self.flip = flip
        self.transit_name = transit_name
        self.bias_indication = bias_indication
        self.flat_indication = flat_indication
        self.light_indication = light_indication
        self.output_directory = None
        self.directory = data_directory

    @staticmethod
    def inv_median(a) -> int:
        return 1 / np.median(a)
    
    def make_subdir(self) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        directory_name = f"Pipeline_Session_{timestamp}_Output"
        self.output_directory = Path(self.directory) / directory_name
        
        if self.output_directory.exists():
            shutil.rmtree(self.output_directory)

        self.output_directory.mkdir(parents=True)
        return self.output_directory

    def create_master_frames(self) -> tuple[CCDData, CCDData]:
        files = ccdp.ImageFileCollection(filenames=self.science_frames)

        biases = files.files_filtered(imagetyp='Bias Frame', include_path=True)
        flats = files.files_filtered(imagetyp='Flat Field', include_path=True)

        calibrated_data = self.make_subdir()

        master_bias = self.process_calibration(biases, calibrated_data, self.bias_indication)
        master_flat = self.process_calibration(flats, calibrated_data, self.flat_indication)
        del files
        return master_bias, master_flat
    
    def process_calibration(self,
                calibration_list: list,
                output_path: Path,
                calibration_image_type: str) -> CCDData | None:
               
                frame_data = [CCDData.read(file, unit=u.adu) for file in calibration_list]
                if not frame_data:
                    return None

                master_frame = ccdp.combine(frame_data,
                                            method='average',
                                            sigma_clip=True,
                                            sigma_clip_low_thresh=5,
                                            sigma_clip_high_thresh=5,
                                            sigma_clip_func=np.ma.median,
                                            sigma_clip_dev_func=mad_std,
                                            mem_limit=500e6)

                master_frame.meta['combined'] = True
                master_frame.data = master_frame.data.astype('float32')
                
                if self.flip:
                    master_frame.data = np.flipud(master_frame.data)
                master_frame.write(output_path / f"master_{calibration_image_type}.fit")
                return master_frame
        
    def calibrate_light_frames(self, 
                               master_bias: CCDData, 
                               master_flat: CCDData, 
                               target_coords_wcs: tuple):
        
        files = ccdp.ImageFileCollection(filenames=self.science_frames)
        # info read from the fits header:
        focal_length = self.get_focal_length_collection(files) # mm
        pixel_size = self.get_pixel_size(files)  # um
        
        #TODO: why is this hardcoded (isnt it related to CCD and located in fits headaer?)
        pixscale = 206.265 * (pixel_size / focal_length)
        c = SkyCoord(target_coords_wcs[0], 
                     target_coords_wcs[1], 
                     frame='icrs', 
                     unit=(u.hourangle, u.degree))
        ra = c.ra.degree
        dec = c.dec.degree
        gain, readout_noise = self.get_gain_readout_noise()
        
        lights = files.files_filtered(imagetyp='Light Frame', include_path=True)
        light_frames = []
        for i,light in enumerate(lights):
            print(f"Reducing light frame {i+1}/{len(lights)}")
            light_frame = CCDData.read(light, unit=u.adu)
            reduced = ccdp.ccd_process(light_frame, 
                                       master_bias=master_bias, 
                                       master_flat=master_flat)
            self.edit_header(reduced, ra, dec, gain, readout_noise)
            if self.flip:
                reduced.data = np.flipud(reduced.data)
            reduced.data = reduced.data.astype('float32')
            light_frames.append(reduced)
            reduced.write(self.output_directory / f"{self.transit_name.strip('/')}_{i+1}_reduced_{self.light_indication}_.fit", 
                          overwrite=True)
        del files
        return light_frames
    
    def get_focal_length_collection(self, files: ccdp.ImageFileCollection) -> float | None:
        for hdu, _ in files.hdus(return_fname=True):
            header = hdu.header
            for key in ['FOCALLEN', 'FOCLEN', 'FOCUSLNG', 'TELFOCUS']:
                if key in header:
                    return float(header[key])
            break #check first file, all are assumed to have the same header.
        return None
    
    def get_pixel_size(self, files: ccdp.ImageFileCollection) -> float | None:
        for hdu, _ in files.hdus(return_fname=True):
            header = hdu.header
            for key in ['XPIXSZ', 'YPIXSZ', 'PIXELSIZE']:
                if key in header:
                    return float(header[key]) #microns
            break 
        return None
    
    def get_gain_readout_noise(self) -> tuple:
        main_path = Path(self.directory)
        files = ccdp.ImageFileCollection(main_path)
        
        biases = files.files_filtered(imagetyp='Bias Frame', include_path=True)
        flats = files.files_filtered(imagetyp='Flat Field', include_path=True)
        
        mean_bias, std_bias = self.process_gain_readout(biases)
        mean_flat, std_flat = self.process_gain_readout(flats)
       
        gain = (mean_flat - mean_bias) / (std_flat**2 - std_bias**2)
        readnoise = gain * std_bias / np.sqrt(2)
        print(f"gain: {gain}, readout noise: {readnoise}")
        return (gain, readnoise)
    
    def process_gain_readout(self, 
                             calibration_list: list) -> tuple[float, float]:
        frame_data = [CCDData.read(file, unit=u.adu)[1500-256:1500+256, 1500-256:1500+256] for file in calibration_list]
        frame_combined = np.median(frame_data, axis=0)
        return np.mean(frame_combined), np.std(frame_combined) 
        
    def edit_header(self, reduced, ra, dec, gain, readout_noise):
        reduced.meta['epoch'] = 2000.0
        reduced.meta['CRVAL1'] = ra
        reduced.meta['CRVAL2'] = dec
        reduced.meta['CRPIX1'] = reduced.meta['NAXIS1'] / 2.0
        reduced.meta['CRPIX2'] = reduced.meta['NAXIS2'] / 2.0
        reduced.meta['CTYPE1'] = 'RA---TAN'
        reduced.meta['CTYPE2'] = 'DEC--TAN'
        reduced.meta['GAIN'] = (gain, 'GAIN in e-/ADU')
        reduced.meta['RDNOISE'] = (readout_noise, 'readout noise in electrons')