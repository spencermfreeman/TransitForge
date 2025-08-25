from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.background import Background2D, SExtractorBackground
from photutils.detection import IRAFStarFinder
from photutils.segmentation import detect_sources
from photutils.utils import calc_total_error
from photutils.aperture import CircularAperture, aperture_photometry
from astropy.visualization import simple_norm
from astropy.nddata import CCDData, Cutout2D
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u
import os
from astropy.time import Time
from astropy.wcs import WCS
import astropy.table
import math

matplotlib.use('Agg')

class Photometry:
    GUI_IMAGE_SIZE = (400,400)
    CUTOUT_SIZE = (512, 512)
    def __init__(self,
                 output_dir: str, 
                 threshold_multiplier: float, 
                 catalogue_indicator: str, 
                 light_frame_indicator: str, 
                 target_radius: int):
        
        self.output_dir = output_dir
        self.threshold_multiplier = threshold_multiplier
        self.catalogue_indicator = catalogue_indicator
        self.light_frame_indicator = light_frame_indicator
        self.target_radius = target_radius
        self.radii = [radius for radius in range(self.target_radius-1, self.target_radius+2) if radius > 0]
        self.frame_index = 0
        self.frame_name = None
        
        #these should be the coordinates used when we take a 2d cutout with dimensions equal to that of the gui, centered on the target star from the user input
        self.target_coordinates_raw = (234, 203)
        self.comparison_coordinates_raw = (229, 221)
        self.validation_coordinates_raw = (249, 190)
        
        self.full_frame_shape = None  # (ny, nx)
        self.cutout_origin = None  # (x0, y0) top-left of cutout in full frame
        self.wcs = None  # WCS of full frame if present
        
        self.dataframe = None
        self.data = {"target" : [], 
                     "comparison" : [], 
                     "validation" : [],
                     "target_error" : [], 
                     "comparison_error" : [],
                     "validation_error" : [],
                     "jd" : []} 
    
    def background_subtract(self, file_path: str) -> tuple[CCDData, Background2D]: 
        data = CCDData.read(file_path, unit=u.adu)
        mean, _, std = sigma_clipped_stats(data, sigma=3.0)
        segm = detect_sources(data - mean, 3 * std, npixels=5)
        bool_array = segm.data != 0
        sigma_clip = SigmaClip(sigma=3.)
        bkg_estimator = SExtractorBackground()
        bkg = Background2D(data, 
                       box_size=(64, 64), 
                       mask=bool_array, 
                       filter_size=(3, 3), 
                       sigma_clip=sigma_clip, 
                       bkg_estimator=bkg_estimator)
        self._update_frame_metadata(data)
        self.file_name = os.path.basename(file_path)
        return CCDData(data - bkg.background.astype(float), unit=u.adu), bkg

    # MAIN OPTIMIZATION: this should take dimensions of the subsection that contains target and comparison stars.

    def _update_frame_metadata(self, ccd: CCDData):
        """Store full frame shape and WCS from CCDData for scaling/cutout."""
        self.full_frame_shape = ccd.data.shape  # (ny, nx)
        print(self.full_frame_shape)
        header = ccd.meta
        try:
            self.wcs = WCS(header)
        except Exception:
            self.wcs = None
            

    # def scale_frame_to_cutout(self, large_coordinates: tuple) -> tuple[float, float]:
    #     """
    #     Map full-frame coordinates to cutout-local coordinates.
    #     large_coordinates: (x, y) in full frame.
    #     Returns: (x_cutout, y_cutout)
    #     """
    #     if large_coordinates: 
    #         x_large, y_large = large_coordinates
    #         x_cutout, y_cutout = Photometry.CUTOUT_SIZE
    #         x_frame, y_frame = self.full_frame_shape
    #         x_small = int(x_large * (x_cutout/x_frame))
    #         y_small = int(y_large * (y_cutout/y_frame))
    #         return (x_small, y_small)
    #     else: return None
    
    
    def _scale_gui_coords_to_frames(self, small_coordinates: tuple) -> tuple[float, float]:
        """
        Map coordinates from the cutout frame (small, 400x400) back to full-frame coordinates (1200x1200).
        small_coordinates: (x, y) in cutout pixel space.
        Returns: (x_full, y_full)
        """
        
        if self.full_frame_shape:
            x_cutout, y_cutout = Photometry.GUI_IMAGE_SIZE
            x_frame, y_frame = self.full_frame_shape
            
            x_small, y_small = small_coordinates
            x_full = int(x_small * (x_frame/x_cutout))
            y_full = int(y_small * (y_frame/y_cutout))
            return (x_full, y_full)
        else: return None
    
    def get_cutout(self, image_bkg_sub: CCDData) -> Cutout2D: 
        coordinates = self._scale_gui_coords_to_frames(self.target_coordinates_raw)
        cutout = Cutout2D(image_bkg_sub, position=coordinates, size=Photometry.CUTOUT_SIZE)
        plt.imshow(cutout.data)
        plt.title('Cutout Data')
        plt.show()
        return cutout
    
    def plot_apertures_on_cutout(self, cutout, apertures, i, target_coord: tuple, comp_coord: tuple, vali_coord: tuple):
        """
        Pop up a matplotlib window showing the cutout with plotted apertures.
        """
        _, ax = plt.subplots(figsize=(6, 6))
        norm = simple_norm(cutout.data, 'sqrt', percent=99)
        ax.plot(target_coord[0], target_coord[1], 'o', color='yellow', markersize=6, label="Target")
        ax.plot(comp_coord[0], comp_coord[1], 'o', color='cyan', markersize=6, label="Comparison")
        ax.plot(vali_coord[0], vali_coord[1], 'o', color='magenta', markersize=6, label="Validation")
        ax.imshow(cutout.data, cmap='gray', norm=norm)
        for ap in apertures:
            ap.plot(color='red', lw=1.0, alpha=0.6, axes=ax)
    
        ax.legend(loc="upper right")
        ax.set_title("Detected Sources with Apertures")
        plt.savefig(f'/Users/spencerfreeman/Desktop/TransitForge/TestData/test_apertures{i}.png')
        
    #cutout should be bkg subtracted
    def cutout_to_table(self, cutout: Cutout2D, bkg: Background2D, frame: CCDData) -> astropy.table.QTable:
        """
        Detect sources in the cutout and perform aperture photometry.
        Returns a DataFrame with fixed columns:
        file_name, xcentroid, ycentroid,
        flux_r{r} and error_r{r} for r in [3,4,5]
        Each row corresponds to one detected source.
        """
        # Prepare empty dataframe schema
        cols = ['file_name', 'xcentroid', 'ycentroid']
        for r in self.radii:
            cols += [f'flux_r{r}', f'error_r{r}']
        df_empty = pd.DataFrame(columns=cols)

        threshold = (self.threshold_multiplier * bkg.background_rms_median).value
        star_finder = IRAFStarFinder(
            fwhm=3.0,
            threshold=threshold,
            exclude_border=True,
            sharplo=0.5,
            sharphi=2.0,
            roundlo=0.0,
            roundhi=0.7
        )
        
        sources = star_finder(cutout.data)
        if sources is None or len(sources) == 0:
            print(f"No sources detected in cutout for {self.file_name}.")
            return df_empty  # zero rows, fixed columns

        positions = list(zip(sources['xcentroid'], sources['ycentroid']))
        apertures = [CircularAperture(positions, r=r) for r in self.radii]
 
        
        # Extract background and its RMS for the same cutout region
        background = bkg.background
        background_rms = bkg.background_rms
        target_coordinates = self._scale_gui_coords_to_frames(self.target_coordinates_raw)
        bg_cutout = Cutout2D(background, position=tuple(target_coordinates), size=Photometry.CUTOUT_SIZE)
        bg_rms_cutout = Cutout2D(background_rms, position=tuple(target_coordinates), size=Photometry.CUTOUT_SIZE)

        gain = self._get_gain(frame)
        cutout.data = self._ensure_quantity(cutout.data, u.adu)
        bg_cutout.data = self._ensure_quantity(bg_cutout.data, u.adu)
        bg_rms_cutout.data = self._ensure_quantity(bg_rms_cutout.data, u.adu)
        
        flux_q = (cutout.data - bg_cutout.data) * gain
        bg_rms_q = bg_rms_cutout.data * gain     

        #strip units and call calc_total_error with effective_gain=1.0
        error = calc_total_error(flux_q.value, bg_rms_q.value, 1.0) * u.adu

        phot_table = aperture_photometry(cutout.data - bg_cutout.data, apertures, error=error)
        return phot_table, cutout.data, apertures
    
    def _get_gain(self, frame: CCDData):
        return frame.meta.get('GAIN', 1.0)
    
    def _ensure_quantity(self, quantity, unit):
        """
        Ensure q is an astropy Quantity in `unit`. If it's already a Quantity, convert;
        if it's plain array assume it's in `unit` and attach the unit.
        """
        if hasattr(quantity, 'unit'):
            return quantity.to(unit)
        else:
            return quantity * unit  # attach assumed unit
        
    def _best_coord_match(self, phot_table: astropy.table.QTable, coord: tuple) -> int:
        """
        Return the index of the phot_table entry that serves as the best match
        to the given coordinate (x, y).
        """
        # Zip x and y columns into list of tuples
        coords_list = list(zip(phot_table['xcenter'].value, phot_table['ycenter'].value))
        
        # Compute distances and find index of minimum
        distances = [self._distance(table_coord, coord) for table_coord in coords_list]
        
        coord_match_id = int(phot_table['id'][min(range(len(distances)), key=lambda i: distances[i])])
        coord_match_row = phot_table[phot_table['id'] == coord_match_id]
        shift = (float(coord_match_row['xcenter'].value - coord[0]), float(coord_match_row['ycenter'].value - coord[1]))
        return coord_match_id, shift

    def _distance(self, coord1: tuple, coord2: tuple) -> float: 
        return math.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)
    
    def _append_data(self, field: str, phot_table: astropy.table.QTable, id: int):
        match_rows = phot_table[phot_table['id'] == id]
        if len(match_rows) == 0:
            raise ValueError(f"No row found with id={id}")

        # Extract aperture_sum_2 value (assuming one match)
        value = match_rows['aperture_sum_2'][0]
        error = match_rows['aperture_sum_err_2'][0]
        
        self.data[field].append(value)
        self.data[f'{field}_error'].append(error)

    #pass small coords in 400x400 scale from gui event
    def _transform_coordinates(self, coord_type:str, coords_to_transform: tuple) -> tuple: 
        #scale 400x400 -> 4096x4096
        x_target_large, y_target_large = self._scale_gui_coords_to_frames(self.target_coordinates_raw) 
        x_transform_large, y_transform_large = self._scale_gui_coords_to_frames(coords_to_transform)
        
        #shift to new coordinates (cutout centered on target star, 400x400)
        cutout_center_x, cutout_center_y = self.CUTOUT_SIZE[0]//2, self.CUTOUT_SIZE[1]//2
        if coord_type == 'target': 
            return (cutout_center_x, cutout_center_y)
        else: 
            return (x_transform_large-x_target_large+cutout_center_x, y_transform_large-y_target_large+cutout_center_y)
    
    def _shift_coordinates(self, coord_type: str, coords_to_shift: tuple) -> tuple: 
        #shift the coordinates of the target based on the offset from 'coords to shift' and the best match in phot table
        #this method assumes the inital match is correct and helps prevent large shifts when processing the entire series of images
        pass
    
    def retrieve_data(self, phot_table: astropy.table.QTable):
        target_coords = self._transform_coordinates('target', self.target_coordinates_raw)
        id_target, error = self._best_coord_match(phot_table, target_coords)
        print("Target error: ", error)
        self._append_data("target", phot_table, id_target)

        comparison_coordinates = self._transform_coordinates('comparison', self.comparison_coordinates_raw)
        id_comparison, error = self._best_coord_match(phot_table, comparison_coordinates)
        self._append_data("comparison", phot_table, id_comparison)
        
        validation_coordinates = self._transform_coordinates('validation', self.validation_coordinates_raw)
        id_validation, error = self._best_coord_match(phot_table, validation_coordinates)
        self._append_data("validation", phot_table, id_validation)
        return target_coords, comparison_coordinates, validation_coordinates
        
    ######################################################################################################################################################################
    
    # def calc_shifts_update_catalogues(self):
    #     catalogue_files = glob.glob(f'{self.output_dir}/*cat.fits')
    #     catalogue_files.sort()
    #     x1 = []
    #     y1 = []
    #     reference_catalogue = Table.read(catalogue_files[0])
    #     for i, catalogue in enumerate(catalogue_files):
    #         if i == 0:
    #             reference_catalogue = Table.read(catalogue)
    #             x1 = reference_catalogue['xcenter']
    #             y1 = reference_catalogue['ycenter']
    #             if 'x_shift' not in reference_catalogue.colnames:
    #                 xcol = Table.Column(x1, name='x_shift')
    #                 ycol = Table.Column(y1, name='y_shift')
    #                 reference_catalogue.add_columns([xcol, ycol])
    #             else:
    #                 reference_catalogue['x_shift'] = x1
    #                 reference_catalogue['y_shift'] = y1
    #             reference_catalogue.write(catalogue, overwrite=True)
    #             print("Initial iteration complete, reference catalogue updated.")
    #         else:
    #             catalogue_of_interest = Table.read(catalogue)
    #             n_catalogue_OI = len(catalogue_of_interest)
    #             x2 = catalogue_of_interest['xcenter']
    #             y2 = catalogue_of_interest['ycenter']
    #             XX = []
    #             YY = []
    #             for j in range(n_catalogue_OI):
    #                 XX.extend((x1 - x2[j]))
    #                 YY.extend((y1 - y2[j]))
    #             XX = np.array(XX)
    #             YY = np.array(YY)
    #             xhist, xbins = np.histogram(XX, range=[-200, 200], bins=400)
    #             yhist, ybins = np.histogram(YY, range=[-200, 200], bins=400)
    #             idx = np.argmax(xhist)
    #             xshift_0 = (xbins[idx] + xbins[idx + 1]) / 2.0
    #             idx = np.argmax(yhist)
    #             yshift_0 = (ybins[idx] + ybins[idx + 1]) / 2.0
    #             print(f"Initial shift X (Iteration {i}): {xshift_0}, Initial shift Y: {yshift_0}")
    #             mask = (np.abs(XX - xshift_0) < 3) & (np.abs(YY - yshift_0) < 3)
    #             print("Mask sum: ", mask.sum())
    #             xshift_finetuned = np.median(XX[mask])
    #             yshift_finetuned = np.median(YY[mask])
    #             print(f"Finetuned Shift (Iteration {i}): ", xshift_finetuned, yshift_finetuned)
    #             if 'x_shift' not in catalogue_of_interest.colnames:
    #                 xcol = Table.Column(x2 + xshift_finetuned, name='x_shift')
    #                 ycol = Table.Column(y2 + yshift_finetuned, name='y_shift')
    #                 catalogue_of_interest.add_columns([xcol, ycol])
    #             else:
    #                 catalogue_of_interest['x_shift'] = x2 + xshift_finetuned
    #                 catalogue_of_interest['y_shift'] = y2 + yshift_finetuned
    #             catalogue_of_interest.write(catalogue, overwrite=True)

    # def _iso_date_to_JD(self, iso_date: str) -> float:
    #     iso_date = iso_date.replace("T", " ")
    #     t = Time(iso_date, format='iso')
    #     return t.jd

    # def calculate_light_curves(self, target_location: tuple, comparison_location: tuple, validation_location: tuple) -> tuple:
    #     catalogue_files = glob.glob(self.output_dir + f'/*{self.catalogue_indicator}*')
    #     processed_light_files = glob.glob(self.output_dir + f'/*{self.light_frame_indicator}*')
    #     catalogue_files.sort()
    #     processed_light_files.sort()
    #     n_files = len(catalogue_files)
    #     target_lc = np.zeros((1 + 2 * self.n_radii, n_files))
    #     comparison_lc = np.zeros((1 + 2 * self.n_radii, n_files))
    #     validation_lc = np.zeros((1 + 2 * self.n_radii, n_files))
    #     print("Calculating target, comparison, and validation light curves.")
    #     for i, file in enumerate(catalogue_files):
    #         header = fits.getheader(processed_light_files[i - 1])
    #         datestr = self._iso_date_to_JD(header['DATE-OBS'])
    #         target_lc[0, i] = datestr
    #         comparison_lc[0, i] = datestr
    #         validation_lc[0, i] = datestr
    #         catalogue = fits.getdata(file)
    #         x_source_locations = catalogue['x_shift']
    #         y_source_locations = catalogue['y_shift']
    #         target_distance_array = np.sqrt((x_source_locations - target_location[0])**2 + (y_source_locations - target_location[1])**2)
    #         minimum_index = np.argmin(target_distance_array)
    #         target_aper_arr = catalogue[minimum_index]
    #         for j in range(self.n_radii):
    #             target_lc[j + 1, i] = target_aper_arr['aperture_sum_' + str(j)]
    #             target_lc[self.n_radii + j + 1, i] = target_aper_arr['aperture_sum_err_' + str(j)]
    #         comparison_distance_array = np.sqrt((x_source_locations - comparison_location[0])**2 + (y_source_locations - comparison_location[1])**2)
    #         minimum_index_comp = np.argmin(comparison_distance_array)
    #         comparison_aper_arr = catalogue[minimum_index_comp]
    #         for j in range(self.n_radii):
    #             comparison_lc[j + 1, i] = comparison_aper_arr['aperture_sum_' + str(j)]
    #             comparison_lc[self.n_radii + j + 1, i] = comparison_aper_arr['aperture_sum_err_' + str(j)]
    #         validation_distance_array = np.sqrt((x_source_locations - validation_location[0])**2 + (y_source_locations - validation_location[1])**2)
    #         minimum_index_vali = np.argmin(validation_distance_array)
    #         validation_aper_arr = catalogue[minimum_index_vali]
    #         for j in range(self.n_radii):
    #             validation_lc[j + 1, i] = validation_aper_arr['aperture_sum_' + str(j)]
    #             validation_lc[self.n_radii + j + 1, i] = validation_aper_arr['aperture_sum_err_' + str(j)]
    #     return (target_lc, comparison_lc, validation_lc)

    # def get_plotting_data(self, target_lc: list, comparison_lc: list, validation_lc: list, iaper: int) -> tuple:
    #     iaper = 6
    #     lc_target_plot = target_lc[iaper + 1, :] / comparison_lc[iaper + 1, :]
    #     lc_validation_plot = validation_lc[iaper + 1, :] / comparison_lc[iaper + 1, :]
    #     a1 = 1.0 / comparison_lc[iaper + 1, :]; e1 = target_lc[iaper + 10 + 1, :]
    #     a2 = target_lc[iaper + 1, :] / comparison_lc[iaper + 1, :]**2; e2 = comparison_lc[iaper + 10 + 1, :]
    #     error_target = np.sqrt(a1**2 * e1**2 + a2**2 * e2**2)
    #     a1 = 1.0 / comparison_lc[iaper + 1, :]; e1 = validation_lc[iaper + 10 + 1, :]
    #     a2 = validation_lc[iaper + 1, :] / comparison_lc[iaper + 1, :]**2; e2 = comparison_lc[iaper + 10 + 1, :]
    #     error_valiation = np.sqrt(a1**2 * e1**2 + a2**2 * e2**2)
    #     print('photerr for target/comparison:', np.median(error_target))
    #     print('photerr for validation/comparison:', np.median(error_valiation))
    #     idx = np.argmin(np.abs(target_lc[0, :] - 51888.67))
    #     norm_targ = np.median(lc_target_plot[idx:])
    #     norm_vali = np.median(lc_validation_plot[idx:])
    #     time_axis_range = [np.min(target_lc[0, :]), np.max(target_lc[0, :])]
    #     return lc_target_plot, lc_validation_plot, error_target, error_valiation, norm_targ, norm_vali, time_axis_range

    # def plot_light_curve(self, jd: list, lc_target_plot: list, lc_validation_plot: list, comparison_lc: list, norm_targ: float,
    #                      norm_vali: float, iaper: int, time_axis_range: list, transit_name: str, date: str, observer_name: str):
    #     plt.figure(figsize=(16, 16))
    #     plt.scatter(jd[0, :], lc_target_plot / (1.005 * norm_targ), color='black', marker='o', s=10, label="Target")
    #     plt.scatter(jd[0, :], lc_validation_plot / norm_vali - 0.02, color='magenta', marker='o', s=10, label="Validation Star")
    #     plt.plot(time_axis_range, [1.0, 1.0], 'b-', linewidth=0.5)
    #     plt.plot(time_axis_range, [0.980, 0.980], 'b-', linewidth=0.5)
    #     plt.ylim([0.970, 1.01])
    #     plt.xlabel("\nJulian Date (JD)", fontsize=15)
    #     plt.ylabel("Relative Flux", fontsize=15)
    #     plt.grid(True)
    #     plt.suptitle(f"\n  {transit_name}", fontsize=30)
    #     plt.title(f"{date}\nObserver: {observer_name}\n", fontsize=12)
    #     plt.legend()
    #     plt.savefig(self.output_dir + f"/{transit_name}")
    #     plt.show()
    #     self.to_csv(jd[0, :], lc_target_plot, lc_validation_plot, comparison_lc, norm_targ, norm_vali, iaper, f"{transit_name}_measurements")
    #     print(sigma_clipped_stats(2.5 * np.log10(lc_validation_plot), sigma=3, maxiters=3))

    # def to_csv(self, jd: list, target_lc: list, validation_lc: list, comparison_lc: list, norm_target: float,
    #            norm_vali: float, iaper: int, filename: str):
    #     time_axis = np.array(jd)
    #     target_lc = np.array(target_lc) / norm_target
    #     validation_lc = np.array(validation_lc) / norm_vali
    #     comparison_lc = np.array(comparison_lc[iaper + 1, :])
    #     data = {'Time Axis': time_axis, 'Target Flux': target_lc, 'Validation Flux': validation_lc, 'Comparison Flux': comparison_lc}
    #     df = pd.DataFrame(data)
    #     file_path = os.path.join(self.output_dir, filename)
    #     df.to_csv(file_path, index=False)
    #     print(f"CSV file created at: {file_path}")

    # def perform_photometry(self, target_location: tuple, comparison_location: tuple, validation_location: tuple) -> tuple:
    #     self.write_photometric_catalogues()
    #     self.calc_shifts_update_catalogues()
    #     target_lc, comparison_lc, validation_lc = self.calculate_light_curves(target_location, comparison_location, validation_location)
    #     return (target_lc, comparison_lc, validation_lc)

    # def plot(self, target_lc: list, comparison_lc: list, validation_lc: list, iaper: int, transit_name: str,
    #          date: str, observer_name: str) -> None:
    #     lc_target_plot, lc_validation_plot, error_target, error_valiation, norm_targ, norm_vali, time_axis_range = self.get_plotting_data(target_lc, comparison_lc, validation_lc, iaper)
    #     self.plot_light_curve(target_lc, lc_target_plot, lc_validation_plot, comparison_lc, norm_targ, norm_vali, iaper, time_axis_range, transit_name, date, observer_name)