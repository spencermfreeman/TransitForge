from data_load import DataLoader
from calibration import Calibration
from photometry import Photometry
import io

class PipelineDriver:
    def __init__(self, entries: dict):
        self.entries = entries
        self.data_loader = DataLoader(self.entries['Data Directory'])
        self.calibration = None
        self.photometry = None
    
    def run_pipeline(self): 
        if all([value for value in self.entries.values()]) is not None:
            #Load Data
            science_frames = self.data_loader.get_science_paths()
            
            #Calibrate Data
            self.calibration = Calibration(science_frames,
                                               self.entries['Flip Frames'],
                                               self.entries['Transit Name'],
                                               self.entries['Bias Indicator'], 
                                               self.entries['Flat Indicator'], 
                                               self.entries['Light Frame Indicator'], 
                                               self.entries['Data Directory'])
            master_flat, master_bias = self.calibration.create_master_frames()
            self.calibration.calibrate_light_frames(master_flat, master_bias, self.entries['RA/DEC'])
            
            #Perform Photometry
            if self.calibration.output_directory:
                self.data_loader = DataLoader(self.calibration.output_directory)
                reduced_light_frames = self.data_loader.get_science_paths(self.entries['Light Frame Indicator'])
                self.photometry = Photometry(self.calibration.output_directory, 
                                             self.entries['Threshold Multiplier'], 
                                             self.entries['Catalogue Indicator'], 
                                             self.entries['Light Frame Indicator'], 
                                             self.entries['Target Radius'])
                
                for reduced_light_frame in reduced_light_frames: 
                    frame_ccd_data, bkg = self.photometry.background_subtract(reduced_light_frame)
                    cutout_ccd_data = self.photometry.get_cutout(frame_ccd_data)
                    photometry_table = self.photometry.cutout_to_table(cutout_ccd_data, bkg, frame_ccd_data)
                    self.photometry.get_data(photometry_table)
                    print(self.photometry.data)
                    break
            #Plot Results
        else: 
            print('Missing Entries')
            
if __name__ == '__main__': 
    driver = PipelineDriver({'Flip Frames' : False,
                             'Data Directory' : '/Users/spencerfreeman/Desktop/TransitForge/TestData',
                             'Transit Name' : 'Qatar-5b', 
                             'Light Frame Indicator' : 'lrp', 
                             'Bias Indicator' : 'bias', 
                             'Flat Indicator' : 'frp', 
                             'RA/DEC' : ('15:00:00', '14:00:00'), 
                             'Threshold Multiplier' : 5, 
                             'Catalogue Indicator' : 'cat', 
                             'Target Radius' : 6})
    driver.run_pipeline()