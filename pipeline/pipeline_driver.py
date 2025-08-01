from data_load import DataLoader
from calibration import Calibration
from photometry import Photometry

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
                                               True, #TODO: Add to interface
                                               self.entries['Transit Name'],
                                               self.entries['Bias Indicator'], 
                                               self.entries['Flat Indicator'], 
                                               self.entries['Light Frame Indicator'], 
                                               self.entries['Data Directory'])
            master_flat, master_bias = self.calibration.create_master_frames()
            self.calibration.calibrate_light_frames(master_flat, master_bias, ('15:00:00','15:00:00'))
            
            #Perform Photometry
            if self.calibration.output_directory:
                self.data_loader = DataLoader(self.calibration.output_directory)
                reduced_light_frames = self.data_loader.get_science_paths(self.entries['Light Frame Indicator'])
                output_data = 
                print(reduced_light_frames)
                
            #Plot Results
        else: 
            print('Missing Entries')

if __name__ == '__main__': 
    driver = PipelineDriver({'Data Directory' : '/Users/spencerfreeman/Desktop/TransitForge/TestData',
                             'Transit Name' : 'Qatar-5b', 
                             'Light Frame Indicator' : 'lrp', 
                             'Bias Indicator' : 'bias', 
                             'Flat Indicator' : 'frp'})
    driver.run_pipeline()