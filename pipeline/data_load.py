import glob
import os

class DataLoader: 
    
    def __init__(self, fp_to_input_data):
        self.path = fp_to_input_data
    
    def get_science_paths(self, indicator) -> list:
        files = glob.glob(os.path.join(self.path, f'*{indicator}*.fit*')) 
        return files
    
    
if __name__ == '__main__':
    fp_to_input = '/Users/spencerfreeman/Desktop/pipeline_gui/test_data'
    data_loader = DataLoader(fp_to_input)
    print(data_loader.get_science_paths('frp'))