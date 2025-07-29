import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
import io

class Plot:
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    suptitle_size = 11
    plt.rcParams['font.family'] = 'Georgia'
    plt.rcParams['text.usetex'] = False
    
    suptitle_size = 16
    
    def __init__(self, input_dict: dict, timeline: list, target_flux_rel_norm: list, validation_flux_norm: list):
        self.input_dict = input_dict
        self.timeline = timeline
        self.target_flux_rel_norm = target_flux_rel_norm
        self.validation_flux_norm = validation_flux_norm

    def generate_fig(self) -> plt.Figure:
        ''' General Plotting '''
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        
        title = self.extract_input("Main Plot Title (Transit Name)")
        observer = self.extract_input("Observer Name")
        date = self.extract_input("Observation Date (MM/DD/YYYY)")
        
        ax.scatter(self.timeline, self.target_flux_rel_norm, color='black', marker='o', s=3, label=f"Target Star: {title}")
        ax.scatter(self.timeline, self.validation_flux_norm, color='magenta', marker='o', s=3, label="Validation Star")
        ax.plot([self.timeline[0], self.timeline[-1]], [1.0, 1.0], 'b-', linewidth=0.5)


        ''' General Configuration '''
        if title and observer and date:
            fig.suptitle(f"  {title}", fontsize=Plot.suptitle_size)
            
        ax.set_title(f"{date}\nObserver(s): {observer}")
        ax.set_xlabel("Julian Date (JD)")
        ax.set_ylabel("Relative Flux, Normalized")
     
        ax.grid(True)
        ax.legend()
        ax.tick_params(colors='black')

        for spine in ax.spines.values():
            spine.set_color('black')

        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        return fig

    def extract_input(self, key: str) -> str:
        value = self.input_dict.get(key, "")
        return value.get() if hasattr(value, "get") else str(value)

    def save_fig_image(self, fig: plt.Figure) -> None:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=700)
        buffer.seek(0)
        image = Image.open(buffer)
        return image
    
if __name__ == '__main__': 
    target_flux = np.random.normal(1, 0.01, 100)
    comp_flux = np.random.normal(1, 0.01, 100)
    timeline = np.arange(len(target_flux))
    plotter = Plot({}, timeline, target_flux, comp_flux)
    plotter.generate_fig()
    plt.show()