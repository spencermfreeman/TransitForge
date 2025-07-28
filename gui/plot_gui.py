import matplotlib.pyplot as plt
import numpy as np

class Plot:
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['font.family'] = 'Georgia'

    def __init__(self, input_dict: dict, timeline: list, target_flux_rel_norm: list, validation_flux_norm: list):
        self.input_dict = input_dict
        self.timeline = timeline
        self.target_flux_rel_norm = target_flux_rel_norm
        self.validation_flux_norm = validation_flux_norm

    def generate_fig(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=60)

        ax.scatter(self.timeline, self.target_flux_rel_norm, color='black', marker='o', s=5, label="Target")
        ax.scatter(self.timeline, self.validation_flux_norm, color='magenta', marker='o', s=5, label="Validation Star")
        ax.plot([self.timeline[0], self.timeline[-1]], [1.0, 1.0], 'b-', linewidth=0.5)

        ax.set_xlabel("Julian Date (JD)", fontsize=12)
        ax.set_ylabel("Relative Flux", fontsize=12)

        # Titles and subtitles
        # title = self.input_dict.get("Main Plot Title (Transit Name)", "").get()
        # observer = self.input_dict.get("Observer Name", "").get()
        # date = self.input_dict.get("Observation Date (MM/DD/YYYY)", "").get()
        
        fig.suptitle("Qatar 5b", fontsize=16)
        ax.set_title("11/15/2025\nObserver: Spencer Freeman", fontsize=10)

        ax.grid(True)
        ax.legend()
        ax.tick_params(colors='black')

        for spine in ax.spines.values():
            spine.set_color('black')

        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle

        return fig


if __name__ == '__main__': 
    target_flux = np.random.normal(1, 0.01, 100)
    comp_flux = np.random.normal(1, 0.01, 100)
    timeline = np.arange(len(target_flux))
    plotter = Plot({}, timeline, target_flux, comp_flux)
    plotter.generate_fig()