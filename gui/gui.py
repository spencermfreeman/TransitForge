import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import ImageTk
import threading
import os
import ast
import glob
from functools import partial
from pipeline.calibration import Calibration
from pipeline.photometry import Photometry
from gui.image_load import ImageLoader

CONFIG_PATH = os.path.join("guide", "config.txt")

CONFIG_SECTIONS = {
    "File I/O": [
        "Main Directory",
        "Output Directory",
        "Light Frame Indicator",
        "Bias Indicator",
        "Flat Indicator",
        "Catalogue Indicator",
        "Target Name"
    ],
    "Photometry Related": [
        "RA/DEC",
        "Target Radius",
        "Target Coordinates (pix)",
        "Comparison Coordinates (pix)",
        "Validation Coordinates (pix)",
        "Source Detection Threshold"
    ],
    "Plotting Information": [
        "Main Plot Title (transit name)",
        "Observation Date (MM/DD/YYYY)",
        "Observer Name"
    ]
}

def load_config():
    config = {}
    if not os.path.exists(CONFIG_PATH):
        return config
    current_section = None
    with open(CONFIG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("---") and line.endswith("---"):
                current_section = line.strip("- ").strip()
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                config[key.strip()] = value.strip()
    return config

def save_config(config):
    lines = []
    for section, keys in CONFIG_SECTIONS.items():
        lines.append(f"--- {section} ---")
        for key in keys:
            if key in config:
                value = config[key].strip()
                if "Directory" in key:
                    value = value.rstrip("/\\")
                lines.append(f"{key}: {value}")
        lines.append("")
    with open(CONFIG_PATH, "w") as f:
        f.write("\n".join(lines))

def run_pipeline_manual(config_dict, log_callback):
    try:
        log_callback("Using manual configuration input...")
        wcs_val = config_dict["RA/DEC"]
        if isinstance(wcs_val, str) and ',' in wcs_val:
            wcs = [s.strip() for s in wcs_val.split(',')]
        else:
            wcs = wcs_val
        target_coords = ast.literal_eval(config_dict["Target Coordinates (pix)"])
        comparison_coords = ast.literal_eval(config_dict["Comparison Coordinates (pix)"])
        validation_coords = ast.literal_eval(config_dict["Validation Coordinates (pix)"])
        main_directory = config_dict["Main Directory"]
        output_dir = config_dict["Output Directory"]
        transit_name = config_dict["Target Name"]
        target_radius = int(config_dict["Target Radius"])
        threshold_multiplier = float(config_dict["Source Detection Threshold"])
        catalogue_indicator = config_dict["Catalogue Indicator"]
        light_frame_indicator = config_dict["Light Frame Indicator"]
        main_title = config_dict["Main Plot Title (transit name)"]
        date = config_dict["Observation Date (MM/DD/YYYY)"]
        observer_name = config_dict["Observer Name"]

        log_callback("Starting calibration...")
        calib = Calibration(main_directory, 
                            flip=True, 
                            transit_name=transit_name,
                            bias_indication="bias", 
                            flat_indication="frp", 
                            light_indication=light_frame_indicator)
        master_bias, master_flat = calib.create_master_frames()
        log_callback("Master frames created.")
        calib.calibrate_light_frames(master_bias, master_flat, wcs)
        log_callback("Light frames calibrated.")

        log_callback("Performing photometry...")
        photom = Photometry(output_dir, threshold_multiplier, catalogue_indicator, light_frame_indicator, n_radii=target_radius)
        target_lc, comparison_lc, validation_lc = photom.perform_photometry(target_coords, comparison_coords, validation_coords)
        log_callback("Photometry completed.")

        log_callback("Generating plots and CSV outputs...")
        photom.plot(target_lc, comparison_lc, validation_lc, target_radius, transit_name, date, observer_name)
        log_callback("Pipeline completed successfully!")
    except Exception as e:
        log_callback(f"Error (GUI): {str(e)}")

class AstroPipelineGUI(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding="10")
        self.master = master
        master.title("TransitForge - Configuration")
        master.geometry("950x800")
        self.pack(fill=tk.BOTH, expand=True)
        self.entries = {}
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        file_io_frame = ttk.LabelFrame(notebook, text="File I/O")
        notebook.add(file_io_frame, text="File I/O")
        
        self.data_directory = None
        self.frames = []
        self.current_index = 0
    
        self.create_file_io_section(file_io_frame)
        photo_frame = ttk.LabelFrame(notebook, text="Photometry")
        notebook.add(photo_frame, text="Photometry")
        
        self.create_photometry_section(photo_frame)
        plot_frame = ttk.LabelFrame(notebook, text="Plotting")
        notebook.add(plot_frame, text="Plotting")

        self.create_plotting_section(plot_frame)
        self.load_config_into_fields()

    
    def create_file_io_section(self, parent):
        fields = [
            ("Main Directory", "directory"),
            ("Output Directory", "directory"),
            ("Light Frame Indicator", "text"),
            ("Bias Indicator", "text"),
            ("Flat Indicator", "text"),
            ("Catalogue Indicator", "text"),
            ("Target Name", "text")
        ]
        for i, (label_text, field_type) in enumerate(fields):
            lbl = ttk.Label(parent, text=label_text + ":")
            lbl.grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            entry = ttk.Entry(parent, width=50)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries[label_text] = entry
            if field_type == "directory":
                btn = ttk.Button(parent, text="Browse", command=lambda e=entry: self.browse_directory(e))
                btn.grid(row=i, column=2, padx=5, pady=5)
                
        # ScrolledText log
        self.log_text = scrolledtext.ScrolledText(parent, width=100, height=15, state=tk.NORMAL)
        self.log_text.grid(row=99, column=0, columnspan=3, pady=(10, 5), sticky="nsew")

        # Run Pipeline button now inside File I/O tab
        run_btn = ttk.Button(parent, text="Run Pipeline", command=self.start_pipeline)
        run_btn.grid(row=100, column=0, columnspan=3, pady=(0, 10))
        run_btn.grid_configure(sticky="n")
        
        parent.grid_rowconfigure(99, weight=1)
        parent.grid_columnconfigure(0, weight=1)

    ''' Photometry Tab Related '''
    
    def create_photometry_section(self, parent):
        # Explicit reference for Data Directory
        lbl = ttk.Label(parent, text="Data Directory:")
        lbl.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

        self.data_dir_entry = ttk.Entry(parent, width=50)
        self.data_dir_entry.grid(row=0, column=1, padx=5, pady=5)
        self.entries["Data Directory"] = self.data_dir_entry

        btn_browse = ttk.Button(parent, text="Browse", command=partial(self.browse_directory, self.data_dir_entry))
        btn_browse.grid(row=0, column=2, padx=5, pady=5)

        btn_load = ttk.Button(parent, text="Load Images", command=self.load_images)
        btn_load.grid(row=1, column=1, columnspan=2, padx=5, pady=5)

        # Photometry fields
        fields = [
            ("RA/DEC", "text"),
            ("Target Radius", "text"),
            ("Target Coordinates (pix)", "text"),
            ("Comparison Coordinates (pix)", "text"),
            ("Validation Coordinates (pix)", "text"),
            ("Source Detection Threshold", "text")
        ]
        for i, (label_text, _) in enumerate(fields):
            lbl = ttk.Label(parent, text=label_text + ":")
            lbl.grid(row=i+2, column=0, sticky=tk.W, padx=5, pady=5)
            entry = ttk.Entry(parent, width=50)
            entry.grid(row=i+2, column=1, padx=5, pady=5)
            self.entries[label_text] = entry
            
        # Placeholder for image display
        self.image_label = ttk.Label(parent)
        self.image_label.grid(row=10, column=0, columnspan=3, pady=10, sticky="n")
        
        nav_frame = ttk.Frame(parent)
        nav_frame.grid(row=11, column=0, columnspan=3, pady=5, sticky="n")

        self.prev_button = ttk.Button(nav_frame, text="←", command=self.show_prev_image)
        self.next_button = ttk.Button(nav_frame, text="→", command=self.show_next_image)
        self.image_counter = ttk.Label(nav_frame, text="(0/0)")

        self.prev_button.pack(side="left", padx=10)
        self.image_counter.pack(side="left", padx=10)
        self.next_button.pack(side="left", padx=10)

        # Enable left/right arrow key bindings
        parent.bind_all("<Left>", lambda e: self.show_prev_image())
        parent.bind_all("<Right>", lambda e: self.show_next_image())

    def load_images(self):
        """Loads images from the data directory."""
        data_dir = self.entries["Data Directory"].get()
        if not os.path.isdir(data_dir):
            messagebox.showerror("Error", "Please select a valid data directory.")
            return

        fits_files = sorted(glob.glob(os.path.join(data_dir, "*.fit*")))
        if not fits_files:
            messagebox.showinfo("No Images", "No FITS files found in the selected directory.")
            return

        image_load = ImageLoader()
        self.frames = [
            (ImageTk.PhotoImage(image_load.fits_to_image(f).resize((400, 400))),f)
            for f in fits_files
        ]

        self.image_index = 0
        self.show_current_image()

    def show_current_image(self):
        if not self.frames:
            return
        current_image = self.frames[self.image_index][0]
        self.image_label.config(image=current_image)
        self.image_label.image = current_image
        self.image_counter.config(text=self.get_image_counter_text())

    def show_prev_image(self):
        if hasattr(self, "frames") and self.frames:
            self.image_index = (self.image_index - 1) % len(self.frames)
            self.show_current_image()

    def show_next_image(self):
        if hasattr(self, "frames") and self.frames:
            self.image_index = (self.image_index + 1) % len(self.frames)
            self.show_current_image()

    def get_image_counter_text(self):
        if hasattr(self, "frames") and self.frames:
            return f"{os.path.basename(self.frames[self.image_index][1])} ({self.image_index + 1} / {len(self.frames)})"
        return "(0/0)"

    ''' Plotting Section '''
    def create_plotting_section(self, parent):
        fields = [
            ("Main Plot Title (transit name)", "text"),
            ("Observation Date (MM/DD/YYYY)", "text"),
            ("Observer Name", "text")
        ]
        for i, (label_text, _) in enumerate(fields):
            lbl = ttk.Label(parent, text=label_text + ":")
            lbl.grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            entry = ttk.Entry(parent, width=50)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries[label_text] = entry

    def browse_directory(self, entry_widget):
        directory = filedialog.askdirectory(title="Select Directory")
        if directory:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, directory)

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def collect_config(self):
        config = {}
        for key, entry in self.entries.items():
            value = entry.get().strip()
            if not value:
                self.log(f"Warning: '{key}' is empty.")
            config[key] = value
        return config

    def load_config_into_fields(self):
        config = load_config()
        for key, entry in self.entries.items():
            if key in config:
                entry.delete(0, tk.END)
                entry.insert(0, config[key])

    def start_pipeline(self):
        config_dict = self.collect_config()
        save_config(config_dict)
        required_keys = [
            "Main Directory", "Output Directory", "Light Frame Indicator",
            "Catalogue Indicator", "Target Name", "RA/DEC", "Target Radius",
            "Target Coordinates (pix)", "Comparison Coordinates (pix)",
            "Validation Coordinates (pix)", "Source Detection Threshold",
            "Main Plot Title (transit name)", "Observation Date (MM/DD/YYYY)",
            "Observer Name"
        ]
        missing = [k for k in required_keys if not config_dict.get(k)]
        if missing:
            messagebox.showerror("Missing Fields", f"The following fields are required:\n{', '.join(missing)}")
            return
        self.log("Starting pipeline process...")
        thread = threading.Thread(target=run_pipeline_manual, args=(config_dict, self.log))
        thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = AstroPipelineGUI(root)
    root.mainloop()
