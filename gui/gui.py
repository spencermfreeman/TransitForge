import os
import glob
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from PIL import Image, ImageTk
from astropy.io import fits
import numpy as np

class ImageLoader:
    def fits_to_image(self, path):
        hdul = fits.open(path)
        data = hdul[0].data
        hdul.close()

        norm_data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
        img = Image.fromarray(norm_data.astype(np.uint8))
        return img

class AstroPipelineGUI(ttk.Frame):
    def __init__(self, root):
        self.root = root
        self.root.title("TransitForge GUI")
        self.root.geometry("1000x900")  # Bigger initial size

        self.entries = {}
        self.frames = []
        self.current_frame_index = 0

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill="both")

        self.create_tabs()

    def create_tabs(self):
        self.file_io_tab = ttk.Frame(self.notebook)
        self.photometry_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.file_io_tab, text="File I/O")
        self.notebook.add(self.photometry_tab, text="Photometry")
        self.notebook.add(self.results_tab, text="Plotting/Results")

        self.create_file_io_section(self.file_io_tab)
        self.create_photometry_section(self.photometry_tab)
        self.create_results_section(self.results_tab)
        
    def browse_directory(self, entry):
        directory = filedialog.askdirectory()
        if directory:
            entry.delete(0, tk.END)
            entry.insert(0, directory)
            
    ''' file i/o section'''
    
    def create_file_io_section(self, parent: ttk.Frame):
        parent.grid_columnconfigure(1, weight=1)

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
            lbl.grid(row=i, column=0, sticky="e", padx=5, pady=5)
            entry = ttk.Entry(parent, width=50)
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="ew")
            self.entries[label_text] = entry
            if field_type == "directory":
                btn = ttk.Button(parent, text="Browse", command=lambda e=entry: self.browse_directory(e))
                btn.grid(row=i, column=2, padx=5, pady=5)

        self.log_text = scrolledtext.ScrolledText(parent, width=100, height=15, state=tk.NORMAL)
        self.log_text.grid(row=99, column=0, columnspan=3, pady=(10, 5), sticky="nsew")

        run_btn = ttk.Button(parent, text="Run Pipeline", command=self.start_pipeline)
        run_btn.grid(row=100, column=0, columnspan=3, pady=(0, 10))

        parent.grid_rowconfigure(99, weight=1)
        parent.grid_columnconfigure(1, weight=1)
    
    ''' photometry section '''
    
    def create_photometry_section(self, parent: ttk.Frame):
        parent.grid_columnconfigure(1, weight=1)

        ttk.Label(parent, text="Data Directory:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        entry = ttk.Entry(parent, width=50)
        entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.entries["Data Directory"] = entry

        ttk.Button(parent, text="Browse", command=lambda e=entry: self.browse_directory(e)).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(parent, text="Load Images", command=lambda e=entry: self.load_images(e)).grid(row=1, column=1, padx=5, pady=5)

        fields = [
            ("RA/DEC", "text"),
            ("Target Radius", "text"),
            ("Target Coordinates (pix)", "text"),
            ("Comparison Coordinates (pix)", "text"),
            ("Validation Coordinates (pix)", "text"),
            ("Source Detection Threshold", "text")
        ]

        for i, (label_text, _) in enumerate(fields):
            ttk.Label(parent, text=label_text + ":").grid(row=i+2, column=0, sticky="e", padx=5, pady=5)
            entry = ttk.Entry(parent, width=50)
            entry.grid(row=i+2, column=1, padx=5, pady=5, sticky="ew")
            self.entries[label_text] = entry

        # Placeholder image
        placeholder = Image.new("L", (400, 400), color=200)
        self.frames = [(ImageTk.PhotoImage(placeholder), "No Files Loaded")]
        self.image_label = ttk.Label(parent, image=self.frames[0][0])
        self.image_label.image = self.frames[0][0]
        self.image_label.grid(row=10, column=0, columnspan=3, pady=10)

        nav_frame = ttk.Frame(parent)
        nav_frame.grid(row=11, column=0, columnspan=3, pady=5)

        self.prev_button = ttk.Button(nav_frame, text="←", command=self.show_prev_image)
        self.next_button = ttk.Button(nav_frame, text="→", command=self.show_next_image)
        self.image_counter = ttk.Label(nav_frame, text=self.get_image_counter_text())

        self.prev_button.pack(side="left", padx=10)
        self.image_counter.pack(side="left", padx=10)
        self.next_button.pack(side="left", padx=10)

        parent.bind_all("<Left>", lambda e: self.show_prev_image())
        parent.bind_all("<Right>", lambda e: self.show_next_image())

    def load_images(self, entry_widget):
        directory = entry_widget.get().strip()
        if not directory:
            return

        image_load = ImageLoader()
        fits_files = sorted(glob.glob(os.path.join(directory, "*lrp.fit*")))
        self.frames = [(ImageTk.PhotoImage(image_load.fits_to_image(f).resize((400, 400))), f) for f in fits_files]

        if self.frames:
            self.current_frame_index = 0
            self.image_label.configure(image=self.frames[0][0])
            self.image_label.image = self.frames[0][0]
            self.image_counter.configure(text=self.get_image_counter_text())

    def get_image_counter_text(self):
        if not self.frames:
            return "0 / 0"
        return f"{os.path.basename(self.frames[self.current_frame_index][1])} ({self.current_frame_index + 1} / {len(self.frames)})"

    def show_prev_image(self):
        if self.frames and self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.image_label.configure(image=self.frames[self.current_frame_index][0])
            self.image_label.image = self.frames[self.current_frame_index]
            self.image_counter.configure(text=self.get_image_counter_text())

    def show_next_image(self):
        if self.frames and self.current_frame_index < len(self.frames) - 1:
            self.current_frame_index += 1
            self.image_label.configure(image=self.frames[self.current_frame_index][0])
            self.image_label.image = self.frames[self.current_frame_index]
            self.image_counter.configure(text=self.get_image_counter_text())

    ''' results/plotting '''
    
    def create_results_section(self, parent: ttk.Frame):
        parent.grid_columnconfigure(1, weight=1)
         
        fields = [
            ("Main Plot Title (Transit Name)", "text"),
            ("Observation Date (MM/DD/YYYY)", "text"),
            ("Observer Name", "text"),
        ]
        
        for i, (label_text, _) in enumerate(fields):
            ttk.Label(parent, text=label_text + ":").grid(row=i+2, column=0, sticky="e", padx=5, pady=5)
            entry = ttk.Entry(parent, width=50)
            entry.grid(row=i+2, column=1, padx=5, pady=5, sticky="ew")
            self.entries[label_text] = entry
    
    def start_pipeline(self):
        self.log_text.insert(tk.END, "[INFO] Pipeline started...\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = AstroPipelineGUI(root)
    root.mainloop()
