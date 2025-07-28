from tkinter import *
from PIL import Image, ImageTk
from astropy.io import fits
import numpy as np
from gui.gui import ImageLoader

class ZoomViewer:
    def __init__(self, canvas, image, on_select, logger, input_dict, zoom_box=50, zoom_factor=4, selection_mode="Target Coordinates (pix)"):
        self.canvas = canvas
        self.image = image
        self.on_select = on_select
        self.zoom_box = zoom_box
        self.zoom_factor = zoom_factor
        self.zoom_window = Toplevel()
        self.zoom_window.title("Zoom")
        self.zoom_window.geometry(f"{zoom_box * zoom_factor}x{zoom_box * zoom_factor}")
        self.zoom_canvas = Canvas(self.zoom_window,
                                  width=zoom_box * zoom_factor,
                                  height=zoom_box * zoom_factor)
        self.zoom_canvas.pack()
        self.tk_zoom = None

        self.canvas.bind("<Motion>", self.update_zoom)
        self.canvas.bind("<Button-1>", self.store_click)
        
        #all of the coordinates needed to run pipeline, these are 400x400 coords, must map back to the original size. 
        #default mode is select target, then one can select validation and others within the 400x400 subsection. 
        self.selection_mode = selection_mode
        self.coordinates = {
            "Target Coordinates (pix)" : (None, None), 
            "Comparison Coordinates (pix)" : (None, None), 
            "Validation Coordinates (pix)" : (None, None)
        }
        self.cutout_x, self.cutout_y = None, None
        self.target_x, self.target_y = None, None
        self.comparison_x, self.comparison_y = None, None
        self.validation_x, self.validation_y = None, None
        self.logger = logger
        self.input_dict = input_dict
        self.target_radius = None
        self.fallback_radius = 15

    def update_zoom(self, event):
        x, y = event.x, event.y
        half = self.zoom_box // 2
        left = max(x - half, 0)
        upper = max(y - half, 0)
        right = min(x + half, self.image.width)
        lower = min(y + half, self.image.height)

        cropped = self.image.crop((left, upper, right, lower))
        zoomed = cropped.resize((self.zoom_box * self.zoom_factor,
                                 self.zoom_box * self.zoom_factor), Image.NEAREST)
        self.tk_zoom = ImageTk.PhotoImage(zoomed)
        self.zoom_canvas.create_image(0, 0, image=self.tk_zoom, anchor="nw")
        self.draw_aperture()

    def store_click(self, event):
        if self.selection_mode == "Target Coordinates (pix)":
            self.coordinates[self.selection_mode] = (event.x, event.y) 
            self.on_select(self.coordinates[self.selection_mode])
            #we must reset non-target selections, since an adjustment of target zoom area can lead to invalid previous selections (comp/vali)
            self.reset_non_target()
            self.update_view()
        if self.selection_mode == "Comparison Coordinates (pix)":
            if self.is_valid_event(event.x, event.y):
                self.coordinates[self.selection_mode] = (event.x, event.y)
                self.update_view()
        elif self.selection_mode == "Validation Coordinates (pix)":
            if self.is_valid_event(event.x, event.y):
                self.coordinates[self.selection_mode] = (event.x, event.y)
                self.update_view()

        log_message = f"PHOTOMETRY LOGGING: \nMode: {self.selection_mode}, \nMouse clicked at (x={event.x}, y={event.y}), \nValid Selection: {self.is_valid_event(event.x, event.y)}\n"
        self.logger.insert(END, log_message)
        
        try: 
            self.target_radius = int(self.input_dict["Target Radius"].get())
        except:
            self.logger.insert(END, f"Invalid Target Radius Input, Defaulting to {self.fallback_radius} pix\n")
        
    def is_valid_event(self, event_x, event_y) -> bool:
        zoom_center = self.coordinates["Target Coordinates (pix)"]
        return (all(coord is not None for coord in zoom_center) and 
                (zoom_center[0] - self.zoom_box//2 < event_x < zoom_center[0]  + self.zoom_box//2) and 
                (zoom_center[1] - self.zoom_box//2 < event_y < zoom_center[1]  + self.zoom_box//2))
        
    def set_mode(self, mode_label: str): 
        self.selection_mode = mode_label 
    
    def reset_non_target(self):
        self.coordinates["Comparison Coordinates (pix)"] = (None, None)
        self.coordinates["Validation Coordinates (pix)"] = (None, None)
        self.input_dict["Comparison Coordinates (pix)"].delete(0, END)
        self.input_dict["Validation Coordinates (pix)"].delete(0, END)

    def update_view(self): 
        entry = self.input_dict[self.selection_mode]
        entry.delete(0, END)
        entry.insert(0, f"{self.coordinates[self.selection_mode]}")

    def draw_aperture(self):
        try:
            radius = int(self.input_dict["Target Radius"].get())
        except:
            radius = self.fallback_radius

        if self.tk_zoom:
            self.zoom_canvas.delete('all')
            self.zoom_canvas.create_image(0, 0, image=self.tk_zoom, anchor="nw")
            
            cx, cy = self.tk_zoom.width() // 2, self.tk_zoom.height() // 2
            x0, y0 = cx - radius, cy - radius
            x1, y1 = cx + radius, cy + radius

            color = {
                "Target Coordinates (pix)": "white",
                "Comparison Coordinates (pix)": "blue",
                "Validation Coordinates (pix)": "yellow"
            }.get(self.selection_mode, "black")
            self.zoom_canvas.create_oval(x0, y0, x1, y1, outline=color, dash=(4, 2))

        
if __name__ == '__main__':
    root = Tk()
    canvas = Canvas(root, width=400, height=400)
    canvas.pack()

    img_load = ImageLoader()
    image_path = "/Users/spencerfreeman/Desktop/TransitForge/TestingNotebooks/test_image/Qatar-5b-0001_lrp.fit"  
    image = img_load.fits_to_image(image_path).resize((400, 400))
    photo = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, image=photo, anchor="nw")

    zoom_viewer = ZoomViewer(canvas, image)

    root.mainloop()
