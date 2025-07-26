from tkinter import *
from PIL import Image, ImageTk
from astropy.io import fits
import numpy as np
from gui.gui import ImageLoader

class ZoomViewer:
    def __init__(self, canvas, image, on_select, logger, zoom_box=50, zoom_factor=4, selection_mode="Target Coordinates (pix)"):
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
        self.zoom_image = None
        self.tk_zoom = None

        self.canvas.bind("<Motion>", self.update_zoom)
        self.canvas.bind("<Button-1>", self.store_click)
        
        #all of the coordinates needed to run pipeline, these are 400x400 coords, must map back to the original size. 
        #default mode is select target, then one can select validation and others within the 400x400 subsection. 
        self.selection_mode = selection_mode
        self.cutout_x, self.cutout_y = None, None
        self.target_x, self.target_y = None, None
        self.comparison_x, self.comparison_y = None, None
        self.validation_x, self.validation_y = None, None
        self.logger = logger

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

    def store_click(self, event):
        if self.selection_mode == "Target Coordinates (pix)":
            self.cutout_x, self.target_x = event.x, event.y
            self.cutout_y, self.target_y = event.y, event.y
            self.on_select(self.cutout_x, self.cutout_y)
            #we must reset non-target selections, since an adjustment of target zoom area can lead to invalid previous selections (comp/vali)
            self.reset_non_target()
        elif self.is_valid_event(event.x, event.y):
            if self.selection_mode == "Comparison Coordinates (pix)":
                self.comparison_x = event.x
                self.comparison_y = event.y
            elif self.selection_mode == "Comparison Coordinates (pix)":
                self.validation_x = event.x
                self.validation_y = event.y 
        
        log_message = f"PHOTOMETRY LOGGING: \nMode: {self.selection_mode}, \nMouse clicked at (x={self.cutout_x}, y={self.cutout_y}), \nValid Selection: {self.is_valid_event(event.x, event.y)}\n"
        self.logger.insert(END, log_message)
        
    def is_valid_event(self, event_x, event_y) -> bool:
        return (event_x > self.cutout_x - self.zoom_box//2 and event_x < self.cutout_x  + self.zoom_box//2 and 
                event_y > self.cutout_y - self.zoom_box//2 and event_y < self.cutout_y  + self.zoom_box//2)
        
    def set_mode(self, mode_label: str): 
        self.selection_mode = mode_label 
    
    def reset_non_target(self):
        self.comparison_x, self.comparison_y = None, None
        self.validation_x, self.validation_y = None, None

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
