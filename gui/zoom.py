from tkinter import *
from PIL import Image, ImageTk
from astropy.io import fits
import numpy as np
from gui.gui import ImageLoader

class ZoomViewer:
    def __init__(self, canvas, image, zoom_box=100, zoom_factor=4):
        self.canvas = canvas
        self.image = image
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
        
        self.cutout_x, self.cutout_y = 0, 0 

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
        self.cutout_x = event.x
        self.cutout_y = event.y
        print(f"Mouse clicked at (x={event.x}, y={event.y})")

        
# Main app
if __name__ == '__main__':
    root = Tk()
    canvas = Canvas(root, width=400, height=400)
    canvas.pack()

    img_load = ImageLoader()
    image_path = "/Users/spencerfreeman/Desktop/TransitForge/TestingNotebooks/test_image/Qatar-5b-0001_lrp.fit"  # Replace with your image file
    #pass the resized image
    image = img_load.fits_to_image(image_path).resize((400, 400))
    photo = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, image=photo, anchor="nw")

    zoom_viewer = ZoomViewer(canvas, image)

    root.mainloop()
