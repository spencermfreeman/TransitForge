import tkinter as tk
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gui.gui import AstroPipelineGUI

root = tk.Tk()
app = AstroPipelineGUI(root)
root.mainloop()
