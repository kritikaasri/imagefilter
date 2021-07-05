# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 17:13:45 2021

@author: Kritika Srivastava
"""

import os
from PIL import Image
import glob

frames = []
imgs = glob.glob("lowpoly/output/*.png")
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
 
# Save into a GIF file that loops forever
frames[0].save('png_to_gif.gif', format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)