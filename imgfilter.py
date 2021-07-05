# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 09:51:58 2021

@author: Kritika Srivastava
"""

import sys
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import cv2
import os
import numpy as np
import pygame
import pygame.gfxdraw
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
from skimage import io 
from collections import defaultdict
import glob
import shutil


def brighten(input_image_path, output_image_path):
    img = Image.open(input_image_path)
    img2 = ImageEnhance.Brightness(img).enhance(1.5)
    img2.show()
    img2.save(output_image_path)
    print("Brightened Image saved")
    return


def saturate(input_image_path, output_image_path):
    img = Image.open(input_image_path)
    converter = ImageEnhance.Color(img)
    img2 = converter.enhance(1.5)
    img2.show()
    img2.save(output_image_path)
    print("Saturated image saved in folder.")
    return


def polyart(input_image_path, output_image_path):
    inp = pygame.surfarray.pixels3d(pygame.image.load(input_image_path))
    perceptual_weight = np.array([0.2126, 0.7152, 0.0722])
    grayscale = (inp * perceptual_weight).sum(axis=-1)
    x = gaussian_filter(grayscale, 2, mode="reflect")
    x2 = gaussian_filter(grayscale, 30, mode="reflect")
    diff = x - x2
    diff[diff < 0] *= 0.1
    diff = np.sqrt(np.abs(diff) / diff.max())
    diff = x - x2
    diff[diff < 0] *= 0.1
    diff = np.sqrt(np.abs(diff) / diff.max())

    def sample(ref, n=1000000):
        np.random.seed(0)
        w, h = x.shape
        xs = np.random.randint(0, w, size=n)
        ys = np.random.randint(0, h, size=n)
        value = ref[xs, ys]
        accept = np.random.random(size=n) < value
        points = np.array([xs[accept], ys[accept]])
        return points.T, value[accept]

    def get_colour_of_tri(tri, image):
        colours = defaultdict(lambda: [])
        w, h, _ = image.shape
        for i in range(0, w):
            for j in range(0, h):
                index = tri.find_simplex((i, j))
                colours[int(index)].append(inp[i, j, :])
        for index, array in colours.items():
            colours[index] = np.array(array).mean(axis=0)
        return colours

    def draw(tri, colours, screen, upscale):
        s = screen.copy()
        for key, c in colours.items():
            t = tri.points[tri.simplices[key]]
            pygame.gfxdraw.filled_polygon(s, t * upscale, c)
            pygame.gfxdraw.polygon(s, t * upscale, c)
        return s

    w, h, _ = inp.shape
    upscale = 2
    screen = pygame.Surface((w * upscale, h * upscale))
    screen.fill(inp.mean(axis=(0, 1)))
    corners = np.array([(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)])
    samples, v = sample(diff)
    plt.scatter(
        samples[:, 0], -samples[:, 1], c=v, s=0.2, edgecolors="none", cmap="viridis"
    )
    points = np.concatenate((corners, samples))

    outdir = "lowpoly/output/"
    os.makedirs(outdir, exist_ok=True)

    for i in range(0, 100):
        n = 5 + i + 2 * int(i ** 2)
        tri = Delaunay(points[:n, :])
        colours = get_colour_of_tri(tri, inp)
        s = draw(tri, colours, screen, upscale)
        s = pygame.transform.smoothscale(s, (w, h))
        pygame.image.save(s, f"lowpoly/output/{i:04d}.png")

    frames = []
    imgs = glob.glob("lowpoly/output/*.png")
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(
        output_image_path,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=100,
        loop=0,
    )
    shutil.rmtree("lowpoly")
    print("Poly art saved in folder.")
    return


def cartoonise(input_image_path, output_image_path):
    img = cv2.imread(input_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    getEdge = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 255)
    cartoon = cv2.bitwise_and(color, color, mask=getEdge)
    io.imshow(cartoon)
    cv2.imwrite(output_image_path, cartoon)
    print("Cartoonised image saved in the folder.")
    return


def neg(input_image_path, output_image_path):
    im = Image.open(input_image_path).convert("RGB")
    im_invert = ImageOps.invert(im)
    im_invert.save(output_image_path)
    im_invert.show()
    print("Negative image saved in the folder")
    return


def pencilsketch(input_image_path, output_image_path):
    img = Image.open(input_image_path)
    img1 = img.filter(ImageFilter.CONTOUR)
    img1.show()
    img1.save(output_image_path)
    print("Pencil Sketch image saved in the folder.")
    return


def blur(input_image_path, output_image_path):
    OriImage = Image.open(input_image_path)
    OriImage.show()
    blurImage = OriImage.filter(ImageFilter.BLUR)
    blurImage.show()
    blurImage.save(output_image_path)
    print("Blurred Image saved in the folder")
    return


def border(image_path, output_image_path):
    img = Image.open(image_path)
    img_with_border = ImageOps.expand(img, border=300, fill="white")
    img_with_border.save(output_image_path)
    img_with_border.show()
    print("Image with border saved in the folder.")
    return


def bnw(input_image_path, output_image_path):
    img = Image.open(input_image_path).convert("LA")
    img.convert('RGB').save(output_image_path)
    img.show()
    print("Black and white image saved in the folder.")
    return


def sepia(image_path, output_image_path):
    img = Image.open(image_path)
    width, height = img.size
    pixels = img.load()  # create the pixel map
    for py in range(height):
        for px in range(width):
            r, g, b = img.getpixel((px, py))
            tr = min(int(0.393 * r + 0.769 * g + 0.189 * b), 255)
            tg = min(int(0.349 * r + 0.686 * g + 0.168 * b), 255)
            tb = min(int(0.272 * r + 0.534 * g + 0.131 * b), 255)
            pixels[px, py] = (tr, tg, tb)
    img.show()
    img.save(output_image_path)
    print("Sepia image saved in the folder.")
    return


filters = {
    0: "Exit",
    1: "Black and white",
    2: "Sepia",
    3: "White Border",
    4: "Blur",
    5: "Pencil Sketch",
    6: "Negative",
    7: "Cartoonise",
    8: "Poly Art",
    9: "Brighten",
    10: "Saturate",
}
for k in sorted(filters):
    print(k, ". ", filters[k])
i = 1
while i and i in filters:
    s = input("Input Fname: ")
    i = int(input("Enter Filter: "))
    if os.path.isfile(s):
        if i in filters:
            if i == 1:
                bnw(s, "bw_" + s)
            elif i == 2:
                sepia(s, "sepia_" + s)
            elif i == 3:
                border(s, "border_" + s)
            elif i == 4:
                blur(s, "blur_" + s)
            elif i == 5:
                pencilsketch(s, "sketch_" + s)
            elif i == 6:
                neg(s, "neg_" + s)
            elif i == 7:
                cartoonise(s, "cartoon_" + s)
            elif i == 8:
                opp = "polyart_" + s.split(".")[0] + ".gif"
                polyart(s, opp)
            elif i == 9:
                brighten(s, "bright_" + s)
            elif i == 10:
                saturate(s, "saturate_" + s)
            else:
                sys.exit()
        else:
            print("Enter a valid option.")
    else:
        print("Enter a valid filename in the next go.")
        continue
sys.exit()
