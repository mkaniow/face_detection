'''
This file resizes every image in folder to one rezolution (WIDTH x HEIGHT)

WIDTH - width of resized image in pixels
HEIGHT - height of resized image in pixels
'''

from PIL import Image
import os, sys

WIDTH = 1280
HEIGHT = 720

def resize(width, height):
    for item in os.listdir('data/images'):
        if os.path.isfile(os.path.join('data', 'images', item)):
            im = Image.open(os.path.isfile(os.path.join('data', 'images', item)))
            im.resize((width, height), Image.ANTIALIAS)

resize(WIDTH, HEIGHT)