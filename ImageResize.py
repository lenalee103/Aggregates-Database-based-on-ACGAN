from PIL import Image
from numpy import array
#from pylab import *
import os

print(os.getcwd())

from PIL import Image
import os, sys

path = os.path.join(os.getcwd(),'16mm1000_Angularity')
dirs = os.listdir(path)

def imgcrop():
    for item in dirs:
        if 'tif' in item:
            im = Image.open(path+'/' +item)
            f, e = os.path.splitext(path+item)
            imgcrop = im.crop((300,300,1200,1000))
            imgcrop.save(f + e, 'JPEG', quality=90)
        else: continue

imgcrop()
