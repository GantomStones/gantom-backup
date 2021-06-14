import torch
import os
from glob import glob
import json
import numpy as np
from PIL import Image, ImageOps
from functions import Shape_classifier, getImageShape, getImageRGB, getCodeInfo
from operator import itemgetter




model = torch.load('py-files/model.pth', map_location=torch.device('cpu'))




# Running Images through model to get prediction and save to_json dictionary
for i in sorted(glob('backup/*')):
    try:
        img = Image.open(i+'/image.png')
    except:
        print(i)  

    try:
        img = img.resize((256,256))
        shape = getImageShape(img, model)
        top_5 = getImageRGB(img)
        codeInfo = list(getCodeInfo(i+'/metadata.json'))

        to_json = {'shape': shape, 'top 5 RGB':top_5, 'code info': codeInfo}

        with open(i+'/rarity_qualities.json', 'w') as f:
            json.dump(to_json, f, indent=3)

    except:
        print('ERROR:', i)

  






    

