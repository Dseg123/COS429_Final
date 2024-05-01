import numpy as np
import time
import os
import sys
import torch
from tkinter import filedialog
import matplotlib.cm as cm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add path to ZoeDepth
sys.path.insert(0, "ZoeDepth")

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize
from PIL import Image
import torch
import open3d as o3d

model_zoe_n = torch.hub.load("ZoeDepth", "ZoeD_N", source="local", pretrained=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

img_names = os.listdir('data/images')

for i in range(len(img_names)):
    print(i)
    in_name = 'data/images/' + img_names[i]
    out_name = 'data/mono_depths/depth_' + str(i) + '.npy'
    image = Image.open(in_name).convert("RGB")  # load
    depth_numpy = zoe.infer_pil(image)
    np.save(out_name, depth_numpy)
