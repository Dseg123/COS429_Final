import os
import random
import numpy as np

env_dir = '/home/dylaneg/Documents/Programming/COS429/COS429_Final/data/environment1'
depth_dir = env_dir + '/depths'
image_dir = env_dir + '/images'

#get num_samples image paths and depth paths from environment env_name
#all samples will have directory numbers within range radius
def get_paths(env_name, num_samples, radius = 0):
    env_dir = f'/home/dylaneg/Documents/Programming/COS429/COS429_Final/data/{env_name}'
    depth_dir = env_dir + '/depths'
    image_dir = env_dir + '/images'
    depth_paths = sorted(os.listdir(depth_dir))
    image_paths = sorted(os.listdir(image_dir))

    if radius <= 0 or radius >= len(depth_paths):
        radius = len(depth_paths)

    inds = [i for i in range(radius)]
    selects = random.sample(inds, num_samples)
    start = random.randint(0, len(depth_paths) - radius)
    selects = sorted([x + start for x in selects])
    
    img_selects = [f'{image_dir}/image_{x}.png' for x in selects]
    dep_selects = [f'{depth_dir}/depth_{x}.npy' for x in selects]

    return img_selects, dep_selects

if __name__ == "__main__":
    print(get_paths('environment1', 10, 50))
    