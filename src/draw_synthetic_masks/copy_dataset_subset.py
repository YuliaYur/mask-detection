from glob import glob
import shutil

from tqdm import tqdm
import numpy as np


np.random.seed(0)


paths = sorted(glob('../dataset/VGG-Face2/data/crops/**/*.jpg'))
chosen_paths = np.random.choice(paths, size=18000)

for i, p in tqdm(enumerate(chosen_paths)):
    shutil.copyfile(p, '../dataset/VGG-Face2/data/for_vgg/no_mask/' + f'{i}.jpg')
