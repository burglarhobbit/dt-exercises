import os
import shutil
import numpy as np

source11 = "dataset_bh/train/annotation"
source12 = "dataset_bh/train/frames"
dest11 = "dataset_bh/val/annotation"
dest12 = "dataset_bh/val/frames"
files1 = os.listdir(source11)
files2 = [frame.replace("annotation", "frames")[:-4] + ".png" for frame in files1]

for f1,f2 in zip(files1, files2):
    rand = np.random.rand(1)
    print(rand)
    if rand < 0.2:
        shutil.move(source11 + '/'+ f1, dest11 + '/'+ f1)
        shutil.move(source12 + '/'+ f2, dest12 + '/'+ f2)