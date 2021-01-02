import os, json
import cv2
from tqdm import tqdm

split_type = 'val'
annot_path = f'dataset_bh/{split_type}/annotation'
image_dir_path = f'dataset_bh/{split_type}/frames'

f = open(os.path.join('dataset','final_anns.json')).read()

js = json.loads(f)

count = len(list(js.keys()))

for key, value in tqdm(js.items(), total=count): # (imagename, [annots])
    im_path = os.path.join(image_dir_path, key)
    txt_path = os.path.join(annot_path, key[:-4]) + ".txt"
    # print(im_path)
    im = cv2.imread(im_path)
    if im is None:
        continue
    h, w, colour = im.shape
    f = open(txt_path, 'w')
    for annot in value:
        cat_id = annot['cat_id']
        bbox = annot['bbox']
        # writing annotations to file

        # [x1,y1,x2,y2] -> left, top, right, bottom

        x_center, y_center, width, height = (bbox[0]+bbox[2]/2)/(w), (bbox[1]+bbox[3]/2)/(h), bbox[2]/w, bbox[3]/h
        print(f'{cat_id - 1} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}', file=f)
        # print(f'{cat_id} {bbox[0]:.2f} {bbox[1]:.2f} {bbox[0]+bbox[2]:.2f} {bbox[1]+bbox[3]:.2f}', file=f)
    f.close()
