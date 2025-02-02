from IPython.display import Image
import os
import re
from PIL import Image

DATA_PATH = '/NN/xerdelsky/NSIETE_project/DatasetCloudyRainy/'
FRAME_PATH = DATA_PATH + 'Snimky/'
MASK_PATH = DATA_PATH + 'Mask/'

folders = ['train_frames/train', 'train_masks/train', 'val_frames/val', 'val_masks/val', 'test_frames/test',
           'test_masks/test']
for folder in folders:
    os.makedirs(DATA_PATH + folder)

all_frames = os.listdir(FRAME_PATH)
all_masks = os.listdir(MASK_PATH)

all_frames.sort(key=lambda var: [int(x) if x.isdigit() else x
                                 for x in re.findall(r'[^0-9]|[0-9]+', var)])
all_masks.sort(key=lambda var: [int(x) if x.isdigit() else x
                                for x in re.findall(r'[^0-9]|[0-9]+', var)])

train_split = int(0.7 * len(all_frames))
val_split = int(0.9 * len(all_frames))

train_frames = all_frames[:train_split]
val_frames = all_frames[train_split:val_split]
test_frames = all_frames[val_split:]

train_masks = [f for f in all_masks if f in train_frames]
val_masks = [f for f in all_masks if f in val_frames]
test_masks = [f for f in all_masks if f in test_frames]


def add_frames(dir_name, image):
    img = Image.open(FRAME_PATH + image)
    img.save(DATA_PATH + '/{}'.format(dir_name) + '/' + image)


def add_masks(dir_name, image):
    img = Image.open(MASK_PATH + image)
    img.save(DATA_PATH + '/{}'.format(dir_name) + '/' + image)


frame_folders = [(train_frames, 'train_frames/train'), (val_frames, 'val_frames/val'),
                 (test_frames, 'test_frames/test')]

mask_folders = [(train_masks, 'train_masks/train'), (val_masks, 'val_masks/val'),
                (test_masks, 'test_masks/test')]

for folder in frame_folders:
    array = folder[0]
    name = [folder[1]] * len(array)

    list(map(add_frames, name, array))

for folder in mask_folders:
    array = folder[0]
    name = [folder[1]] * len(array)

    list(map(add_masks, name, array))
