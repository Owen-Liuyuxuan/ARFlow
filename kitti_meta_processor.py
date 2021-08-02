import numpy as np
import os
import imageio
import cv2

META_TRAIN_FILE = "train_files.txt"

def read_split_file(file:str):
    imdb:list = []
    with open(file, 'r') as f:
        lines = f.readlines() # 2011_09_26/2011_09_26_drive_0022_sync 473 r
        for i  in range(len(lines)):
            line = lines[i].strip().split()

            folder = line[0]
            index = int(line[1])
            side = line[2]
            datetime = folder.split("/")[0]
            imdb.append(
                dict(
                    folder=folder,
                    index=index,
                    side=side,
                    datetime=datetime
                )
            )
    return imdb

class SimpleDataset:
    def __init__(self, meta_train_file, raw_path, frame_idxs):
        print(f"Start loading split file {meta_train_file}")
        self.imdb = read_split_file(meta_train_file) # list of dict
        self.raw_path = raw_path
        self.frame_idxs = frame_idxs
        self.num_images = len(self.imdb)
        print(f"Load split file {meta_train_file} done")
        print(f"Total images: {self.num_images}")

    def __getitem__(self, i):
        obj = self.imdb[i]

        folder  = obj['folder']
        index   = obj['index']
        side    = obj['side']
        datetime= obj['datetime']

        side = 'l'
        data = dict()
        for idx in self.frame_idxs:
            data[("image", idx)] = self.get_color(folder, index + idx, side)

        return data

    def __len__(self):
        return self.num_images

    def get_color(self, folder, frame_index, side):
        camera_folder = {"l": "image_02", "r" : "image_03"}[side]
        image_dir = os.path.join(self.raw_path, folder, camera_folder, 'data', '%010d.png'%frame_index)

        return imageio.imread(image_dir).astype(np.float32)

def kitti_save_flow(flow, flow_file):
    h, w, c = flow.shape
    flow_array = np.zeros(shape=(h, w, 3), dtype=np.uint16)
    flow_array[:, :, 0] = (flow[:, :, 0] * 64 + 2 ** 15).astype(np.uint16)
    flow_array[:, :, 1] = (flow[:, :, 1] * 64 + 2 ** 15).astype(np.uint16)

    cv2.imwrite(flow_file, flow_array)

