import os
import math
import pickle

import utility
from data import common

import torch
import cv2

from tqdm import tqdm
from PIL import Image
import xml.etree.ElementTree as ET
import re
import glob

class VideoTester13_15():
    def __init__(self, args, my_model, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.model = my_model
        self.filename, _ = os.path.splitext(os.path.basename(args.dir_demo))

    def test(self):
        torch.set_grad_enabled(False)

        self.ckp.write_log('\nEvaluation on video:')
        self.model.eval()
        file_path = '/data1/KIST/data/20200904_KIST_DATA/'
        # file_names = glob.glob(file_path+'*.mp4')
        # file_names2 = []
        # for f in file_names:
        #     f = f.split("/")
        #     file_names2.append(f[-1][:-4]) '카메라13_20200904170300_20200904172800_0',
        file_names2 =['카메라15_20200904170300_20200904172800_0']

        timer_test = utility.timer()
        # for idx_scale, scale in enumerate(self.scale):
        for idx ,file in enumerate(file_names2):
            video_file = file_path+file+".mp4"
            vidcap = cv2.VideoCapture(video_file)
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(video_file,total_frames)

            vidwri = cv2.VideoWriter(
                self.ckp.get_path('./original/{}.avi'.format(file, self.scale)),
                cv2.VideoWriter_fourcc(*'XVID'),
                vidcap.get(cv2.CAP_PROP_FPS),
                (
                    int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
            )
            f = open('frame_no_{}.txt'.format(file), "wb")
            tqdm_test = tqdm(range(total_frames), ncols=80)
            tree = ET.parse('/data1/KIST/data/20200904_KIST_DATA/{}_FrameInfo.xml'.format(file))
            root = tree.getroot()
            child = root[0]
            l = []
            n = 0
            for t in tqdm_test:
                if t % 30 == 1:
                    for k in child[t-1]:
                        print(t+1, child[t].attrib)
                        if k.tag.startswith('H'):
                            l.append(t)
                            ret, frame = vidcap.read()
                            vidwri.write(frame)
                            # n = t
                            break

                ret, frame = vidcap.read()
                if not ret: break

                # vidwri.write(frame)
            pickle.dump(l,f)
            vidcap.release()
            vidwri.release()

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        torch.set_grad_enabled(True)

    # def prepare(self, *args):
    #     device = torch.device('cpu' if self.args.cpu else 'cuda')
    #     def _prepare(tensor):
    #         if self.args.precision == 'half': tensor = tensor.half()
    #         return tensor.to(device)
    #
    #     return [_prepare(a) for a in args]

