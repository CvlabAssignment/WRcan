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

class VideoTesterd():
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
        file_path = '/home/cvlab3/Downloads/WRCAN-PyTorch/experiment/deblur_jpeg/wrcan/original/'
        file_names = glob.glob(file_path+'*.avi')

        # file_names2 = ['카메라03_20200904170300_20200904172800_0','카메라06_20200904170300_20200904172800_0','카메라17_20200904170300_20200904172800_0']
        file_names2 =[]
        for f in file_names:
            f = f.split("/")
            file_names2.append(f[-1][:-4])

        timer_test = utility.timer()
        # for idx_scale, scale in enumerate(self.scale):
        for idx ,file in enumerate(file_names2):

            video_file = file_path+file+".avi"
            vidcap = cv2.VideoCapture(video_file)

            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            # print(file,total_frames)

            vidwri = cv2.VideoWriter(
                self.ckp.get_path('./deblurred/{}_x1.avi'.format(file, self.scale)),
                cv2.VideoWriter_fourcc(*'XVID'),
                vidcap.get(cv2.CAP_PROP_FPS),
                (
                    int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
            )

            tqdm_test = tqdm(range(total_frames), ncols=80)

            f1 = open('frame_no_{}.txt'.format(file), "rb")
            f = pickle.load(f1)

            for t in tqdm_test:
                ret, lr = vidcap.read()
                if not ret: break

                lr, = common.set_channel(lr, n_channels=self.args.n_colors)
                lr, = common.np2Tensor(lr, rgb_range=self.args.rgb_range)
                lr, = self.prepare(lr.unsqueeze(0))
                sr = self.model(lr, 0)
                sr = utility.quantize(sr, self.args.rgb_range).squeeze(0)

                normalized = sr * 255 / self.args.rgb_range
                ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
                vidwri.write(ndarr)

                cv2.imwrite("./images_x1/{}|{}.jpeg".format(file, f[t]),ndarr)


            # pickle.dump(l,f)
            vidcap.release()
            vidwri.release()

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

