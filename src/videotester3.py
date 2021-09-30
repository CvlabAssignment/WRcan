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
import lxml.html
from lxml.html import HtmlElement
import re

class VideoTester3():
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

        timer_test = utility.timer()
        for idx_scale, scale in enumerate(self.scale):
            vidcap = cv2.VideoCapture(self.args.dir_demo)
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            # print('fps',vidcap.get(cv2.CAP_PROP_FPS))
            # _x{}
            vidwri = cv2.VideoWriter(
                self.ckp.get_path('{}_ex.avi'.format(self.filename, scale)),
                cv2.VideoWriter_fourcc(*'XVID'),
                vidcap.get(cv2.CAP_PROP_FPS),
                (
                    int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
            )
            f = open('frame_no_{}.txt'.format(self.filename), "wb")
            #  total_frames[26100:28530]
            tqdm_test = tqdm(range(total_frames), ncols=80)
            tree = ET.parse('../../카메라05_20200904170300_20200904172800_0_FrameInfo.xml')
            root = tree.getroot()
            child = root[0]
            l = []

            for t in tqdm_test:
                if t % 600 == 1:
                    for k in child[t]:
                        # print('k',child[t].attrib,k.tag)
                        if k.tag.startswith('H'):
                            # print('c,t', t, k.tag)
                            l.append(t)
                            ret, frame = vidcap.read()
                            vidwri.write(frame)
                            break
                        # if not ret: break
                # print('t',t)
                ret, frame = vidcap.read()
                if not ret: break


                # print(root.tag)
                # n = 0
                # for child in root[0]:
                # print(t)
                # if t > 26098 and t < 28529:
                    # print(t)

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

