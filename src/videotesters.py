import os
import math
import pickle

import numpy as np

import utility
from data import common

import torch
import cv2

from tqdm import tqdm

class VideoTesters():
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
        split = {}
        for idx_scale, scale in enumerate(self.scale):
            vidcap = cv2.VideoCapture(self.args.dir_demo)
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            print('total', total_frames)
            # print('width', int(scale * vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            # print('scale', scale) scale *
            # vidwri = cv2.VideoWriter(
            #     self.ckp.get_path('{}_x{}2.avi'.format(self.filename, scale)),
            #     cv2.VideoWriter_fourcc(*'XVID'),
            #     vidcap.get(cv2.CAP_PROP_FPS),
            #     (
            #         int(vidcap.get(scale * cv2.CAP_PROP_FRAME_WIDTH)),
            #         int(vidcap.get(scale * cv2.CAP_PROP_FRAME_HEIGHT))
            #     )
            # )
            # print(self.ckp.get_path('{}_x{}2.avi'.format(self.filename, scale)))
            tqdm_test = tqdm(range(total_frames), ncols=80)
            frames = []
            pids = []
            for t in tqdm_test:
                success, lr = vidcap.read()
                if not success: break
                cv2.imwrite("./images5ox1/{}.jpeg".format(t), lr)
                pids.append(t+1)
                frames.append(lr)
                # lr, = common.set_channel(lr, n_channels=self.args.n_colors)
                # lr, = common.np2Tensor(lr, rgb_range=self.args.rgb_range)
                # # print('lr',lr.unsqueeze(0))
                # lr, = self.prepare(lr.unsqueeze(0))
                # sr = self.model(lr, idx_scale)
                # sr = utility.quantize(sr, self.args.rgb_range).squeeze(0)
                #
                # # ndarr = sr.byte().permute(1, 2, 0).cpu().numpy()
                # normalized = sr * 255 / self.args.rgb_range
                # ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
                # vidwri.write(ndarr)
            # video = np.stack(frames, axis=0)
            # split['query'] = video
            # split['train'] = video
            # split['num_query_imgs'] = 1809
            # split['num_gallery_imgs'] = 1809
            # split['num_query_pids'] = pids
            # split['num_gallery_pids'] = pids
            vidcap.release()
            # vidwri.release()
        # with open('split_o_x1.pickle', 'wb') as fw:
        #     pickle.dump(split, fw)

        # self.ckp.write_log(
        #     'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        # )
        # print(self.ckp)
        # torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

