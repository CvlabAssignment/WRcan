import glob
import os
import math

import utility
from data import common

import torch
import cv2

from tqdm import tqdm


class PictureTester():
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
        src_dir = "./images5o_crop/"
        all = glob.iglob(os.path.join(src_dir, "*.jpg"))
        # for idx_scale, scale in enumerate(self.scale):
        for i , img in enumerate(all):
            lr = cv2.imread(img)
            lr, = common.set_channel(lr, n_channels=self.args.n_colors)
            lr, = common.np2Tensor(lr, rgb_range=self.args.rgb_range)
            lr, = self.prepare(lr.unsqueeze(0))
            sr = self.model(lr, 0)
            sr = utility.quantize(sr, self.args.rgb_range).squeeze(0)

            normalized = sr * 255 / self.args.rgb_range
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            ndarr.save('./example/' + str(i) + '.jpg')
            # vidwri.write(ndarr)


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

