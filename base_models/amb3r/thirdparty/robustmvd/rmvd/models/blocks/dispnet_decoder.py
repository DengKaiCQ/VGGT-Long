# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/models/blocks/dispnet_decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ReLUAndSigmoid


def iconv_block(in_planes, out_planes):
    return nn.Sequential(nn.Conv2d(in_planes + 2, out_planes, kernel_size=3, stride=1, padding=1, bias=True),
                         nn.LeakyReLU(0.2, inplace=True))


def pred_block(in_planes, first):
    inc = 0 if first else 2
    return nn.Sequential(nn.Conv2d(in_planes + inc, 2, kernel_size=3, stride=1, padding=1, bias=True), ReLUAndSigmoid(inplace=True, min=-10, max=10))


def deconv(in_planes, out_planes, first):
    inc = 0 if first else 2

    return nn.Sequential(
        nn.ConvTranspose2d(in_planes + inc, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.2, inplace=True)
    )


class DispnetDecoder(nn.Module):
    def __init__(self):

        super().__init__()

        C_curr = 1024
        self.pred_0 = pred_block(C_curr, first=True)

        C_last = C_curr
        C_curr = int(C_curr / 2)  # 512

        self.deconv_1 = deconv(C_last, C_curr, first=True)
        self.rfeat1 = iconv_block(C_curr + 512, C_curr)
        self.pred_1 = pred_block(C_curr, first=True)

        C_last = C_curr  # 512
        C_curr = int(C_curr / 2)  # 256

        self.deconv_2 = deconv(C_last, C_curr, first=True)
        self.rfeat2 = iconv_block(C_curr + 512, C_curr)
        self.pred_2 = pred_block(C_curr, first=True)

        C_last = C_curr  # 256
        C_curr = int(C_curr / 2)  # 128

        self.deconv_3 = deconv(C_last, C_curr, first=True)
        self.rfeat3 = iconv_block(C_curr + 256, C_curr)
        self.pred_3 = pred_block(C_curr, first=True)

        C_last = C_curr  # 128
        C_curr = int(C_curr / 2)  # 64

        self.deconv_4 = deconv(C_last, C_curr, first=True)
        self.rfeat4 = iconv_block(C_curr + 128, C_curr)
        self.pred_4 = pred_block(C_curr, first=True)

        C_last = C_curr  # 64
        C_curr = int(C_curr / 2)  # 32

        self.deconv_5 = deconv(C_last, C_curr, first=True)
        self.rfeat5 = iconv_block(C_curr + 64, C_curr)
        self.pred_5 = pred_block(C_curr, first=True)

    def forward(self, enc_fused, all_enc):

        preds = {}

        pred_0 = self.pred_0(enc_fused)  # >= 0
        self.add_outputs(pred=pred_0, preds=preds)

        deconv_1 = self.deconv_1(enc_fused)
        pred_0_up = F.interpolate(pred_0, size=deconv_1.shape[-2:], mode='bilinear', align_corners=False).detach()
        rfeat1 = self.rfeat1(torch.cat((all_enc['conv5_1'], deconv_1, pred_0_up), 1))
        pred_1 = self.pred_1(rfeat1)
        self.add_outputs(pred=pred_1, preds=preds)

        deconv_2 = self.deconv_2(rfeat1)
        pred_1_up = F.interpolate(pred_1, size=deconv_2.shape[-2:], mode='bilinear', align_corners=False).detach()
        rfeat2 = self.rfeat2(torch.cat((all_enc['conv4_1'], deconv_2, pred_1_up), 1))
        pred_2 = self.pred_2(rfeat2)
        self.add_outputs(pred=pred_2, preds=preds)

        deconv_3 = self.deconv_3(rfeat2)
        pred_2_up = F.interpolate(pred_2, size=deconv_3.shape[-2:], mode='bilinear', align_corners=False).detach()
        rfeat3 = self.rfeat3(torch.cat((all_enc['conv3_1'], deconv_3, pred_2_up), 1))
        pred_3 = self.pred_3(rfeat3)
        self.add_outputs(pred=pred_3, preds=preds)

        deconv_4 = self.deconv_4(rfeat3)
        pred_3_up = F.interpolate(pred_3, size=deconv_4.shape[-2:], mode='bilinear', align_corners=False).detach()
        rfeat4 = self.rfeat4(torch.cat((all_enc['conv2'], deconv_4, pred_3_up), 1))
        pred_4 = self.pred_4(rfeat4)
        self.add_outputs(pred=pred_4, preds=preds)

        deconv_5 = self.deconv_5(rfeat4)
        pred_4_up = F.interpolate(pred_4, size=deconv_5.shape[-2:], mode='bilinear', align_corners=False).detach()
        rfeat5 = self.rfeat5(torch.cat((all_enc['conv1'], deconv_5, pred_4_up), 1))
        pred_5 = self.pred_5(rfeat5)
        self.add_outputs(pred=pred_5, preds=preds)

        return preds

    def add_outputs(self, pred, preds):

        pred_mean = pred[:, 0:1, :, :]
        pred_invdepth_log_b = pred[:, 1:2, :, :]
        pred_invdepth_b = torch.exp(pred_invdepth_log_b)
        pred_invdepth_ent = torch.log(2 * pred_invdepth_b + 1e-4) + 1

        preds.setdefault('invdepth_uncertainties_all', []).append(pred_invdepth_ent)
        preds.setdefault('invdepth_log_bs_all', []).append(pred_invdepth_log_b)
        preds.setdefault('invdepths_all', []).append(pred_mean)

        preds['invdepth_uncertainty'] = pred_invdepth_ent
        preds['invdepth_log_b'] = pred_invdepth_log_b
        preds['invdepth'] = pred_mean
