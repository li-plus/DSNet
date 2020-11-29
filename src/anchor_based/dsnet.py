import torch
from torch import nn

from anchor_based import anchor_helper
from helpers import bbox_helper
from modules.models import build_base_model


class DSNet(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, anchor_scales,
                 num_head):
        super().__init__()
        self.anchor_scales = anchor_scales
        self.num_scales = len(anchor_scales)
        self.base_model = build_base_model(base_model, num_feature, num_head)

        self.roi_poolings = [nn.AvgPool1d(scale, stride=1, padding=scale // 2)
                             for scale in anchor_scales]

        self.layer_norm = nn.LayerNorm(num_feature)
        self.fc1 = nn.Sequential(
            nn.Linear(num_feature, num_hidden),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden)
        )
        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)

    def forward(self, x):
        _, seq_len, _ = x.shape
        out = self.base_model(x)
        out = out + x
        out = self.layer_norm(out)

        out = out.transpose(2, 1)
        pool_results = [roi_pooling(out) for roi_pooling in self.roi_poolings]
        out = torch.cat(pool_results, dim=0).permute(2, 0, 1)[:-1]

        out = self.fc1(out)

        pred_cls = self.fc_cls(out).sigmoid().view(seq_len, self.num_scales)
        pred_loc = self.fc_loc(out).view(seq_len, self.num_scales, 2)

        return pred_cls, pred_loc

    def predict(self, seq):
        seq_len = seq.shape[1]
        pred_cls, pred_loc = self(seq)

        pred_cls = pred_cls.cpu().numpy().reshape(-1)
        pred_loc = pred_loc.cpu().numpy().reshape((-1, 2))

        anchors = anchor_helper.get_anchors(seq_len, self.anchor_scales)
        anchors = anchors.reshape((-1, 2))

        pred_bboxes = anchor_helper.offset2bbox(pred_loc, anchors)
        pred_bboxes = bbox_helper.cw2lr(pred_bboxes)

        return pred_cls, pred_bboxes
