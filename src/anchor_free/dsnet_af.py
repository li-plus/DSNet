from torch import nn

from anchor_free import anchor_free_helper
from modules.models import build_base_model


class DSNetAF(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, num_head):
        super().__init__()
        self.base_model = build_base_model(base_model, num_feature, num_head)
        self.layer_norm = nn.LayerNorm(num_feature)

        self.fc1 = nn.Sequential(
            nn.Linear(num_feature, num_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden)
        )
        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)
        self.fc_ctr = nn.Linear(num_hidden, 1)

    def forward(self, x):
        _, seq_len, _ = x.shape
        out = self.base_model(x)

        out = out + x
        out = self.layer_norm(out)

        out = self.fc1(out)

        pred_cls = self.fc_cls(out).sigmoid().view(seq_len)
        pred_loc = self.fc_loc(out).exp().view(seq_len, 2)

        pred_ctr = self.fc_ctr(out).sigmoid().view(seq_len)

        return pred_cls, pred_loc, pred_ctr

    def predict(self, seq):
        pred_cls, pred_loc, pred_ctr = self(seq)

        pred_cls *= pred_ctr
        pred_cls /= pred_cls.max() + 1e-8

        pred_cls = pred_cls.cpu().numpy()
        pred_loc = pred_loc.cpu().numpy()

        pred_bboxes = anchor_free_helper.offset2bbox(pred_loc)
        return pred_cls, pred_bboxes
