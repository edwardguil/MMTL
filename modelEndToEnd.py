import torch
from torch import nn
import torch.nn.functional as F
from modelVisual import VisualModule
from modelLanguage import LanguageModule


class EndToEndModule(nn.Module):
    def __init__(self, num_class, classify=True, backbone_weights="./S3D_kinetics400.pt", tgt_lang="de_DE"):
        super(EndToEndModule, self).__init__()
        self.visual = VisualModule(num_class, classify, backbone_weights=backbone_weights)
        self.mapper = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Linear(1024, 1024),
        )
        self.language = LanguageModule(tgt_lang=tgt_lang)

    def get_targets(self):
        return self.language.targets

    def forward(self, x, target):
        y, gloss_pred = self.visual(x)
        y = self.mapper(y)
        y = self.language(y, target)
        return y, gloss_pred
