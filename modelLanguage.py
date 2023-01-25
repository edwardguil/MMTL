import torch
from transformers import MBartTokenizer, MBartModel
from torch import nn
import torch.nn.functional as F


class LanguageModule(nn.Module):
    def __init__(self, num_class, map=True):
        super(LanguageModule, self).__init__()
        self.mapper = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Linear(1024, 1024)
        )
        self.model = MBartModel.from_pretrained("facebook/mbart-large-cc25")
        self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")

    def forward(self, x):
        if map:
            x = self.mapper(x)
        y = self.model(x)
        return y