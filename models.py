import torch
from torch import nn
import torch.nn.functional as F
from torch_position_embedding import PositionEmbedding
from transformers import MBartForConditionalGeneration
from customTokenizer import MBartTokenizer
from s3d import *


class EndToEndModule(nn.Module):
    def __init__(self, num_class, classify=True, backbone_weights="./S3D_kinetics400.pt", src_lang="de_GL", tgt_lang="de_DE", freeze=False, imbeds=True):
        super(EndToEndModule, self).__init__()
        self.visual = VisualModule(num_class, classify=classify, backbone_weights=backbone_weights, freeze=freeze)
        self.mapper = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Linear(1024, 1024),
        )
        self.language = LanguageModule(num_class, src_lang=src_lang, tgt_lang=tgt_lang, imbeds=imbeds)

    def forward(self, x, labels):
        y, gloss_pred = self.visual(x)
        input_imbeds = self.mapper(y)
        y = self.language(input_imbeds, labels)
        return y, gloss_pred, input_imbeds

    def get_targets(self):
        return self.language.targets

    def add_tokens(self, tokens):
        self.language.add_tokens(tokens)

class LanguageModule(nn.Module):
    def __init__(self, num_class, src_lang="de_GL", tgt_lang="de_DE", imbeds=False):
        super(LanguageModule, self).__init__()
        self.model = MBartForConditionalGeneration.from_pretrained("mbart-large-cc25")
        self.tokenizer = MBartTokenizer.from_pretrained("mbart-large-cc25", src_lang=src_lang, tgt_lang=tgt_lang)
        self.position_embeddings = PositionEmbedding(num_embeddings=num_class, embedding_dim=1024, mode=PositionEmbedding.MODE_ADD)
        self.tgt_lang = tgt_lang
        self.src_lang = src_lang
        self.imbeds = imbeds

    def forward(self, x, labels):
        if self.imbeds:
            y = self.position_embeddings(x)
            return self.model(inputs_embeds=y, labels=labels)
        else:
            return self.model(input_ids=x, labels=labels)

    def generate_sentence(self, x, device):
        # Create attention mask
        attention_mask = torch.ones(x.shape[:2], dtype=torch.long).to(device)
        # Create tensor of the tgt_lang id
        decoder_input_ids = torch.ones((x.shape[0], 1), dtype=torch.long)*self.tokenizer.lang_code_to_id[self.tgt_lang]
        decoder_input_ids = decoder_input_ids.to(device)
        if self.imbeds:
            return self.model.generate(inputs_embeds=x, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, max_length=100, num_beams=4, repetition_penalty=100.0)
        else:
            return self.model.generate(input_ids=x, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, max_length=100, num_beams=4, repetition_penalty=100.0)

    def add_tokens(self, tokens):
        self.tokenizer.add_tokens(tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

class VisualModule(nn.Module):
    def __init__(self, num_class, classify=True, backbone_weights="./S3D_kinetics400.pt", freeze=False):
        super(VisualModule, self).__init__()
        self.classify = classify
        self.freeze = freeze
        self.backbone = load_weights(BackBone(), backbone_weights)
        self.head = nn.Sequential(
            # Projection block
            nn.Linear(832, 832), # Input: N x T/4 x 832
            BatchNormTemporal1d(832),
            nn.ReLU(),
            # Conv Block - Temporal Convolutional (1d convolution) (N, Time, Len)
            Conv1dTemporal(832, 832, kernel_size=3, stride=1, padding='same'),
            Conv1dTemporal(832, 832, kernel_size=3, stride=1, padding='same'),
            nn.Linear(832, 512),
            nn.ReLU() # The output from this is going into VL-Mapper or Classifier
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, num_class),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                y = self.backbone(x)
        else:
            y = self.backbone(x)

        y = self.head(y)
        gloss_pred = None
        if self.classify:
            gloss_pred = self.classifier(y).permute(1, 0, 2)

        return y, gloss_pred


class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()
        self.base = nn.Sequential(
            SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            BasicConv3d(64, 64, kernel_size=1, stride=1),
            SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            Mixed_3b(),
            Mixed_3c(),
            nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
            Mixed_4b(),
            Mixed_4c(),
            Mixed_4d(),
            Mixed_4e(),
            Mixed_4f(),
        )

    def forward(self, x):
        y = self.base(x)
        # Use spatial poolig to make data spatially invariant (if it isn't already)
        # y = temporal_spatial_pyramid_pool(y)
        y = F.avg_pool3d(y, (1, y.size(3), y.size(4)), stride=1)
        y = y.view(y.size(0), y.size(2), y.size(1))

        return y


class Conv1dTemporal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
        super(Conv1dTemporal, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                stride=stride, padding=padding)

    def forward(self, x):
        y = self.conv(x.permute(0, 2, 1))
        return y.permute(0, 2, 1)


class BatchNormTemporal1d(nn.Module):
    def __init__(self, features):
        super(BatchNormTemporal1d, self).__init__()
        self.bn = nn.BatchNorm1d(features)

    def forward(self, x):
        if x.shape[0] == 1:
            return x
        y = self.bn(x.permute(0, 2, 1))
        return y.permute(0, 2, 1)