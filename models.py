import torch
from torch import nn
import torch.nn.functional as F
from torch_position_embedding import PositionEmbedding
from transformers import MBartForConditionalGeneration
from customTokenizer import MBartTokenizer
from s3d import *


class EndToEndModule(nn.Module):
    """
    A PyTorch module that implements an end-to-end sign language translator. 
    This network contains two distinct modules, the Visual Module and a 
    Language Module, bridged by an intermediate MLP (V-L Mapper). By 
    default, the V-L mapper takes the dense gloss representation as
    input. More information on this network can be found here:
    https://arxiv.org/abs/2203.04287

    Args:
        num_class (int): The number of classes in the input dataset.
        backbone_weights (str): The path to the pre-trained backbone model weights.
        src_lang (str): The source language code for the transformer.
        tgt_lang (str): The target language code for the transformer.
        freeze (bool): Whether to freeze the weights of the backbone model.

    Methods:
        forward(x, labels)
            Performs a forward pass of the module.
        get_targets()
            Returns the target values of the transformer.
        add_tokens(tokens)
            Expands the language modules tokenizer to include passed tokens.

    Attributes:
        visual : VisualModule
            The visual module used for the translation.
        mapper : nn.Sequential
            The mapper used for the translation.
        language : LanguageModule
            The language module used for the translation.
    """
    def __init__(self, num_class, backbone_weights="./S3D_kinetics400.pt", src_lang="de_GL", tgt_lang="de_DE", freeze=False):
        """
        Initializes the EndToEndModule object.

        Args:
            num_class (int): The number of classes in the input dataset.
            backbone_weights (str): The path to the pre-trained backbone model weights.
            src_lang (str): The source language code for the transformer.
            tgt_lang (str): The target language code for the transformer.
            freeze (bool): Whether to freeze the weights of the backbone model.
        """
        super(EndToEndModule, self).__init__()
        self.visual = VisualModule(num_class, backbone_weights=backbone_weights, freeze=freeze)
        self.mapper = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Linear(1024, 1024),
        )
        self.language = LanguageModule(num_class, src_lang=src_lang, tgt_lang=tgt_lang, imbeds=True)

    def forward(self, x, labels):
        """
        Performs a forward pass of the module.

        Args:
            x (tensor): The input tensor.
            labels (tensor): The target tensor.

        Returns:
            tensor: The output tensor.
            tensor: The gloss prediction.
            tensor: The input embeddings.
        """
        y, gloss_pred = self.visual(x)
        input_imbeds = self.mapper(y)
        y = self.language(input_imbeds, labels)
        return y, gloss_pred, input_imbeds

    def get_targets(self):
        """
        Returns the target values of the transformer.

        Returns:
            tensor: The target tensor.
        """
        return self.language.targets

    def add_tokens(self, tokens):
        """
        Expands the language modules tokenizer to include passed tokens.

        Args:
            tokens (list): The tokens to add.
        """
        self.language.add_tokens(tokens)

class LanguageModule(nn.Module):
    """
    A Pytorch module for the language component of the end-to-end sign language
    translation network. Responsible for taking glosses (either actual or dense) 
    and translating them into the target language. Can be used independantly.

    Args:
        num_class (int): The number of classes in the input dataset.
        src_lang (str): The source language code for the transformer.
        tgt_lang (str): The target language code for the transformer.
        imbeds (bool): Whether the transformer will take raw input imbeddings
            or input ids in it's forward pass function. Raw input imbeddings 
            will have positonal imbeddings added.

    Methods:
        forward(x, labels)
            Forward pass of the module.
        generate_sentence(x, device)
            Generates a sentence from the input tensor.
        add_tokens(tokens)
            Adds tokens to the module.

    Attributes:
        model : MBartForConditionalGeneration
            The pre-trained MBart model used for translation.
        tokenizer : MBartTokenizer
            The custom MBart tokenizer.
        position_embeddings : PositionEmbedding
            The a helper function to add position embeddings.
        tgt_lang : str
            The target language code.
        src_lang : str
            The source language code.
        imbeds : bool
            If True, the input sequence is pre embedded, else requires embedding.
    
    """
    def __init__(self, num_class, src_lang="de_GL", tgt_lang="de_DE", imbeds=False):
        """
        Initializes a new instance of the LanguageModule.

        Args:
            num_class (int): The number of classes in the input dataset.
            src_lang (str): The source language code for the transformer.
            tgt_lang (str): The target language code for the transformer.
            imbeds (bool): Whether the transformer will take raw input embeddings
                or input ids in its forward pass function. Raw input embeddings
                will have positional embeddings added.
        """
        super(LanguageModule, self).__init__()
        self.model = MBartForConditionalGeneration.from_pretrained("mbart-large-cc25")
        self.tokenizer = MBartTokenizer.from_pretrained("mbart-large-cc25", src_lang=src_lang, tgt_lang=tgt_lang)
        self.position_embeddings = PositionEmbedding(num_embeddings=num_class, embedding_dim=1024, mode=PositionEmbedding.MODE_ADD)
        self.tgt_lang = tgt_lang
        self.src_lang = src_lang
        self.imbeds = imbeds

    def forward(self, x, labels):
        """
        Forward pass of the module.

        Args:
            x (tensor): The input tensor.
            labels (tensor): The tensor of labels.

        Returns:
            tensor: The tensor output of the forward pass.
        """
        if self.imbeds:
            y = self.position_embeddings(x)
            return self.model(inputs_embeds=y, labels=labels)
        else:
            return self.model(input_ids=x, labels=labels)

    def generate_sentence(self, x, device):
        '''
        Translates the glosses into a sentence (index form). Can be 'dense' 
        gloss representation, or gloss indices. 

        Args:
            x (tensor): the gloss to translate.
            device (str): device the glosses are stored.
        '''
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
        """
        Expands the tokenizer to include the passed tokens, and resizes
        the token imbeddings layer in the transformer.  

        Args:
            tokens (list[str]): A list of strings to be added. 
        """
        self.tokenizer.add_tokens(tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

class VisualModule(nn.Module):
    """A Pytorch module for the Visual component of end-to-end sign language
     translation network. Responsible for predicting a sequence of glosses
     in an input video. Can be used independantly.

    Args:
        num_class (int): The number of classes in the input dataset.
        backbone_weights (str): Path to the pretrained backbone weights. 
            If file dosen't exist, simply skips loading.
        freeze (bool): Whether to freeze the backbone weights or not.

    Methods:
        forward(x)
            Forward pass of the module.
"""


    def __init__(self, num_class, backbone_weights="./S3D_kinetics400.pt", freeze=False):
        super(VisualModule, self).__init__()
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
        """
        Forward pass of the module.

        Args:
            x (tensor): The input tensor.

        Returns:
            y: The tensor output of the forward pass.
            gloss_pred: 
        """
        if self.freeze:
            with torch.no_grad():
                y = self.backbone(x)
        else:
            y = self.backbone(x)

        y = self.head(y)
        gloss_pred = self.classifier(y).permute(1, 0, 2)

        return y, gloss_pred


class BackBone(nn.Module):
    """
    A Pytorch module for the Backbone component in the Visual Module. 
    Is simply the first four blocks of the S3D network, hence pretraining
    can be performed on S3D and the weights can be loaded onto this module.
    """
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
    """
    Applies a 1D convolution over an input signal composed of (batch, seq_len, channels).

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the convolving kernel
        stride (int or tuple): Stride of the convolution. Default: 1
        padding (str): Type of padding mode for convolution. 
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
        super(Conv1dTemporal, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                stride=stride, padding=padding)

    def forward(self, x):
        y = self.conv(x.permute(0, 2, 1))
        return y.permute(0, 2, 1)


class BatchNormTemporal1d(nn.Module):
    '''
    Applies Batch Normalization over a 2D input signal composed of 
    (batch, seq_len, channels).
    
    Args:
        features: number of channels in the input.

    '''
    def __init__(self, features):
        super(BatchNormTemporal1d, self).__init__()
        self.bn = nn.BatchNorm1d(features)

    def forward(self, x):
        if x.shape[0] == 1:
            return x
        y = self.bn(x.permute(0, 2, 1))
        return y.permute(0, 2, 1)