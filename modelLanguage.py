import torch
from transformers import MBartTokenizer, MBartModel
from torch import nn
import torch.nn.functional as F


class LanguageModule(nn.Module):
    def __init__(self, tgt_lang="de_DE"):
        super(LanguageModule, self).__init__()
        self.model = MBartModel.from_pretrained("/home/groups/auslan-ai/models/mbart-large-cc25")
        self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", tgt_lang=tgt_lang)
        self.tgt_lang = tgt_lang
        self.targets = None

    def forward(self, x, target):
        self.targets = self.tokenizer(text_target=target, return_tensors="pt", padding=True)['input_ids']
        y = self.model(decoder_input_ids=self.targets, inputs_embeds=x)
        return y

    def generate(self, x):
        # Create attention mask, possibly unecessary
        attention_mask = torch.ones(x.shape[:2], dtype=torch.long)
        # Create tensor of the tgt_lang id
        decoder_input_ids = torch.ones((x.shape[0], 1), dtype=torch.long)*self.tokenizer.lang_code_to_id[self.tgt_lang]
        return self.mdoel.generate(attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, inputs_embeds=x, max_length=100, num_beams=4)