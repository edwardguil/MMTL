import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from s3d import S3D
from modelVisual import VisualModule
from modelLanguage import LanguageModule
from datasets import ElarDataset, PheonixDataset, elar_collate_fn
from functools import partial


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ------- Data Prep
    weights_file = './S3D_kinetics400.pt'    
    # Prepare Data Sample
    #dataset = ElarDataset('train_elar.json', 'classes_elar.txt', './ELAR_data')#, mean=[0.2481, 0.2395, 0.3078], std=[0.2542, 0.2231, 0.2403])
    dataset = PheonixDataset("PHEONIX-2014-T", class_file="classes_pheonix.json", split="test")
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=partial(elar_collate_fn, device=device))

    #mean, std = batch_mean_and_sd(dataloader)
    #print(mean, " : ", std)
    num_class = dataset.num_classes()

    # ------- Model Setup
    model_visual = VisualModule(num_class).to(device)
    model_language = LanguageModule(num_class).to(device)
    # torch.backends.cudnn.benchmark = False
    # model_visual.eval()
    # model_language.eval()
    model_visual.train(True)
    model_language.train(True)

    num_epochs = 10 
    optimizer = torch.optim.Adam(list(model_visual.parameters()) )
    #optimizer = torch.optim.Adam(list(model_visual.parameters()) + list(model_language.parameters()))
    loss_fn_visual = torch.nn.CTCLoss()
    loss_fn_language = torch.nn.CrossEntropyLoss()

    print("Started training")
    # ------- Model Training
    for epoch in range(num_epochs):
        for step, (input, input_lengths, gloss_targets, gloss_lengths, freetransl) in enumerate(dataloader):
            # forward pass
            # with torch.no_grad():
            logits, gloss_pred = model_visual(input)
            loss_visual = loss_fn_visual(gloss_pred, gloss_targets, input_lengths, gloss_lengths)

            # loss_language = loss_fn_language(sentence_pred, sentence_targets)

            # backward pass
            optimizer.zero_grad()
            lossCombined = loss_visual #+ loss_language
            lossCombined.backward()
            optimizer.step()

        print (f'Epoch [{epoch+1}/{num_epochs}], Visual Loss: {loss_visual.item():.4f}')


def batch_mean_and_sd(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for step, (data, input_lengths, gloss_targets, gloss_lengths, freetransl) in enumerate(dataloader):
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3,4])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3,4])
        num_batches += 1
    
    mean = channels_sum / num_batches

    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std     

def batch_mean_and_sd_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for step, (data, target) in enumerate(dataloader):
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3,4])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3,4])
        num_batches += 1
    
    mean = channels_sum / num_batches

    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std  

def transform(snippet):
    ''' stack & normalization '''
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)

    return snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)

def load_weights(model, weights_file, device):
    if os.path.isfile(weights_file):
        print ('loading weight file')
        weight_dict = torch.load(weights_file, map_location=torch.device(device))
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print (' size? ' + name, param.size(), model_dict[name].size())
            else:
                print (' name? ' + name)

        print (' loaded')
    else:
        print ('weight file?')
    return model

if __name__ == '__main__':
    main()

