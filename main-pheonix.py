import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from s3d import S3D
from modelEndToEnd import EndToEndModule
from datasets import ElarDataset, PheonixDataset, BobslDataset, elar_collate_fn
from torchtext.data.metrics import bleu_score
from functools import partial
import psutil

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ------- Data Prep
    weights_file = os.path.join('S3D_kinetics400+WLASL', 'S3D_kinetics400+WLASL3-7.pkl')    
    # Prepare Data Sample
    #dataset = ElarDataset('train_elar.json', 'classes_elar.txt', './ELAR_data')#, mean=[0.2481, 0.2395, 0.3078], std=[0.2542, 0.2231, 0.2403])
    
    dataset = PheonixDataset("/home/groups/auslan-ai/PHOENIX-2014-T", class_file="classes_pheonix.json", split="train", mean=[0.5367, 0.5268, 0.5192], std=[0.2869, 0.2955, 0.3259])
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=partial(elar_collate_fn, device=device), shuffle=True)
    datasetEval = PheonixDataset("/home/groups/auslan-ai/PHOENIX-2014-T", class_file="classes_pheonix.json", split="val", mean=[0.5367, 0.5268, 0.5192], std=[0.2869, 0.2955, 0.3259])
    dataloaderEval = DataLoader(datasetEval, batch_size=1, collate_fn=partial(elar_collate_fn, device=device), shuffle=True)
    num_class = dataset.num_classes()

    # /home/groups/auslan-ai/
    # dataset = BobslDataset("/home/groups/auslan-ai/bobsl", split="train")
    # dataloader = DataLoader(dataset, batch_size=1, collate_fn=partial(elar_collate_fn, device=device), shuffle=False)
    #datasetEval = BobslDataset("/home/groups/auslan-ai/bobsl", split="val")
    #dataloaderEval = DataLoader(datasetEval, batch_size=1, collate_fn=partial(elar_collate_fn, device=device), shuffle=True)
    # print("Started mean and std calculation PHEONIX")
    # mean, std = batch_mean_and_sd(dataloader)
    # print(mean, " : ", std)
    # exit()


    # ------- Model Setup
    model = EndToEndModule(num_class, classify=True, backbone_weights=weights_file, tgt_lang="de_DE").to(device)
    model.train(True)
    num_epochs = 20
    #optimizer = torch.optim.Adam(list(model_visual.parameters()) + list(model_language.parameters()))
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn_visual = torch.nn.CTCLoss()
    loss_fn_language = torch.nn.CrossEntropyLoss()

    print("Started training")
    # ------- Model Training
    for epoch in range(num_epochs):
        culm_loss_visual = 0
        culm_loss_lang = 0
        # culm_bleu = 0
        for step, (input, input_lengths, gloss_targets, gloss_lengths, freetransl) in enumerate(dataloader):
            # forward pass
            logits, gloss_pred = model(input, freetransl)
            loss_visual = loss_fn_visual(gloss_pred, gloss_targets, input_lengths, gloss_lengths)
            loss_lang = loss_fn_language(logits['logits'].permute(0, 2, 1), model.get_targets())
            
            culm_loss_visual += loss_visual.item()
            culm_loss_lang += loss_lang.item()
            # culm_blue += bleu_score(sentence_pred, freetransl)

            # backward pass
            optimizer.zero_grad()
            lossCombined = loss_visual + loss_visual
            lossCombined.backward()
            optimizer.step()

        culm_loss_visual_eval = 0
        culm_loss_lang_eval = 0
        culm_bleu_eval = 0
        for step_eval, (input, input_lengths, gloss_targets, gloss_lengths, freetransl) in enumerate(dataloaderEval):
            with torch.no_grad():
                logits, gloss_pred = model(input, freetransl)
                # Add positional embeddings?
                loss_visual = loss_fn_visual(gloss_pred, gloss_targets, input_lengths, gloss_lengths)
                loss_lang = loss_fn_language(logits['logits'].permute(0, 2, 1), model.get_targets())

                culm_loss_visual_eval += loss_visual.item()
                culm_loss_lang_eval += loss_lang.item()
                # culm_blue_eval += bleu_score(sentence_pred, freetransl)

        # BLEU: {culm_bleu/(step + 1):.4f} , BLEU Eval: {culm_bleu_eval/(step + 1):.4f}
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss Visual: {culm_loss_visual/(step + 1):.4f}, Loss Lang: {culm_loss_lang/(step + 1):.4f}, Loss Visual Eval: {culm_loss_visual_eval/(step_eval + 1):.4f}, Loss Lang Eval: {culm_loss_lang_eval/(step_eval + 1):.4f}')
        #print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {culm_loss/(step + 1):.4f}, Acc: {culm_acc/(step + 1):.4f}, Eval Loss: {culm_loss_eval/(step_eval + 1):.4f}, Eval Acc: {culm_acc_eval/(step_eval + 1):.4f}')
        torch.save(model.state_dict(),f'./eTe-kinetics400+WLASL+Pheonix-{epoch}.pkl')


def batch_mean_and_sd(dataloader):
    with torch.no_grad():
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        try:
            for step, (data, input_lengths, gloss_targets, gloss_lengths, freetransl) in enumerate(dataloader):
                # Mean over batch, height and width, but not over the channels
                used = psutil.virtual_memory()[2]
                available = psutil.virtual_memory()[3]/1000000000
                channels_sum += torch.mean(data, dim=[0,2,3,4])
                channels_squared_sum += torch.mean(data**2, dim=[0,2,3,4])
                num_batches += 1

        except Exception as e:
            print(f"Exception Caught: {e}")
            print('RAM memory % used:', used)
            print('RAM Used (GB):', available)
            print(f"Step: {step}. Freetransl {freetransl}.")
            print(f"Channels Sum: {channels_sum}. Channels Squared Sum: {channels_squared_sum}")
            exit()

        mean = channels_sum / num_batches

        std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std     

def batch_mean_and_sd_std(dataloader):
    with torch.no_grad():
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

