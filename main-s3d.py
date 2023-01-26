import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from s3d import S3D
from datasets import WLASLDataset


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ------- Data Prep
    weights_file = './S3D_kinetics400.pt'    
    # Prepare Data Sample
    dataset = WLASLDataset("WLASL", class_file="classes_wlasl.json", split="train")
    dataloader = DataLoader(dataset)
    datasetEval = WLASLDataset("WLASL", class_file="classes_wlasl.json", split="eval")
    dataloaderEval = DataLoader(dataset)

    mean, std = batch_mean_and_sd_std(dataloader)
    # print(mean, " : ", std)
    # exit()
    num_class = dataset.num_classes()

    # ------- Model Setup
    model = S3D(num_class)
    load_weights(model, weights_file, device)
    model.train(True)
    num_epochs = 10 
    optimizer = torch.optim.SGD(list(model.parameters()), momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    metric = Accuracy(task="multiclass", num_classes=num_class, top_k=5)

    # ------- Model Training
    for epoch in range(num_epochs):
        culm_loss = 0
        culm_acc = 0
        for step, (input, target) in enumerate(dataloader):
            # forward pass
            logits = model(input)
            loss = loss_fn(logits, target)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for metrics
            culm_loss += loss.item()
            culm_acc  += metric(torch.softmax(logits, 0), target)


        culm_loss_eval = 0
        culm_acc = 0
        for step1, (input, target) in enumerate(dataloaderEval):
            with torch.no_grad():
                # forward pass
                logits = model(input)
                loss = loss_fn(logits, target)

                # for metrics
                culm_loss_eval += loss.item()
                culm_acc_eval += metric(torch.softmax(logits, 0), target)
        

        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {culm_loss_train/step:.4f}, Acc: {culm_acc/step:.4f}, Eval Loss: {culm_loss_eval/step1:.4f}, Eval Acc: {culm_acc_eval/step1:.4f}')


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

