import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from s3d import S3D
from models import load_weights
from datasets import WLASLDataset, s3d_collate_fn, batch_mean_and_sd_std
from functools import partial


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ------ File Preps
    root = "WLASL"
    class_file = "classes_wlasl.json"
    # Mean and std can be calculated with helper 'batch_mean_and_sd_std'
    mean = [0.3269, 0.3319, 0.3176]
    std = [0.3490, 0.3429, 0.3342]

    # -------  Data Prep   
    dataset = WLASLDataset(root, class_file=class_file, split="train", mean=mean, std=std)
    datasetEval = WLASLDataset(root, class_file=class_file, split="val", mean=mean, std=std)

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=partial(s3d_collate_fn, device=device), shuffle=True)
    dataloaderEval = DataLoader(datasetEval, batch_size=2, collate_fn=partial(s3d_collate_fn, device=device), shuffle=True)

    # ------- Model Setup
    weights_file = './S3D_kinetics400.pt'
    num_class = dataset.num_classes()
    model = S3D(num_class)   
    load_weights(model, weights_file)

    model.to(device)
    model.train(True)

    # -------  Helper setup
    optimizer = torch.optim.SGD(list(model.parameters()), momentum=0.9, lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()
    metric = Accuracy(task="multiclass", num_classes=num_class, top_k=5).to(device)

    # -------  Model Training
    print("Started training...")
    num_epochs = 20 
    updated = False
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
        culm_acc_eval = 0
        for step_eval, (input, target) in enumerate(dataloaderEval):
            with torch.no_grad():
                # forward pass
                logits = model(input)
                loss = loss_fn(logits, target)

                # for metrics
                culm_loss_eval += loss.item()
                culm_acc_eval += metric(torch.softmax(logits, 0), target)
        
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {culm_loss/(step + 1):.4f}, Acc: {culm_acc/(step + 1):.4f}, Eval Loss: {culm_loss_eval/(step_eval + 1):.4f}, Eval Acc: {culm_acc_eval/(step_eval + 1):.4f}')
        torch.save(model.state_dict(),f'S3D_kinetics400+WLASL-{epoch}.pkl')
        if epoch == 6 and not updated:
            updated = True
            for g in optimizer.param_groups:
                g['lr'] = 0.01

if __name__ == '__main__':
    main()

