import torch
from torch.utils.data import DataLoader
from models import VisualModule, LanguageModule, EndToEndModule, load_weights
from modelLanguage import LanguageModule
from datasets import ElarDataset, PheonixDataset, BobslDataset, collate_fn, batch_mean_and_sd
from torchtext.data.metrics import bleu_score
from functools import partial

def main(weights):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ------- Data Prep
    # weights_file = os.path.join('S3D_kinetics400+WLASL', 'S3D_kinetics400+WLASL3-7.pkl')   
    # Specify necessary files
    root = "PHOENIX-2014-T"
    class_file = "classes_pheonix.json"
    # Mean and std can be calculated with the helper function batch_mean_and_sd
    mean = [0.5367, 0.5268, 0.5192]
    std = [0.2869, 0.2955, 0.3259]
    
    dataset = PheonixDataset(root, class_file=class_file, split="train", mean=mean, std=std)
    dataloader = DataLoader(dataset, batch_size=5, collate_fn=partial(collate_fn, device=device), shuffle=True)
    datasetEval = PheonixDataset(root, class_file=class_file, split="val", mean=mean, std=std)
    dataloaderEval = DataLoader(datasetEval, batch_size=5, collate_fn=partial(collate_fn, device=device), shuffle=True)

    num_class = dataset.num_classes()
    # ------- Model Setup
    model = EndToEndModule(num_class, classify=True, backbone_weights="", src_lang="de_GL", tgt_lang="de_DE", freeze=True)
    model.add_tokens(dataset.get_class_names())
    load_weights(model, weights)
    model.to(device)
    model.train(True)

    # ------- Loss & Optimizer Setup
    num_epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn_visual = torch.nn.CTCLoss()
    loss_fn_language = torch.nn.CrossEntropyLoss()

    print("Started training")
    # ------- Model Training
    for epoch in range(num_epochs):
        culm_loss_visual = 0
        culm_loss_lang = 0
        for step, (input, input_lengths, gloss_targets, gloss_lengths, freetransl) in enumerate(dataloader):
            labels = model.language.tokenizer(text_target=freetransl, return_tensors="pt", padding=True)['input_ids'].to(device)

            # forward pass
            logits, gloss_pred, _ = model(input, labels)
            logits = logits['logits']
            targets = torch.nn.functional.one_hot(torch.roll(labels, 1, 1), num_classes=logits.shape[-1]).type(torch.float)

            loss_visual = loss_fn_visual(gloss_pred, gloss_targets, input_lengths, gloss_lengths)
            loss_lang = loss_fn_language(logits, targets)
            
            culm_loss_visual += loss_visual.item()
            culm_loss_lang += loss_lang.item()

            # backward pass
            optimizer.zero_grad()
            lossCombined = loss_visual + loss_lang
            lossCombined.backward()
            optimizer.step()

        culm_loss_visual_eval = 0
        culm_loss_lang_eval = 0
        culm_bleu_eval = 0
        model.eval()
        for step_eval, (input, input_lengths, gloss_targets, gloss_lengths, freetransl) in enumerate(dataloaderEval):
            with torch.no_grad():
                labels = model.language.tokenizer(text_target=freetransl, return_tensors="pt", padding=True)['input_ids'].to(device)

                # forward pass
                logits, gloss_pred, input_imbeds = model(input, labels)
                logits = logits['logits']
                targets = torch.nn.functional.one_hot(torch.roll(labels, 1, 1), num_classes=logits.shape[-1]).type(torch.float)

                loss_visual = loss_fn_visual(gloss_pred, gloss_targets, input_lengths, gloss_lengths)
                loss_lang = loss_fn_language(logits, targets)
                
                culm_loss_visual_eval += loss_visual.item()
                culm_loss_lang_eval += loss_lang.item()

                generate = model.language.generate_sentence(input_imbeds, device)
                pred_sentence_strings = model.language.tokenizer.batch_decode(generate, skip_special_tokens=True)
                pred_sentence_strings = [sentence.split(" ") for sentence in pred_sentence_strings]
                actual_sentence_strings = model.language.tokenizer.batch_decode(torch.roll(labels, 1, 1), skip_special_tokens=True)
                actual_sentence_strings = [sentence.split(" ") for sentence in actual_sentence_strings]
                
                culm_bleu_eval += bleu_score(pred_sentence_strings, actual_sentence_strings, max_n=1, weights=[1.0])

        model.train(True)
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss Visual: {culm_loss_visual/(step + 1):.4f}, Loss Lang: {culm_loss_lang/(step + 1):.4f}, Loss Visual Eval: {culm_loss_visual_eval/(step_eval + 1):.4f}, Loss Lang Eval: {culm_loss_lang_eval/(step_eval + 1):.4f}, BLEU Eval: {culm_bleu_eval/(step + 1):.4f}')
        torch.save(model.state_dict(),f'EndToEndPheonix-{epoch}.pkl')



if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("")

