import sys, torch
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
    train_data_file = "train_elar.json"
    test_data_file = "test_elar.json"
    elar_root = "ELAR_Data"
    class_file = "classes_elar.txt"
    # This can be calculated with the helper function batch_mean_and_sd
    elar_mean = [0.2481, 0.2395, 0.3078]
    elar_std = [0.2542, 0.2231, 0.2403]

    dataset = ElarDataset(train_data_file, class_file, elar_root, mean=elar_mean, std=elar_std)
    datasetEval = ElarDataset(test_data_file, class_file, elar_root, mean=elar_mean, std=elar_std)

    dataloader = DataLoader(dataset, batch_size=5, collate_fn=partial(collate_fn, device=device), shuffle=False)
    dataloaderEval = DataLoader(datasetEval, batch_size=5, collate_fn=partial(collate_fn, device=device), shuffle=False)

    num_class = dataset.num_classes()
    # ------- Model Setup
    model = EndToEndModule(num_class, backbone_weights="", src_lang="au_GL", tgt_lang="en_XX", freeze=True)
    model.add_tokens(dataset.class_names)
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
        torch.save(model.state_dict(),f'EndToEndElar-{epoch}.pkl')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("")

