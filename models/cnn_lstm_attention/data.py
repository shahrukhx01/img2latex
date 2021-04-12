import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
from tqdm import tqdm
import torchvision.transforms as transforms




class Vocabulary:
    def __init__(self, formulas_file, freq_threshold=10):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
        self.formulas = pd.read_csv(formulas_file).formulas.tolist()

    def __len__(self):
        return len(self.itos)

    def tokenizer_latex(self, text):
        return [token.lower() for token in text.split()]

    def build_vocabulary(self):
        frequencies = {}
        idx = 4

        for idx, formula in tqdm(enumerate(self.formulas)):
            for symbol in self.tokenizer_latex(formula):
                if symbol not in frequencies:
                    frequencies[symbol] = 1
                else:
                    frequencies[symbol] += 1

        
        for formula in tqdm(self.formulas):
            for symbol in self.tokenizer_latex(formula):
                if frequencies[symbol] >= self.freq_threshold:
                    self.stoi[symbol] = idx
                    self.itos[idx] = symbol
                    idx += 1

    def numericalize(self, text, max_length):
        tokenized_text = self.tokenizer_latex(text)
        numericalized_seq = [self.stoi["<PAD>"]] * (max_length-2) ## to account for start and end token

        for idx, token in enumerate(tokenized_text):
            token_id = self.stoi["<UNK>"]
            if token in self.stoi:
                token_id = self.stoi[token]
            
            numericalized_seq[idx] = token_id

        return numericalized_seq


class Im2LatexDataset(Dataset):
    def __init__(self, root_dir, data_file, vocab, transform, max_length=512):
        self.root_dir = root_dir
        self.max_length = max_length
        self.df = pd.read_csv(data_file)
        ## dropping too long sequences
        mask = (self.df['formula'].str.len() <= max_length) 
        self.df = self.df.loc[mask]

        ## to be remove dev set
        self.df = self.df.head(100)

        ## resize images and convert to tensor transform
        self.transform = transform

        # Get img, formula columns
        self.imgs = self.df["image"]
        self.formulas = self.df["formula"]

        # Initialize vocabulary and build vocab
        formulas = pd.read_csv(data_file)
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        formula = self.formulas[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_formula = [self.vocab.stoi["<SOS>"]]
        numericalized_formula += self.vocab.numericalize(formula, self.max_length)
        numericalized_formula.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_formula)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        formulas = [item[1] for item in batch]
        targets = torch.stack(formulas)

        return imgs, targets


def get_loader(root_folder, data_file, vocab, transform, batch_size=8, num_workers=2,
                shuffle=True, pin_memory=True):

    dataset = Im2LatexDataset(root_folder, data_file, vocab, transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset


if __name__ == "__main__":
    
    vocab = Vocabulary('im2latex_formulas.norm.csv')
    vocab.build_vocabulary()
    transform = transforms.Compose([transforms.Resize((160, 480)), transforms.ToTensor(),])

    train_loader, train_dataset = get_loader( "formula_images_processed/formula_images_processed/", "im2latex_train.csv", 
                                            vocab, transform, batch_size=8)

    for idx, (imgs, formulas) in enumerate(train_loader):
        print(imgs.shape)
        print(formulas.shape)