import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint
from model import EncoderDecoderAttn
from data import *


def train():
    
    vocab = Vocabulary('data/im2latex_formulas.norm.csv')
    vocab.build_vocabulary()
    transform = transforms.Compose([transforms.Resize((160, 480)), transforms.ToTensor(),])

    

    torch.backends.cudnn.benchmark = True
    load_model = False
    save_model = True

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100
    max_length = 512
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, train_dataset = get_loader( "data/formula_images_processed/formula_images_processed/", "data/im2latex_train.csv", 
                                            vocab, transform, batch_size=batch_size)


    # for tensorboard
    #writer = SummaryWriter("runs/im2latex")
    step = 0

    # initialize model, loss etc
    model = EncoderDecoderAttn(embed_size, hidden_size, vocab_size, num_layers, max_length, device).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    for epoch in range(num_epochs):
        # Uncomment the line below to see a couple of test cases
        print("training starting...")

        if epoch>0 and save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, formulas) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            formulas = formulas.to(device)

            outputs = model(imgs, formulas[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), formulas.reshape(-1)
            )

            #writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()


if __name__ == "__main__":
    train()