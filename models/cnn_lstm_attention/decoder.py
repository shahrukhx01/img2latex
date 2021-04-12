import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, max_length=512, device='cpu'):
        super(Decoder, self).__init__()

        self.lstm_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        ## embedding look up for formula symbols
        self.embed = nn.Embedding(vocab_size, embed_size)

        ## attention block to focus on relevant parts of context vector and input embeddings
        self.attn = nn.Linear(hidden_size + embed_size, max_length)
        self.attn_combine = nn.Linear(hidden_size + embed_size, self.hidden_size)

        ## LSTM for processing formula sequence
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)

        ## output layer for 
        self.out = nn.Linear(hidden_size, vocab_size)

        ## dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)
    
    def init_hidden(self, batch_size):
        """
        Initializes hidden and context weight matrix before each 
		forward pass through LSTM
        """
        return (Variable(torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(self.device)),
                Variable(torch.zeros(self.lstm_layers, batch_size, self.hidden_size)).to(self.device))

    def forward(self, input, encoder_outputs):
        hidden = self.init_hidden(input.size(-1))

        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs)
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        _, (new_hidden, new_cell)= self.lstm(attn_applied, hidden)
        output = self.linear(new_hidden[-1])
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output