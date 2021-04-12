from encoder import Encoder
from decoder import Decoder
import torch.nn as nn

class EncoderDecoderAttn(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_length, device):
        super(EncoderDecoderAttn, self).__init__()
        self.encoder_cnn = Encoder(embed_size)
        self.decoder_rnn = Decoder(embed_size, hidden_size, vocab_size, num_layers, max_length, device)

    def forward(self, images, formulas):
        features = self.encoder_cnn(images)
        outputs = self.decoder_rnn(formulas, features)
        return outputs

    """def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]"""