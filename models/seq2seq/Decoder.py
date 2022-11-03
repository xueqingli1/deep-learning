import random

import torch
import torch.nn as nn
import torch.optim as optim


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN"):
        super().__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN", "LSTM".                                                   #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #############################################################################
        self.embedding = nn.Embedding(num_embeddings=self.output_size, embedding_dim=self.emb_size)
        if model_type == "RNN":
            self.rnn = nn.RNN(input_size=self.emb_size, hidden_size=self.decoder_hidden_size, batch_first=True)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(input_size=self.emb_size, hidden_size=self.decoder_hidden_size, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.decoder_hidden_size, out_features=self.output_size),
            nn.LogSoftmax(dim=-1)
        )
        self.dropout = nn.Dropout(p=dropout)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input, hidden):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len); HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden weights of the previous time step from the decoder
            Returns:
                output (tensor): the output of the decoder
                hidden (tensor): the weights coming out of the hidden unit
        """
        output = None

        #############################################################################
        # TODO: Implement the forward pass of the encoder.                          #
        #      Please apply the dropout to the embedding layer                      #
        #############################################################################

        input = self.embedding(input)
        input = self.dropout(input)
        output, hidden = self.rnn(input, hidden)
        # if self.model_type == "RNN":
        #     output, hidden = self.rnn(input, hidden)
        # elif self.model_type == "LSTM":
        #     output, hidden, _) = self.rnn(input, hidden)

        output = self.linear(output)
        # output = torch.squeeze(output, dim=1)
        output = output[:,0,:]

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output, hidden

