import random

import torch
import torch.nn as nn
import torch.optim as optim

# import custom models



class Seq2Seq(nn.Module):
    """ The Sequence to Sequence model.
        You will need to complete the init function and the forward function.
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.device = device
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the Seq2Seq model. You should use .to(device) to make sure  #
        #    that the models are on the same device (CPU/GPU). This should take no  #
        #    more than 2 lines of code.                                             #
        #############################################################################
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, source, out_seq_len = None):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len)
                out_seq_len (int): the maximum length of the output sequence. If None, the length is determined by the input sequences.
        """

        batch_size = source.shape[0]

        if out_seq_len is None:
            out_seq_len = source.shape[1]

        #############################################################################
        # TODO:                                                                     #
        #   Implement the forward pass of the Seq2Seq model. Please refer to the    #
        #   following steps:                                                        #
        #       1) Get the last hidden representation from the encoder. Use it as   #
        #          the first hidden state of the decoder                            #
        #       2) The first input for the decoder should be the <sos> token, which #
        #          is the first in the source sequence.                             #
        #       3) Feed this first input and hidden state into the decoder          #  
        #          one step at a time in the sequence, adding the output to the     #
        #          final outputs.                                                   #
        #       4) Update the input and hidden weights being fed into the decoder   #
        #          at each time step. The decoder output at the previous time step  # 
        #          will have to be manipulated before being fed in as the decoder   #
        #          input at the next time step.                                     #
        #  Note: Don't hardcode cuda, it will cause errors in gradescope            #
        #                 if Training freezes it is due to memory error             #                                                           #
        #############################################################################

        _, hidden = self.encoder(source)

        token = source[:, 0:1]
        # token = torch.unsqueeze(token, dim=1)

        outputs = torch.zeros((batch_size, self.decoder.output_size), device=self.device)
        # outputs = torch.zeros(batch_size, out_seq_len, self.decoder.output_size)

        for i in range(1, out_seq_len):
            output, hidden = self.decoder(token, hidden)
            # outputs[:, i] = output
            token = torch.argmax(output, dim=1)
            token = torch.unsqueeze(token, dim=1)
            # token = token.long()
            # if i < out_seq_len - 1:
            outputs = torch.cat((outputs, output))

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outputs



        

