import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class forward function.

    def __init__(self, input_size, hidden_size):
        """ Init function for VanillaRNN class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns:
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # i_t, the input gate
        self.W_ii = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size))

        # f_t, the forget gate
        self.W_if = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_if = nn.Parameter(torch.Tensor(hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_hf = nn.Parameter(torch.Tensor(hidden_size))

        # g_t, the cell gate
        self.W_ig = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size))

        # o_t, the output gate
        self.W_io = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size))


        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. If init_states is None, you     #
        #   should initialize h_t and c_t to be zero vectors.                          #
        ################################################################################
        h_t, c_t = None, None
        h_t = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size), requires_grad=False)
        c_t = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size), requires_grad=False)

        num_sequence = x.shape[1]

        # print("hidden_size", self.hidden_size)
        # print("input size", self.input_size)
        # print("init_states", init_states)
        # print("x shape", x.shape)

        for idx in range(num_sequence):
            input = x[:, idx, :]
            input_gate_x = torch.matmul(input, self.W_ii) + self.b_ii
            input_gate_h = torch.matmul(h_t, self.W_hi) + self.b_hi

            input_gate_t = torch.sigmoid(torch.add(input_gate_x, input_gate_h))


            forget_gate_x = torch.matmul(input, self.W_if) + self.b_if
            forget_gate_h = torch.matmul(h_t, self.W_hf) + self.b_hf
            forget_gate_t = torch.sigmoid(torch.add(forget_gate_x, forget_gate_h))

            # g_t, the cell gate
            cell_gate_x = torch.matmul(input, self.W_ig) + self.b_ig
            cell_gate_h = torch.matmul(h_t, self.W_hg) + self.b_hg
            cell_gate_t = torch.tanh(torch.add(cell_gate_x, cell_gate_h))

            output_gate_x = torch.matmul(input, self.W_io) + self.b_io
            output_gate_h = torch.matmul(h_t, self.W_ho) + self.b_ho
            output_gate_t = torch.sigmoid(output_gate_x + output_gate_h)

            c_t = torch.mul(forget_gate_t, c_t) + torch.mul(input_gate_t, cell_gate_t)
            h_t = torch.mul(output_gate_t, torch.tanh(c_t))

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)
