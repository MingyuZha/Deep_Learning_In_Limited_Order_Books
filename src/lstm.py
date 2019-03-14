import torch.nn as nn
import torch.nn.functional as F
import torch

class LSTM_model(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, fc_size, output_size):
        super(LSTM_model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(num_layers=num_layers,
                            input_size = self.input_size,
                            hidden_size = self.hidden_size,
                            batch_first = True,
                            dropout =  0.2)
        self.fc = nn.Linear(hidden_size, fc_size)
        self.decoder = nn.Linear(fc_size, output_size)
        # self.hidden = self._init_hidden()

    # def _init_hidden(self):
    #     return (Variable(torch.zeros(self.num_layers, self.batch_size, self.num_hiddens) ),
    #             Variable(torch.zeros(self.num_layers, self.batch_size, self.num_hiddens)) )

    def forward(self, order_book_data):
        """
        order_book_data: [batch_size, sequence_len, dims]
        """
        lstm_output, _ = self.lstm(order_book_data, None)
        output = F.relu(self.fc(lstm_output))
        output = self.decoder(output)
        return F.log_softmax(output, dim=2)

