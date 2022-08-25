import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512, n_layers=2):
        super(Actor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.learning_rate = 0.001

        self.lstm = nn.LSTM(input_size=self.state_size, hidden_size=self.hidden_size, num_layers=self.n_layers, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.action_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(1)
        hidden = self.init_hidden(batch_size)

        lstm_out = self.lstm(x, hidden)[0]
        out = self.dropout(lstm_out)
        out = self.fc(out)

        actions = self.sigmoid(out)

        return actions

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(),
                  weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to())
        return hidden


if __name__ == '__main__':
    output_size = 4
    h_size = 512
    n = 2
    input_size = 1028
    model = Actor(input_size, output_size, h_size, n)
    print(model)
    input = torch.randn(1, 1028)
    input = torch.unsqueeze(input, 1)
    y = model(input)
    print(y)
