import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, actions):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q, actions], 2)))
        return self.l3(q)


if __name__ == '__main__':
    input_size = 1028
    output_size = 4
    model = Critic(input_size, output_size)
    print(model)

    input = torch.randn(1, input_size)
    input = torch.unsqueeze(input, 1)
    output = torch.rand(1, output_size)
    output = torch.unsqueeze(output, 1)

    y = model(input, output)
    print(y)