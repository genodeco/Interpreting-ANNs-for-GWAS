import torch.nn as nn
import torch.optim as optim

class Classifier(nn.Module):
    def __init__(self, data_shape, init_dropout):
        super(Classifier, self).__init__()

        self.data_shape = data_shape
        self.init_dropout = init_dropout

        self.layers = nn.Sequential(
        nn.Dropout(p=init_dropout, inplace=False),
        nn.Linear(data_shape, int(data_shape//1000)),
        nn.Dropout(p=0.6, inplace=False),
        nn.ReLU(),

        nn.Linear(int(data_shape//1000), int(data_shape//10000)),
        nn.Dropout(p=0.6, inplace=False),
        nn.GELU(),

        nn.Linear(int(data_shape//10000), 1),
        nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x
