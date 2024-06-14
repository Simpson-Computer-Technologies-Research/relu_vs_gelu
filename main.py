##
## Let's do a fun comparison between ReLU and GeLU (activation functions)
##
## GeLU is very similar to ReLU, however, it can sometimes capture more
## complex features of data.
##
from torch import nn
import torch
import numpy as np
from _types import Activation


class NN(nn.Module):
    def __init__(self, activation: Activation = "ReLU"):
        super(NN, self).__init__()

        self.activation = activation

        ##
        ## Basically, we're going to use a neural network to predict a person's
        ## age based on their height, weight, and whether they're male or female
        ##
        ## We want 3 inputs and 1 output
        ##
        ## We'll stick to a simple 3-layer MLP (linear -> activation -> linear -> activation -> linear -> activation)
        ##
        self.gelu_sequence = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.relu_sequence = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "GeLU":
            return self.gelu_sequence(x)
        else:
            return self.relu_sequence(x)


if __name__ == "__main__":
    gelu_net = NN(activation="GeLU")
    relu_net = NN(activation="ReLU")

    train_data = torch.Tensor(
        [
            [1.70, 70, 1],
            [1.60, 50, 0],
            [1.80, 80, 1],
            [1.85, 90, 1],
            [1.75, 75, 0],
            [1.65, 55, 0],
        ]
    )
    train_labels = torch.Tensor([[25], [20], [30], [35], [27], [22]])

    test_data = torch.Tensor(
        [
            [1.75, 80, 1],
            [1.65, 55, 0],
        ]
    )
    test_labels = torch.Tensor([[30], [22]])

    epochs = 1000

    ##
    ## Let's test the GeLU network
    ##
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(gelu_net.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = gelu_net(train_data)
        loss = criterion(output, train_labels)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item()}")

    with torch.no_grad():
        output = gelu_net(test_data)
        print(f"\nTest Loss (GeLU): {criterion(output, test_labels)}\n")

    ##
    ## Let's test the ReLU network
    ##
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(relu_net.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = relu_net(train_data)
        loss = criterion(output, train_labels)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item()}")

    with torch.no_grad():
        output = relu_net(test_data)
        print(f"\nTest Loss (ReLU): {criterion(output, test_labels)}\n")

    print("GeLU Network:")
    print(gelu_net(test_data))
    print("ReLU Network:")
    print(relu_net(test_data))

##
## Example output:
##
## Test Loss (GeLU): 0.08334237337112427
## Test Loss (ReLU): 0.45898932218551636
##


##
## End of file
##
