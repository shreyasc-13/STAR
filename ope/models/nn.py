# class NeuralNetwork(nn.Module):
#     def __init__(self, shape, action_space_dim):
#         super().__init__()
#         num_layers = 2
#         hidden_dim = 16
#         self.in_dim, self.out_dim
#         self.in_dim = shape  # the batch dim will be removed during input
#         self.out_dim = action_space_dim
#         layers = []
#         layers.append(nn.Linear(in_dim, hidden_dim))
#         layers.append(nn.ReLU())
#         for _ in range(num_layers-1):
#             layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(nn.ReLU())
#         layers.append(nn.Linear(hidden_dim, out_dim))
#         self.layer_stack = nn.Sequential(*layers)

#     def forward(self, x):
#         logits = self.layer_stack(x)
#         return logits


import torch
import torch.nn as nn


class defaultFFNN(nn.Module):
    def __init__(self, shape, action_space_dim):
        super(defaultFFNN, self).__init__()

        # The first dim is 1 here, and will be width for a CNN. The batch size is trimmed out at input.
        _, input_dim = shape

        # The num layers are hardcoded in here.
        # And so is the hidden dim.
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, action_space_dim)
        )

    def forward(self, state, action):
        # import pdb
        # pdb.set_trace()
        output = self.net(state)
        action = action.unsqueeze(1)  # Adjust action for indexing
        return torch.masked_select(output, action)

    def predict(self, state):
        return self.net(state).squeeze(1)

    def predict_w_softmax(self, state):
        raise "needs some action indexing fix possibly"
        return nn.Softmax()(self.net(state))

    def weight_init(self):
        pass


class defaultModelBasedFFNN(nn.Module):
    def __init__(self, shape, action_space_dim):
        super(defaultModelBasedFFNN, self).__init__()

        _, input_dim = shape

        self.features = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        # FIXME: hacky to be in line with the CNN MB definition
        # For a state as input, the prediction output is [pred_s_a1, pred_s_a2, ...]: input_dim * action_dim
        self.states_head = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim * action_space_dim)
        )

        self.rewards_head = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, action_space_dim)
        )

        self.dones_head = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, action_space_dim),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        features = self.features(state)
        T = self.states_head(features)
        R = self.rewards_head(features)
        D = self.dones_head(features)

        action = action.unsqueeze(1)  # Adjust action for indexing

        state_dim = state.shape[-1]  # input_dim

        T_out = []
        for i, a in enumerate(action.float().argmax(2).squeeze(1)):
            idx = a.item()
            T_out.append(T.squeeze(1)[i, idx*state_dim: (idx+1)*state_dim])

            # start_indices = [
            #     a.item() * state_dim for a in action.float().argmax(2).squeeze(1)]
            # end_indices = [(a.item() + 1) *
            #    state_dim for a in action.float().argmax(2).squeeze(1)]
        T_out = torch.stack(T_out)

        # import pdb
        # pdb.set_trace()

        return T_out.unsqueeze(1), torch.masked_select(R, action), torch.masked_select(D, action)
        # return T[torch.arange(len(action)), action.view(-1), :], torch.masked_select(R, action), torch.masked_select(D, action)

    def predict(self, state):
        raise "same changes as above needed"
        features = self.features(state)
        T = self.states_head(features)
        R = self.rewards_head(features)
        D = self.dones_head(features)
        return T, R, D

    def weight_init(self):
        pass


class NeuralNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, hidden_dim=16):
        super().__init__()
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.in_dim = in_dim
        self.out_dim = out_dim
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.layer_stack = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.layer_stack(x)
        return logits
