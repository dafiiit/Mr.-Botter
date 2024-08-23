import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from sklearn.cluster import KMeans


class ComplexNetwork(nn.Module):
    def __init__(self, hyperspace_dim, time_dim, memory_size, expert_net, stem_net):
        """
        Initialize the complex network with the given dimensions and networks.
        Input: hyperspace_dim - dimensionality of the hyperspace
                time_dim - dimensionality of the time input
                memory_size - size of the LSTM memory
                expert_net - expert network for short-term information
                stem_net - stem network for long-term information
        Output: None
        """
        super(ComplexNetwork, self).__init__()
        self.hyperspace_dim = hyperspace_dim
        self.time_dim = time_dim
        self.memory_size = memory_size

        # LSTM for short-term information
        self.lstm = nn.LSTM(hyperspace_dim + time_dim, memory_size, batch_first=True)

        # Expert network and stem network
        self.expert_net = expert_net
        self.stem_net = stem_net

        # Add random neurons
        self.add_random_neurons(10)

        # Second stem neuron region
        self.fusion_net = nn.Linear(2 * hyperspace_dim, hyperspace_dim)

    def add_random_neurons(self, num_neurons):
        """
        Add random neurons to the expert network.
        Input: num_neurons - number of neurons to add
        Output: None
        """
        for _ in range(num_neurons):
            if random.choice([True, False]):
                # Add in depth
                new_layer = nn.Linear(self.hyperspace_dim, self.hyperspace_dim)
                self.expert_net.add_module(
                    f"random_layer_{len(self.expert_net)}", new_layer
                )
            else:
                # Add in width
                last_layer = list(self.expert_net.children())[-1]
                if isinstance(last_layer, nn.Linear):
                    new_layer = nn.Linear(
                        last_layer.in_features, last_layer.out_features + 1
                    )
                    new_layer.weight.data[:, :-1] = last_layer.weight.data
                    new_layer.weight.data[:, -1] = torch.randn_like(
                        new_layer.weight.data[:, 0]
                    )
                    new_layer.bias.data[:-1] = last_layer.bias.data
                    new_layer.bias.data[-1] = torch.randn(1)
                    self.expert_net[-1] = new_layer

    def forward(self, hyperspace, time):
        """
        Forward pass through the network.
        Input: hyperspace - input tensor for the hyperspace
                time - input tensor for the time
        Output: fused_out - output tensor from the network
        """
        # Combine hyperspace and time
        combined_input = torch.cat((hyperspace, time), dim=1)

        # LSTM for short-term information
        lstm_out, _ = self.lstm(combined_input.unsqueeze(1))
        lstm_out = lstm_out.squeeze(1)

        # Run expert network and stem network in parallel
        expert_out = self.expert_net(combined_input)
        stem_out = self.stem_net(combined_input)

        # Fuse outputs
        fused_out = self.fusion_net(torch.cat((expert_out, stem_out), dim=1))

        return fused_out


def train_step(model, optimizer, hyperspace, time, target):
    """
    Perform a single training step on the given model.
    Input: model - complex network model
            optimizer - optimizer for the model
            hyperspace - input tensor for the hyperspace
            time - input tensor for the time
            target - target tensor for the model
    Output: loss - loss value for the model
    """
    optimizer.zero_grad()
    output = model(hyperspace, time)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()


def prune_neurons(net, prune_rate):
    """
    Prune a percentage of the neurons in the given network by removing them from the architecture.
    Input: net - network to prune
           prune_rate - percentage of neurons to prune
    Output: new_net - pruned network
    """
    new_layers = []

    for module in net.children():
        if isinstance(module, nn.Linear):
            # Calculate the number of neurons to prune
            weight = module.weight.data.abs()
            num_neurons = module.out_features
            num_to_prune = int(prune_rate * num_neurons)

            if num_to_prune > 0:
                # Identify the neurons with the smallest L1-norm to prune
                l1_norms = weight.norm(p=1, dim=1)
                _, indices_to_prune = torch.topk(l1_norms, num_to_prune, largest=False)

                # Create new weight and bias tensors with the pruned neurons removed
                keep_indices = list(
                    set(range(num_neurons)) - set(indices_to_prune.tolist())
                )

                new_out_features = len(keep_indices)
                new_weight = module.weight.data[keep_indices, :]
                new_bias = (
                    module.bias.data[keep_indices] if module.bias is not None else None
                )

                # Create a new Linear layer with pruned dimensions
                new_layer = nn.Linear(
                    module.in_features, new_out_features, bias=module.bias is not None
                )
                new_layer.weight.data = new_weight
                if new_bias is not None:
                    new_layer.bias.data = new_bias

                new_layers.append(new_layer)
            else:
                # If nothing to prune, keep the layer as is
                new_layers.append(module)
        else:
            # Keep other layers unchanged
            new_layers.append(module)

    # Create a new Sequential model with pruned layers
    new_net = nn.Sequential(*new_layers)
    return new_net


def calculate_neuron_topic(
    model, layer_idx, neuron_idx, num_iterations=1000, learning_rate=0.1
):
    """
    Calculate the topic of a neuron in the given model.
    Input: model - complex network model
            layer_idx - index of the layer containing the neuron
            neuron_idx - index of the neuron in the layer
            num_iterations - number of optimization iterations
            learning_rate - learning rate for optimization
    Output: hyperspace - tensor representing the topic of the neuron
    """
    hyperspace = torch.randn(1, model.hyperspace_dim, requires_grad=True)
    time = torch.zeros(1, model.time_dim)

    optimizer = optim.Adam([hyperspace], lr=learning_rate)

    for _ in range(num_iterations):
        optimizer.zero_grad()
        combined_input = torch.cat((hyperspace, time), dim=1)
        activation = model.expert_net[: layer_idx + 1](combined_input)
        neuron_activation = activation[0, neuron_idx]
        loss = -neuron_activation
        loss.backward()
        optimizer.step()

    return hyperspace.detach()


def split_network(model, num_clusters=2):
    """
    Split the given model into multiple sub-networks based on neuron topics.
    Input: model - complex network model
            num_clusters - number of sub-networks to create
    Output: new_networks - list of sub-networks
    """
    topics = []
    neuron_indices = []

    for layer_idx, layer in enumerate(model.expert_net):
        if isinstance(layer, nn.Linear):
            for neuron_idx in range(layer.out_features):
                topic = calculate_neuron_topic(model, layer_idx, neuron_idx)
                topics.append(topic.squeeze().numpy())
                neuron_indices.append((layer_idx, neuron_idx))

    topics_array = np.array(topics)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(topics_array)

    new_networks = [nn.Sequential() for _ in range(num_clusters)]

    for cluster in range(num_clusters):
        cluster_neurons = [
            i for i, label in enumerate(cluster_labels) if label == cluster
        ]

        for layer_idx, layer in enumerate(model.expert_net):
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                out_features = sum(
                    1 for i in cluster_neurons if neuron_indices[i][0] == layer_idx
                )

                if out_features > 0:
                    new_layer = nn.Linear(in_features, out_features)
                    new_networks[cluster].add_module(f"layer_{layer_idx}", new_layer)

                    neuron_count = 0
                    for i in cluster_neurons:
                        if neuron_indices[i][0] == layer_idx:
                            orig_neuron_idx = neuron_indices[i][1]
                            new_layer.weight.data[neuron_count] = layer.weight.data[
                                orig_neuron_idx
                            ]
                            new_layer.bias.data[neuron_count] = layer.bias.data[
                                orig_neuron_idx
                            ]
                            neuron_count += 1

            elif isinstance(layer, nn.ReLU):
                new_networks[cluster].add_module(f"relu_{layer_idx}", nn.ReLU())

    print(f"Network split into {num_clusters} sub-networks:")
    for i, network in enumerate(new_networks):
        print(f"Sub-network {i+1}:")
        print(network)
        print(
            f"Number of neurons: {sum(layer.out_features for layer in network if isinstance(layer, nn.Linear))}"
        )
        print()

    return new_networks


# Example usage
def main():
    hyperspace_dim = 100
    time_dim = 10
    memory_size = 50
    expert_net = nn.Sequential(
        nn.Linear(hyperspace_dim + time_dim, 64),
        nn.ReLU(),
        nn.Linear(64, hyperspace_dim),
    )
    stem_net = nn.Sequential(
        nn.Linear(hyperspace_dim + time_dim, 64),
        nn.ReLU(),
        nn.Linear(64, hyperspace_dim),
    )

    model = ComplexNetwork(hyperspace_dim, time_dim, memory_size, expert_net, stem_net)
    optimizer = optim.Adam(model.parameters())

    # Training cycle
    for epoch in range(100):
        hyperspace = torch.randn(1, hyperspace_dim)
        time = torch.randn(1, time_dim)
        target = torch.randn(1, hyperspace_dim)

        loss = train_step(model, optimizer, hyperspace, time, target)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
            prune_rate = 0.1 if loss < 0.5 else 0.05
            prune_neurons(model.expert_net, prune_rate)
            prune_neurons(model.stem_net, prune_rate)

        if epoch % 50 == 0:
            split_network(model)

    print("Training completed")


if __name__ == "__main__":
    main()
