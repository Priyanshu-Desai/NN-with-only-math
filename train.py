import neural_network as nn
import json
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import random


# the settings for training the ai
epochs = 10
batch_size = 60
learning_rate = 0.01
total_batches = int(60000 / batch_size)

# open json config file if exists
try:
    with open('neural_network.json', 'r') as f:
        config = json.load(f)
        print("Loaded neural network configuration from neural_network.json")
except FileNotFoundError:
    config = {
        '0': {
            'type': 'input',
            'size': 784,
        },
        '1': {
            'type': 'weight',
            'size': [128, 784],
            'weights': [random.uniform(-1, 1) for _ in range(784 * 128)],
        },
        '2': {
            'type': 'middle',
            'size': 128,
            'biases': [random.uniform(-1, 1) for _ in range(128)],
        },
        '3': {
            'type': 'weight',
            'size': [10, 128],
            'weights': [random.uniform(-1, 1) for _ in range(128 * 10)],
        },
        '4': {
            'type': 'output',
            'size': 10,
            'biases': [random.uniform(-1, 1) for _ in range(10)],
        },
    }
    with open('neural_network.json', 'w') as f:
        json.dump(config, f, indent=4)
    print("Created new neural network configuration, saved to neural_network.json")

f.close()

ai = nn.NeuralNetwork(config)

# load the mnist dataset
train_dataset = MNIST(root='./data', train=True, download=True, transform=ToTensor())
print("MNIST dataset loaded successfully.")

# split the dataset into matches of 60 each
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(f"Dataset split into batches of {batch_size}.")

print("---------------------------------")
# loop though all the epochs
for epoch in range(epochs):
    current_epoch = epoch + 1
    print(f"Starting epoch {current_epoch}/{epochs}...")

    total_epoch_loss = 0.0

    # loop through the batches of the epoch
    for batch_idx, (batch_images, batch_labels) in enumerate(train_loader):
        current_batch = batch_idx + 1
        print(f"Starting batch {current_batch}/{total_batches} of epoch {current_epoch}...")

        # hold weights and biases gradients
        batch_weight_grads = None
        batch_bias_grads = None

        total_batch_loss = 0.0

        # loop through the images in the batch
        for i in range(len(batch_images)):
            current_image = i + 1

            
            # get a single image tensor from the batch
            image_tensor = batch_images[i]

            # flatten the image tensor to a 1D vector
            flat_image = image_tensor.view(-1).tolist()

            # convert the flat_image to matrix
            input_matrix = nn.Matrix("784x1", *flat_image)

            # get the label for the image
            label = batch_labels[i].item()

            # train
            ai.setInput(*input_matrix.flatten())
            result, _ = ai.forward()
            prediction = result.flatten().index(max(result.flatten()))
            loss = ai.loss(result, label)
            deltas = ai.backpropogate(result, label)
            weight_grads, bias_grads = ai.gradients()

            # add to running total of gradients
            if batch_weight_grads is None:
                batch_weight_grads = weight_grads
                batch_bias_grads = bias_grads
            else:
                for layer_idx in range(len(weight_grads)):
                    batch_weight_grads[layer_idx] = batch_weight_grads[layer_idx] + weight_grads[layer_idx]
                    batch_bias_grads[layer_idx] = batch_bias_grads[layer_idx] + bias_grads[layer_idx]

            # add to running total of loss
            total_batch_loss += loss
            total_epoch_loss += loss

            print(f"Epoch: {current_epoch}/{epochs}, Batch: {current_batch}/{total_batches}, Image: {current_image}/{batch_size}, Prediction: {prediction}, Label: {label}, Loss: {loss:.4f}")

        # average the gradients over the batch
        avg_weight_grads = [grad.scalar_multiply(1 / batch_size) for grad in batch_weight_grads]
        avg_bias_grads = [grad.scalar_multiply(1 / batch_size) for grad in batch_bias_grads]

        #average the loss over the batch
        avg_batch_loss = total_batch_loss / batch_size

        print(f"Finished batch {current_batch}/{total_batches} of epoch {current_epoch}. Average Loss: {avg_batch_loss:.4f}")

        # update the weights and biases
        ai.update(avg_weight_grads, avg_bias_grads, learning_rate)
        print("AI parameters updated")

        # save the neural network configuration to json file
        with open ("neural_network.json", "w") as f:
            json.dump(ai.save(), f, indent=4)

        print("Neural network configuration saved to neural_network.json")
        print('---------------------------------')