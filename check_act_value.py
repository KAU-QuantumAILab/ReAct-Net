import torch
from torch.nn import ReLU
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from models import BReLU

# Forward hook function to extract activations before and after ReLU
def hook_fn_pre(module, input, output, storage, index, model_name):
    storage[f'{model_name}_pre_act_{index}'] = input[0].detach().cpu().numpy()

def hook_fn_post(module, input, output, storage, index, model_name):
    storage[f'{model_name}_post_act_{index}'] = output.detach().cpu().numpy()

# Function to register hooks for all ReLU layers
def register_hooks(model, storage, model_name, activation):
    index = 0
    for layer in model.modules():
        if isinstance(layer, activation):  # Check for ReLU layers
            layer.register_forward_hook(lambda m, i, o, idx=index: hook_fn_pre(m, i, o, storage, idx, model_name))
            layer.register_forward_hook(lambda m, i, o, idx=index: hook_fn_post(m, i, o, storage, idx, model_name))
            index += 1

# Function to visualize activation maps and negative value ratio
def visualize_activations(storage, model_name, ax, num_layers):
    for i in range(num_layers):
        pre_act = storage[f'{model_name}_pre_act_{i}']
        post_act = storage[f'{model_name}_post_act_{i}']

        # Calculate negative ratio
        neg_ratio_pre = np.mean(pre_act < 0)
        neg_ratio_post = np.mean(post_act < 0)

        # Flatten and visualize as heatmap
        sns.heatmap(pre_act.mean(axis=(0, 1)), ax=ax[i*2], cmap='coolwarm', cbar=False)
        ax[i*2].set_title(f'{model_name} Pre-Act {i+1}\nNeg %: {neg_ratio_pre:.2%}')

        sns.heatmap(post_act.mean(axis=(0, 1)), ax=ax[i*2+1], cmap='coolwarm', cbar=False)
        ax[i*2+1].set_title(f'{model_name} Post-Act {i+1}\nNeg %: {neg_ratio_post:.2%}')

# Main function to compare models and visualize activations
def compare_models(model1, model2, image, device):
    model1.to(device).eval()
    model2.to(device).eval()
    
    # Storage for activations
    storage = {}

    # Register hooks
    register_hooks(model1, storage, 'ReLU', ReLU)
    register_hooks(model2, storage, 'BReLU', BReLU)

    # Move image to device
    image = image.to(device).unsqueeze(0)  # Add batch dimension

    # Forward pass through both models
    with torch.no_grad():
        _ = model1(image)
        _ = model2(image)

    # Plot activations
    num_relu_layers = 17
    fig, axes = plt.subplots(num_relu_layers, 4, figsize=(20, num_relu_layers * 2))

    # Visualize for both models
    visualize_activations(storage, 'ReLU', axes[:, :2], num_relu_layers)
    visualize_activations(storage, 'BReLU', axes[:, 2:], num_relu_layers)

    plt.tight_layout()
    plt.show()

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image = torch.randn(3, 32, 32)  # Simulating a CIFAR10 image, replace with actual image
model1 = ...  # Load your first pretrained CIFAR10 model
model2 = ...  # Load your second pretrained CIFAR10 model

compare_models(model1, model2, image, device)
