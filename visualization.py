import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image
import yaml
from unet import UNet
from utils import DiceLoss,load_config


class CarotidSegmentation:
    """Class for loading and generating predictions via trained segmentation model."""

    def __init__(self, model='unet_1'):  # Default set to 'unet_1'
        # Define the config directory and model config filename
        config_path = r'C:\Users\vikas\OneDrive\Desktop\SamanyahackthonDSE'
        config_file = 'unet_1.yaml'
        config_full_path = os.path.join(config_path, config_file)

        # Load configuration
        print(f"Loading configuration from: {config_full_path}")
        try:
            config = self.load_config(config_full_path)
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")

        # Define image transforms
        self._image_transforms = transforms.Compose([transforms.CenterCrop(512)])

        # Initialize UNet model with loaded configuration
        print("Initializing the UNet model...")
        self.net = UNet(
            input_channels=config['in_channels'],
            output_channels=config['n_classes'],
            depth=config['depth'],
            batch_norm=config['batch_norm'],
            padding=config['padding'],
            
            up_mode=config['up_mode']
        )

        # Load pre-trained model weights
        weights_path = r'C:\Users\vikas\OneDrive\Desktop\SamanyahackthonDSE\model.h5'
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights file not found: {weights_path}")

        print(f"Loading model weights from: {weights_path}")
        self.load_keras_weights(weights_path)
        self.net.eval()

    def load_config(self, config_file):
        """Load configuration from a YAML file."""
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def load_keras_weights(self, h5_file):
        """Transfers weights from a Keras model to the PyTorch UNet model."""
        keras_model = tf.keras.models.load_model(h5_file)
        keras_layers = keras_model.layers
        pytorch_layers = list(self.net.modules())[1:]  # Access the PyTorch model layers here

        # Initialize a list to keep track of the PyTorch layers that accept weights
        pytorch_layers_to_map = [layer for layer in pytorch_layers if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d)]

        for i, (keras_layer, pytorch_layer) in enumerate(zip(keras_layers, pytorch_layers_to_map)):
            print(f"Mapping layer {i}: {keras_layer.name} -> {type(pytorch_layer).__name__}")

            if isinstance(pytorch_layer, nn.Conv2d):
                keras_weights = keras_layer.get_weights()

                if len(keras_weights) > 0:
                    keras_kernel = keras_weights[0]
                    keras_bias = keras_weights[1] if len(keras_weights) > 1 else None
                       # Convert kernel from Keras (H, W, In, Out) to PyTorch (Out, In, H, W)
                    # Convert Keras kernel shape to PyTorch format
                    if len(keras_kernel.shape) == 4:
                        pytorch_kernel = torch.tensor(keras_kernel).permute(3, 2, 0, 1).float()

                        # Check if shapes match
                        if pytorch_kernel.shape == pytorch_layer.weight.shape:
                            with torch.no_grad():
                                pytorch_layer.weight.copy_(pytorch_kernel)
                        else:
                            print(f"Skipping layer {i}: Mismatched kernel shapes! Keras: {keras_kernel.shape}, PyTorch: {pytorch_layer.weight.shape}")

                        # Handle bias
                        if keras_bias is not None and pytorch_layer.bias is not None:
                            pytorch_bias = torch.tensor(keras_bias).float()
                            if pytorch_bias.shape == pytorch_layer.bias.shape:
                                with torch.no_grad():
                                    pytorch_layer.bias.copy_(pytorch_bias)
                            else:
                                print(f"Skipping bias for layer {i}: Mismatched bias shapes!")
                    else:
                        print(f"Unexpected kernel shape for layer {i}: {keras_kernel.shape}")

            elif isinstance(pytorch_layer, nn.BatchNorm2d):
                # Handle BatchNorm2D layers
                if hasattr(keras_layer, 'gamma'):
                    with torch.no_grad():
                        pytorch_layer.weight.copy_(torch.tensor(keras_layer.gamma.numpy()).float())
                        pytorch_layer.bias.copy_(torch.tensor(keras_layer.beta.numpy()).float())
                        pytorch_layer.running_mean.copy_(torch.tensor(keras_layer.moving_mean.numpy()).float())
                        pytorch_layer.running_var.copy_(torch.tensor(keras_layer.moving_variance.numpy()).float())
                else:
                    print(f"Skipping BatchNorm layer {i}: Keras layer has no BatchNorm parameters.")

        self.net.eval()
        print("Keras weights successfully loaded into PyTorch model.")

    def get_image(self, image_loc):
        """Loads and preprocesses the input image."""
        if not os.path.exists(image_loc):
            raise FileNotFoundError(f"Image file not found: {image_loc}")

        image = read_image(image_loc).float()
        if image.shape[0] == 3:
            print("Converting RGB image to Grayscale.")
            image = transforms.functional.rgb_to_grayscale(image)

        if image.shape[0] != self.net.input_channels:
            raise ValueError(f"Image channel mismatch. Expected {self.net.input_channels}, got {image.shape[0]}")

        image = self._image_transforms(image)
        return image.unsqueeze(0)  # Add batch dimension

    def get_label(self, image_loc):
        """Loads and preprocesses the label (mask)."""
        mask_loc = image_loc.replace("US images", "Expert mask images")
        if not os.path.exists(mask_loc):
            raise FileNotFoundError(f"Mask file not found: {mask_loc}")
        mask = read_image(mask_loc).float().unsqueeze(0)
        mask = self._image_transforms(mask) > 0
        return mask.type(torch.int8)

    def predict(self, image_loc):
        """Generates predictions for the input image."""
        image = self.get_image(image_loc)
        print(f"Image shape: {image.shape}")  # Debugging print to check image shape

        preds = self.net(image)
        
        if preds is None:
            print("Prediction is None!")
        else:
            # Add Debugging Code Here
            print("Prediction Output Stats:")
            print(f"Shape: {preds.shape}, Min: {preds.min().item()}, Max: {preds.max().item()}")
            
        return preds
    def eval_model(self, image_loc):
        """Evaluates the model on a given input."""
        pred = self.predict(image_loc)

        if pred is None:
            print("Prediction returned None!")
            return None  # Handle case where prediction is None
        
        label = self.get_label(image_loc)
        
        if label is None:
            print("Label is None!")
            return None  # Handle case where label is None
        
        loss = DiceLoss()(pred, label)
        
        if loss is None:
            print("Loss is None!")
            return None  # Handle case where loss computation failed
        
        return loss.item()  # Ensure this is only called if loss is not None

    def analyze_plaque(self, extracted_background, plot=True):
        """Analyzes plaque regions in the artery and visualizes the results."""
        valid_pixels = extracted_background[~np.isnan(extracted_background)]
        mean_intensity = np.mean(valid_pixels)
        std_dev_intensity = np.std(valid_pixels)
        threshold = mean_intensity + 0.001 * std_dev_intensity

        plaque_mask = extracted_background >= threshold
        total_artery_area = np.sum(~np.isnan(extracted_background))
        plaque_area = np.sum(plaque_mask)
        plaque_percentage = (plaque_area / total_artery_area) * 100

        if plot:
            plt.figure(figsize=(10, 5))
            plt.imshow(extracted_background, cmap='Greys_r', alpha=1)
            plt.imshow(np.where(plaque_mask, extracted_background, np.nan), cmap='Blues', alpha=0.6)
            plt.colorbar(label="Pixel Intensity")
            plt.title("Plaque Analysis with Results")
            # Add analysis results as text annotations
            plt.text(
                0.05, 0.95,
                f"Total Area: {total_artery_area}\nPlaque Area: {plaque_area}\nPlaque %: {plaque_percentage:.2f}%\nThreshold: {threshold:.2f}",
                color='white', fontsize=10, transform=plt.gca().transAxes, bbox=dict(facecolor='black', alpha=0.5)
            )
            plt.show()

        return {
            "total_artery_area": total_artery_area,
            "plaque_area": plaque_area,
            "plaque_percentage": plaque_percentage,
            "threshold": threshold,
            "plaque_mask": plaque_mask,
        }

    def plot_pred_and_analyze(self, image_loc, labels=False, text=True, image_name=None):
        """Plots model prediction and analyzes plaque on a single page."""
        # Get the image and prediction
        image = self.get_image(image_loc)
        preds = self.predict(image_loc)
        pred_out = preds[0][0].detach().numpy()
        background = image[0][0].detach().numpy()

        # Normalize the prediction output
        pred_out = np.clip(pred_out, 0, 1)

        # Analyze plaque on the background
        plaque_analysis = self.analyze_plaque(background, plot=False)  # Analyze but don't plot yet
        plaque_mask = plaque_analysis["plaque_mask"]

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))  # Two plots side-by-side

        # Plot the prediction
        axes[0].imshow(background, cmap='Greys_r', alpha=1)
        axes[0].imshow(pred_out, cmap='YlOrRd', alpha=pred_out * 0.5)
        axes[0].set_title('Carotid Artery Segmentation')
        if labels:
            label_out = self.get_label(image_loc)[0][0].detach().numpy()
            label_out = np.clip(label_out, 0, 1)
            axes[0].imshow(label_out, cmap='RdYlBu', alpha=label_out * 0.5)
            if text:
                dice_loss = round(self.eval_model(image_loc), 4)
                axes[0].set_xlabel(f'Prediction = Red, True Label = Blue \n Dice Loss: {dice_loss}')
        elif text:
            axes[0].set_xlabel('Prediction = Red')

        # Plot the plaque analysis
        axes[1].imshow(background, cmap='Greys_r', alpha=1)
        axes[1].imshow(plaque_mask, cmap='Blues', alpha=0.6)
        axes[1].set_title('Plaque Analysis')
        axes[1].text(
            1, 1,
            f"Total Area in pixels: {plaque_analysis['total_artery_area']}\n"
            f"Plaque Area in pixels: {plaque_analysis['plaque_area']}\n"
            f"Plaque in  %: {plaque_analysis['plaque_percentage']:.2f}%\n"
            f"Threshold (limit): {plaque_analysis['threshold']:.2f}",
            color='white', fontsize=10, transform=axes[1].transAxes, bbox=dict(facecolor='black', alpha=0.5)
        )

        plt.tight_layout()
        plt.show()
        return plaque_analysis

# Main block
if __name__ == "__main__":
    try:
        # Initialize the CarotidSegmentation class
        segmentation = CarotidSegmentation(model='unet_1')  # Adjust model name/path if needed
        
        # Define the path to the sample image
        sample_image = r'C:\Users\vikas\OneDrive\Desktop\SamanyahackthonDSE\Common Carotid Artery Ultrasound Images\US images\202201121748100022VAS_slice_1419.png'  # Replace with actual image path
        
        # Get predicted background and analyze
        plaque_analysis_results = segmentation.plot_pred_and_analyze(sample_image, labels=True)
        print("Plaque Analysis Results:", plaque_analysis_results)
            
    except Exception as e:
        print(f"Error occurred: {e}")
