import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
import torchinfo
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import cv2
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ==================================
# Global Constants for MFCCs
# ==================================
SAMPLE_RATE = 22050
N_MFCC = 40        # Number of MFCC coefficients - THIS IS THE KEY CHANGE FROM N_MELS
N_FFT = 2048       # Window size for FFT
HOP_LENGTH = 512   # Hop length for FFT
MAX_TIME_STEPS = 400 # Max number of time steps (width of the feature image)

# ==================================
# Custom CNN Model Definition (with dynamic linear layer size)
# ==================================
class CustomCNN(nn.Module):
    def __init__(self, num_classes=2): # Added num_classes parameter for flexibility
        super(CustomCNN, self).__init__()
        
        # Define the convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1: Input (1, N_MFCC, MAX_TIME_STEPS) e.g., (1, 40, 400)
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # Output: (32, 40, 400)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), # Output: (32, 20, 200)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Output: (64, 20, 200)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # Output: (64, 10, 100)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Output: (128, 10, 100)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # Output: (128, 5, 50)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # Output: (256, 5, 50)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # Output: (256, 3, 25)
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Dynamically calculate the flattened size for the linear layer
        # This is the most reliable way to avoid shape mismatch errors
        with torch.no_grad(): # Disable gradient computation for this dummy pass
            # Create a dummy input tensor matching the expected input shape for the model
            # Batch size is 1 for this calculation.
            dummy_input = torch.rand(1, 1, N_MFCC, MAX_TIME_STEPS) 
            dummy_output = self.features(dummy_input)
            # Calculate the number of elements per sample after flattening
            # dummy_output.numel() is total elements, divide by batch size (1)
            linear_input_size = dummy_output.numel() // dummy_output.shape[0] 
        
        print(f"Calculated linear layer input size: {linear_input_size}") # Confirm the calculated size

        # Define the classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(), # Flattens the output of features (e.g., from (256, 3, 25) to (19200))
            nn.Linear(linear_input_size, 64), # Connect to the first fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5), # Regularization
            nn.Linear(64, num_classes) # Output layer for classification (e.g., 2 classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ==================================
# Audio Dataset Class (adapted for MFCCs)
# ==================================
class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.label_map = {}

        # Scan root_dir for subdirectories (classes)
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.label_map[class_name] = idx
                # Scan each class directory for .wav files
                for file_name in os.listdir(class_path):
                    if file_name.endswith('.wav'):
                        self.samples.append((os.path.join(class_path, file_name), idx))
        self.classes = sorted(self.label_map.keys()) # Store class names for evaluation

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        try:
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            # Calculate MFCCs (instead of Mel Spectrogram)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)

            # Resize MFCCs to a fixed size (MAX_TIME_STEPS wide, N_MFCC high)
            # Note: OpenCV's resize expects (width, height)
            mfcc = cv2.resize(mfcc, (MAX_TIME_STEPS, N_MFCC)) # Shape: (N_MFCC, MAX_TIME_STEPS)

            # Normalization (mean 0, std 1)
            # Add a small epsilon to std to prevent division by zero for silent/constant audio
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

            # Convert to PyTorch tensor and add a channel dimension (for CNN input)
            # Shape becomes: (1, N_MFCC, MAX_TIME_STEPS)
            feature_data = torch.from_numpy(mfcc).float().unsqueeze(0)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Return a zero tensor and a -1 label if an error occurs during loading/processing
            feature_data = torch.zeros((1, N_MFCC, MAX_TIME_STEPS), dtype=torch.float32)
            label = -1 # Use -1 to mark invalid samples
        return feature_data, label

# ==================================
# Main Training and Evaluation Function
# ==================================
def main(args):
    print("Code execution started.")

    # Initialize Weights & Biases run
    run = wandb.init(project="vocalsound_mfcc", config=vars(args)) # New project name
    config = wandb.config

    # Define paths to your dataset directories
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    train_path = os.path.join(script_dir, 'datasets', 'train')
    test_path = os.path.join(script_dir, 'datasets', 'test')

    print("Data loading initiated...")

    # Create dataset instances
    train_data = AudioDataset(train_path)
    test_data = AudioDataset(test_path)

    # Determine the number of classes from the training dataset
    num_classes = len(train_data.classes)
    run.config['num_classes'] = num_classes

    # Log class names to W&B config
    if hasattr(train_data, 'classes'):
        run.config['class_names'] = train_data.classes
        print(f"Detected classes: {train_data.classes}")
    else:
        print("Warning: Class names not found in dataset. Ensure AudioDataset sets 'self.classes'.")


    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of test samples: {len(test_data)}")
    print("Data loading complete.")

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    # Initialize the model with the determined number of classes
    model = CustomCNN(num_classes=num_classes)

    # Print model summary using torchinfo
    # Input shape: (batch_size, channels, height, width) for torchinfo
    summary = torchinfo.summary(model, input_size=(config.batch_size, 1, N_MFCC, MAX_TIME_STEPS))
    run.config['total_params'] = summary.total_params
    run.config['mult_adds'] = summary.total_mult_adds
    print("\nModel Summary:")
    print(summary)
    print("Model initialization complete.")

    # Set up device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    # Define loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss() # Suitable for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 5

    print("Beginning model training...")
    for epoch in range(config.epochs):
        model.train() # Set model to training mode
        total_train_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Filter out samples where data loading failed (label == -1)
            valid_indices = (labels != -1)
            if not valid_indices.any(): # If all samples in batch failed
                # print(f"Skipping training batch {batch_idx} due to all samples having label -1.")
                continue

            inputs, labels = inputs[valid_indices].to(device), labels[valid_indices].to(device)

            optimizer.zero_grad()      # Clear previous gradients
            outputs = model(inputs)    # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward()            # Backward pass
            optimizer.step()           # Update model parameters
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval() # Set model to evaluation mode
        correct, total = 0, 0
        total_test_loss = 0.0
        all_labels = []
        all_predictions = []
        all_probabilities = [] # For ROC AUC, if binary

        print("Evaluating on test set...")
        with torch.no_grad(): # Disable gradient calculation for inference
            for inputs, labels in test_loader:
                # Filter out samples where data loading failed
                valid_indices = (labels != -1)
                if not valid_indices.any():
                    # print(f"Skipping test batch {batch_idx} due to all samples having label -1.")
                    continue

                inputs = inputs[valid_indices].to(device)
                labels = labels[valid_indices].to(device)

                if inputs.size(0) == 0: # Ensure there are valid inputs left after filtering
                    continue

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()

                probabilities = torch.softmax(outputs, dim=1) # Get probabilities for each class
                _, predicted = torch.max(outputs, 1) # Get the class with the highest probability

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                # For binary classification ROC, collect probabilities of the positive class (class 1)
                if num_classes == 2:
                    all_probabilities.extend(probabilities[:, 1].cpu().numpy())
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_test_loss = total_test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        scheduler.step(avg_test_loss) # Adjust learning rate based on test loss

        # Early Stopping logic
        if avg_test_loss < best_val_loss:
            best_val_loss = avg_test_loss
            patience_counter = 0
            # Save the best model checkpoint
            torch.save(model.state_dict(), os.path.join(config.chkpt_path, run.id + '_best_model.pt'))
            print(f"[{epoch+1}/{config.epochs}] Validation loss improved ({best_val_loss:.4f}). Saving model.")
        else:
            patience_counter += 1
            print(f"[{epoch+1}/{config.epochs}] Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break # Exit training loop

        # Log metrics to Weights & Biases for the current epoch
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'test_accuracy': accuracy,
            'lr': optimizer.param_groups[0]['lr'],
        })

        print(f"[{epoch+1}/{config.epochs}] Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    print("\nTraining complete. Performing final evaluation.")

    # Load the best model for final evaluation
    model_save_path = os.path.join(config.chkpt_path, run.id + '_best_model.pt')
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        print("Loaded best model for final evaluation.")
    else:
        print("Best model checkpoint not found. Using the last trained model for final evaluation.")

    # Convert lists to numpy arrays for sklearn metrics
    y_true = np.array(all_labels)
    y_pred_classes = np.array(all_predictions)
    y_scores = np.array(all_probabilities)

    # Handle cases where no valid samples were processed in the test set
    if len(y_true) == 0:
        print("No valid samples for final evaluation. Skipping metrics and plots.")
        wandb.finish()
        return

    # Determine target names for classification report and confusion matrix
    if len(test_data.classes) == num_classes:
        target_names = test_data.classes
    else:
        unique_true_labels = np.unique(y_true)
        target_names = [f'Class {l}' for l in unique_true_labels]

    # ==================================
    # Evaluation Metrics and Visualizations
    # ==================================
    print("\n─────── Classification Report ───────")
    # zero_division=0 prevents warnings when a class has no predicted samples
    report = classification_report(y_true, y_pred_classes, target_names=target_names, output_dict=True, zero_division=0)
    print(json.dumps(report, indent=4)) # Pretty print JSON report

    # Log individual metrics from the classification report to W&B
    wandb_log_metrics = {}
    for class_name, metrics in report.items():
        if isinstance(metrics, dict): # For per-class metrics
            for metric_name, value in metrics.items():
                if metric_name in ['precision', 'recall', 'f1-score', 'support']:
                    wandb_log_metrics[f'classification_report/{class_name}_{metric_name}'] = value
        elif class_name in ['accuracy', 'macro avg', 'weighted avg']: # For overall metrics
            if class_name == 'accuracy':
                wandb_log_metrics[f'classification_report/accuracy'] = metrics
            else:
                for metric_name, value in metrics.items():
                    if metric_name in ['precision', 'recall', 'f1-score', 'support']:
                        wandb_log_metrics[f'classification_report/{class_name.replace(" ", "_")}_{metric_name}'] = value
    
    wandb.log(wandb_log_metrics)
    # Log classification report as an HTML table for better readability in W&B
    wandb.log({"classification_report_text": wandb.Html(classification_report(y_true, y_pred_classes, target_names=target_names, zero_division=0))})


    print("\n─────── Confusion Matrix ───────")
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    wandb.log({"confusion_matrix": wandb.Image(plt)}) # Log plot to W&B
    plt.show()
    plt.close()

    # ROC Curve (only applicable for binary classification)
    # This part should be updated to handle multi-class if num_classes > 2
    if num_classes == 2:
        print("\n─────── ROC Curve ───────")
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            wandb.log({"roc_curve": wandb.Image(plt)}) # Log plot to W&B
            plt.show()
            plt.close()

            print(f"✅ Model AUC Score: {roc_auc:.4f}")
            wandb.log({'final_auc_score': roc_auc})
        except ValueError as e:
            print(f"Could not plot ROC curve: {e}. Ensure there are at least two classes present in the true labels and predicted scores for ROC calculation.")
            wandb.log({'final_auc_score': 'N/A'}) # Log 'N/A' if ROC fails
    else:
        print("\nSkipping ROC Curve: ROC is typically for binary classification.")
        wandb.log({'final_auc_score': 'N/A (Multi-class)'})


    wandb.finish() # End the Weights & Biases run

# ==================================
# Command-line Arguments
# ==================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size for training and testing.")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4, help="Initial learning rate for the optimizer.")
    parser.add_argument("-m", "--momentum", type=float, default=0.0, help="Momentum for optimizer (not used by Adam default).")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0, help="Weight decay (L2 penalty) for optimizer.")
    parser.add_argument("--depth", type=int, default=18, help="Model depth (placeholder, not used in this CustomCNN).")
    parser.add_argument("--width", type=int, default=64, help="Model width (placeholder, not used in this CustomCNN).")
    parser.add_argument("--groups", type=int, default=1, help="Number of groups for convolutions (placeholder).")
    parser.add_argument("--chkpt_path", type=str, default="./checkpoints", help="Path to save model checkpoints.")
    args = parser.parse_args()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.chkpt_path, exist_ok=True)

    main(args)