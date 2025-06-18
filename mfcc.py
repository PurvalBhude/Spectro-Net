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

# constant which i am using in this
SAMPLE_RATE = 22050
N_MFCC = 40 
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MAX_TIME_STEPS = 400

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            # Input channel is still 1, but height is N_MFCC instead of N_MELS (128)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Output after first block: (32, N_MFCC/2, 200)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Output after second block: (64, N_MFCC/4, 100)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Output after third block: (128, N_MFCC/8, 50)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Output after fourth block: (256, N_MFCC/16, 25)
        )
        # Calculate the size for the linear layer dynamically based on N_MFCC
        # After 4 stride 2 convolutions, the height dimension becomes N_MFCC / (2^4) = N_MFCC / 16
        # Assuming N_MFCC = 40, then 40 / 16 = 2.5. This means you will need to adjust the height
        # of your output after the convolutions to be an integer.
        # A common approach is to use AdaptiveMaxPool2d or carefully adjust strides/paddings
        # to ensure integer dimensions.
        # Let's re-evaluate the output dimensions for a typical N_MFCC like 40:
        # Input: (1, 40, 400)
        # Block 1 (stride 2): (32, 20, 200)
        # Block 2 (stride 2): (64, 10, 100)
        # Block 3 (stride 2): (128, 5, 50)
        # Block 4 (stride 2): (256, 2, 25) -- This works well with N_MFCC = 40!

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (N_MFCC // 16) * 25, 64), # Dynamic calculation: 256 * (40/16) * 25 = 256 * 2 * 25 = 12800 for N_MFCC=40
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 7) # as there are two classes
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# here i am loading the dataset
class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.label_map = {}

        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.label_map[class_name] = idx
                for file_name in os.listdir(class_path):
                    if file_name.endswith('.wav'):
                        self.samples.append((os.path.join(class_path, file_name), idx))
        self.classes = sorted(self.label_map.keys()) # Store class names

    def __len__(self):
        return len(self.samples)

    # conversion of audio to MFCC
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        try:
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            # Calculate MFCCs instead of Mel spectrogram
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)

            # Resize MFCCs
            # Note: OpenCV's resize expects (width, height), so (MAX_TIME_STEPS, N_MFCC)
            mfcc = cv2.resize(mfcc, (MAX_TIME_STEPS, N_MFCC)) # (400, N_MFCC)

            # Normalization process
            mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

            # converting in tensor to easy passing in model
            mfcc = torch.from_numpy(mfcc).float().unsqueeze(0) # Shape: (1, N_MFCC, 400)
            feature_data = mfcc

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            feature_data = torch.zeros((1, N_MFCC, MAX_TIME_STEPS), dtype=torch.float32)
            label = -1
        return feature_data, label

def main(args):
    print("code started")
    run = wandb.init(project="iiit_hyerabad_mfcc", config=vars(args)) # Changed project name
    config = wandb.config

    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    train_path = os.path.join(script_dir, 'datasets/train')
    test_path = os.path.join(script_dir, 'datasets/test')

    print("data loading initiated")

    train_data = AudioDataset(train_path)
    test_data = AudioDataset(test_path)

    # Pass class names to wandb config
    if hasattr(train_data, 'classes'):
        run.config['class_names'] = train_data.classes
    else:
        print("Warning: Class names not found in dataset. Ensure AudioDataset sets 'self.classes'.")


    print("data loaded")

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    model = CustomCNN()

    # The input_size for torchinfo.summary should reflect the actual batch dimension
    # and the channel, height, width of your MFCCs.
    summary = torchinfo.summary(model, input_size=(config.batch_size, 1, N_MFCC, MAX_TIME_STEPS)) # Changed N_MELS to N_MFCC
    run.config['total_params'] = summary.total_params
    run.config['mult_adds'] = summary.total_mult_adds

    print("model begin training")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    #training optimizer and activation functions
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    # early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 5

    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if (labels == -1).any():
                print(f"Skipping batch {batch_idx} due to error in data loading.")
                continue

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        correct, total = 0, 0
        total_test_loss = 0.0
        all_labels = []
        all_predictions = []
        all_probabilities = []

        #test data running
        with torch.no_grad():
            for inputs, labels in test_loader:
                # Filter out samples with label -1
                valid_indices = (labels != -1)
                if not valid_indices.any():
                    continue # Skip if no valid samples in batch

                inputs = inputs[valid_indices].to(device)
                labels = labels[valid_indices].to(device)

                if inputs.size(0) == 0: # Check if any valid inputs remain
                    continue

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()

                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Only calculate avg_test_loss if test_loader was not empty
        if len(test_loader) > 0:
            avg_test_loss = total_test_loss / len(test_loader)
        else:
            avg_test_loss = 0.0 # Or handle as an error/warning
            print("Warning: Test loader is empty, test loss set to 0.0.")

        accuracy = correct / total if total > 0 else 0.0 # Handle division by zero

        scheduler.step(avg_test_loss)

        # Early Stopping check
        if avg_test_loss < best_val_loss:
            best_val_loss = avg_test_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(config.chkpt_path, run.id + '_best_model.pt'))
            print(f"[{epoch+1}/{config.epochs}] Validation loss improved. Saving model.")
        else:
            patience_counter += 1
            print(f"[{epoch+1}/{config.epochs}] Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break # Exit training loop

        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'test_accuracy': accuracy,
            'lr': optimizer.param_groups[0]['lr'],
        })

        print(f"[{epoch+1}/{config.epochs}] Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Final evaluation after training
    model_save_path = os.path.join(config.chkpt_path, run.id + '_best_model.pt')
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        print("Loaded best model for final evaluation.")
    else:
        print("Best model checkpoint not found. Using the last trained model for final evaluation.")

    y_true = np.array(all_labels)
    y_pred_classes = np.array(all_predictions)
    y_scores = np.array(all_probabilities)

    # Ensure there are samples to evaluate
    if len(y_true) == 0:
        print("No valid samples for final evaluation. Skipping metrics and plots.")
        wandb.finish()
        return

    # Get class names for classification report and confusion matrix
    target_names = test_data.classes if hasattr(test_data, 'classes') and len(test_data.classes) == 2 else [str(i) for i in range(2)]

    print("\n─────── Classification Report ───────")
    report = classification_report(y_true, y_pred_classes, target_names=target_names, output_dict=True)
    print(json.dumps(report, indent=4)) # Print pretty JSON

    # Log individual metrics from the classification report
    wandb_log_metrics = {}
    for class_name, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                if metric_name in ['precision', 'recall', 'f1-score', 'support']:
                    wandb_log_metrics[f'classification_report/{class_name}_{metric_name}'] = value
        elif class_name in ['accuracy', 'macro avg', 'weighted avg']:
            # For accuracy and overall averages
            for metric_name, value in metrics.items():
                 if metric_name in ['precision', 'recall', 'f1-score', 'support']:
                    wandb_log_metrics[f'classification_report/{class_name.replace(" ", "_")}_{metric_name}'] = value
            if class_name == 'accuracy':
                wandb_log_metrics[f'classification_report/accuracy'] = metrics # Accuracy is directly a value, not a dict
    
    wandb.log(wandb_log_metrics)
    wandb.log({"classification_report_text": wandb.Html(classification_report(y_true, y_pred_classes, target_names=target_names))})


    print("\n─────── Confusion Matrix ───────")
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.show()
    plt.close()

    print("\n─────── ROC Curve ───────")
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
    wandb.log({"roc_curve": wandb.Image(plt)})
    plt.show()
    plt.close()

    print(f"✅ Model AUC Score: {roc_auc:.4f}")
    wandb.log({'final_auc_score': roc_auc})

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4)
    parser.add_argument("-m", "--momentum", type=float, default=0.0)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--width", type=int, default=64) 
    parser.add_argument("--groups", type=int, default=1) 
    parser.add_argument("--chkpt_path", type=str, default="./")
    args = parser.parse_args()
    main(args)