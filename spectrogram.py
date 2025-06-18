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
from sklearn.preprocessing import label_binarize # Import for multi-class ROC

# constant which i am using in this
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MAX_TIME_STEPS = 400

class CustomCNN(nn.Module):
    def __init__(self, num_classes): # num_classes is now a mandatory argument
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Output after first block: (32, 64, 200) assuming input (1, 128, 400)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Output after second block: (64, 32, 100)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Output after third block: (128, 16, 50)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Output after fourth block: (256, 8, 25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 25, 64), # From (256, 8, 25)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes) # Dynamically set output classes
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# here i am loading the dataset
class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.label_map = {}
        self.classes = [] # Initialize classes list

        # Iterate through subdirectories to find classes
        class_folders = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        if not class_folders:
            print(f"Warning: No class subfolders found in {root_dir}. Please ensure your dataset is structured as root_dir/class_name/audio_files.wav")
        
        for idx, class_name in enumerate(class_folders):
            class_path = os.path.join(root_dir, class_name)
            self.label_map[class_name] = idx
            self.classes.append(class_name) # Populate self.classes in order of idx
            for file_name in os.listdir(class_path):
                if file_name.endswith('.wav'):
                    self.samples.append((os.path.join(class_path, file_name), idx))
        
        self.num_classes = len(self.classes) # Store the detected number of classes

    def __len__(self):
        return len(self.samples)

    # convertion of audio in spectrogram
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        try:
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            # resizing the image to pass in the model
            mel_spec = cv2.resize(mel_spec, (MAX_TIME_STEPS, N_MELS)) # (400, 128)

            # Normalization process
            mel_spec = (mel_spec - np.mean(mel_spec)) / np.std(mel_spec)

            # converting in tensor to easy passing in model
            mel_spec = torch.from_numpy(mel_spec).float().unsqueeze(0) # Shape: (1, 128, 400)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            mel_spec = torch.zeros((1, N_MELS, MAX_TIME_STEPS), dtype=torch.float32)
            label = -1 # Mark as invalid
        return mel_spec, label

def main(args):
    print("code started")
    run = wandb.init(project="RAVDESS_spectrogram", config=vars(args))
    config = wandb.config

    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    train_path = os.path.join(script_dir, 'datasets/train')
    test_path = os.path.join(script_dir, 'datasets/test')

    print("data loading initiated")

    train_data = AudioDataset(train_path)
    test_data = AudioDataset(test_path) # Also load test data to get its classes (should match train)

    # Dynamically determine the number of classes from the dataset
    num_classes = train_data.num_classes
    if num_classes == 0:
        raise ValueError(f"No classes found in the training dataset at {train_path}. Please check your folder structure.")

    # Pass class names and number of classes to wandb config
    run.config['class_names'] = train_data.classes
    run.config['num_classes'] = num_classes

    print(f"Detected {num_classes} classes: {train_data.classes}")
    print("data loaded")

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    model = CustomCNN(num_classes=num_classes) # Pass the detected number of classes to the model

    # The input_size for torchinfo.summary should reflect the actual batch dimension
    # and the channel, height, width of your spectrogram.
    summary = torchinfo.summary(model, input_size=(config.batch_size, 1, N_MELS, MAX_TIME_STEPS))
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
        processed_train_batches = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Filter out invalid samples
            valid_indices = (labels != -1)
            if not valid_indices.any():
                print(f"Skipping training batch {batch_idx} due to all invalid samples.")
                continue

            inputs = inputs[valid_indices].to(device)
            labels = labels[valid_indices].to(device)

            if inputs.size(0) == 0: # Check if any valid inputs remain
                continue

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            processed_train_batches += 1

        avg_train_loss = total_train_loss / processed_train_batches if processed_train_batches > 0 else 0.0


        model.eval()
        correct, total = 0, 0
        total_test_loss = 0.0
        all_labels = []
        all_predictions = []
        all_probabilities = [] # Store probabilities for all classes
        processed_test_batches = 0

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

                probabilities = torch.softmax(outputs, dim=1) # Softmax for all classes
                _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy()) # Store full probability distribution
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                processed_test_batches += 1

        # Only calculate avg_test_loss if test_loader has valid data
        if processed_test_batches > 0:
            avg_test_loss = total_test_loss / processed_test_batches
        else:
            avg_test_loss = 0.0
            print("Warning: No valid samples processed in test set for current epoch. Test loss set to 0.0.")

        accuracy = correct / total if total > 0 else 0.0 # Handle division by zero

        scheduler.step(avg_test_loss)

        # Early Stopping check
        if avg_test_loss < best_val_loss:
            best_val_loss = avg_test_loss
            patience_counter = 0
            # Save the best model
            os.makedirs(config.chkpt_path, exist_ok=True) # Ensure checkpoint directory exists
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
    y_scores_array = np.array(all_probabilities) # Use this for multi-class ROC

    # Ensure there are samples to evaluate
    if len(y_true) == 0:
        print("No valid samples for final evaluation. Skipping metrics and plots.")
        wandb.finish()
        return

    # Get class names for classification report and confusion matrix
    # It's important to use the classes from the test_data, or ensure train_data and test_data have identical class sets.
    # For robustness, we'll use test_data.classes, but also verify its consistency with num_classes.
    if len(test_data.classes) != num_classes:
        print("Warning: Number of classes in test data does not match training data. Using training data classes for consistency.")
    target_names = train_data.classes # Use train_data classes as the source of truth for the model's output

    print("\n─────── Classification Report ───────")
    report = classification_report(y_true, y_pred_classes, target_names=target_names, output_dict=True, zero_division=0)
    print(json.dumps(report, indent=4)) # Print pretty JSON

    # Log individual metrics from the classification report
    wandb_log_metrics = {}
    for class_name, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                if metric_name in ['precision', 'recall', 'f1-score']:
                    wandb_log_metrics[f'classification_report/{class_name}_{metric_name}'] = value
        elif class_name in ['accuracy', 'macro avg', 'weighted avg']:
            if isinstance(metrics, dict): # For macro/weighted avg which are dicts
                for metric_name, value in metrics.items():
                    if metric_name in ['precision', 'recall', 'f1-score', 'support']:
                        wandb_log_metrics[f'classification_report/{class_name.replace(" ", "_")}_{metric_name}'] = value
            else: # For accuracy which is a direct value
                if class_name == 'accuracy':
                    wandb_log_metrics[f'classification_report/accuracy'] = metrics

    wandb.log(wandb_log_metrics)
    wandb.log({"classification_report_text": wandb.Html(classification_report(y_true, y_pred_classes, target_names=target_names, zero_division=0))})


    print("\n─────── Confusion Matrix ───────")
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(max(8, num_classes), max(7, num_classes-1))) # Adjust figure size dynamically
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.show()
    plt.close()

    print("\n─────── ROC Curve (One-vs-Rest) ───────")
    plt.figure(figsize=(10, 8))
    # Check if y_true and y_scores_array are suitable for multi-class ROC
    if num_classes == 2:
        # Binary ROC curve
        # Ensure y_scores_array has at least two columns for binary classification
        if y_scores_array.shape[1] < 2:
            print("Warning: y_scores_array has less than 2 columns for binary classification ROC. Skipping ROC curve.")
            wandb.finish()
            return

        fpr, tpr, _ = roc_curve(y_true, y_scores_array[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        wandb.log({'final_auc_score': roc_auc}) # Log only for binary case
        print(f"✅ Model AUC Score: {roc_auc:.4f}")
    else:
        # Multi-class One-vs-Rest ROC curves
        # Convert y_true to one-hot encoding for roc_curve
        y_true_binarized = label_binarize(y_true, classes=range(num_classes))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(num_classes):
            # Only compute if the class has at least one positive sample in y_true_binarized
            # and corresponding scores are available in y_scores_array
            if np.sum(y_true_binarized[:, i]) > 0 and y_scores_array.shape[1] > i:
                fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_scores_array[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], lw=2,
                         label=f'ROC curve of class {target_names[i]} (area = {roc_auc[i]:0.2f})')
            else:
                print(f"Warning: Class {target_names[i]} (index {i}) not present in true labels or scores for ROC calculation.")
                roc_auc[i] = 0.0 # Assign 0 AUC if class not present


        # Compute micro-average ROC curve and AUC
        # Ensure y_true_binarized and y_scores_array are not empty or malformed
        if y_true_binarized.size > 0 and y_scores_array.size > 0:
            fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_scores_array.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            plt.plot(fpr["micro"], tpr["micro"],
                     label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
                     color='deeppink', linestyle=':', linewidth=4)
            wandb.log({'final_micro_auc_score': roc_auc["micro"]})
            print(f"✅ Model Micro-average AUC Score: {roc_auc['micro']:.4f}")
        else:
            print("Warning: Cannot compute micro-average ROC due to empty or malformed true labels/scores.")


        # Compute macro-average ROC curve and AUC
        valid_fpr_tpr_indices = [i for i in range(num_classes) if i in fpr and fpr[i].size > 0]
        if valid_fpr_tpr_indices:
            all_fpr = np.unique(np.concatenate([fpr[i] for i in valid_fpr_tpr_indices]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in valid_fpr_tpr_indices:
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

            mean_tpr /= len(valid_fpr_tpr_indices) # Divide by the number of valid classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            plt.plot(fpr["macro"], tpr["macro"],
                     label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
                     color='navy', linestyle=':', linewidth=4)
            wandb.log({'final_macro_auc_score': roc_auc["macro"]})
            print(f"✅ Model Macro-average AUC Score: {roc_auc['macro']:.4f}")
        else:
            print("Warning: Cannot compute macro-average ROC due to no valid class ROC curves.")


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


    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4)
    parser.add_argument("-m", "--momentum", type=float, default=0.0) # Not used in Adam, but kept for consistency
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("--depth", type=int, default=18) # Not used in CustomCNN, but kept for consistency
    parser.add_argument("--width", type=int, default=64) # Not used in CustomCNN, but kept for consistency
    parser.add_argument("--groups", type=int, default=1) # Not used in CustomCNN, but kept for consistency
    parser.add_argument("--chkpt_path", type=str, default="./checkpoints") # Changed default for checkpoints
    args = parser.parse_args()
    main(args)