import numpy as np
import pandas as pd
import h5py
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns  # For improved confusion matrix visualization
from sklearn.metrics import confusion_matrix

def load_data():
    """Load and process data from Postures.csv"""
    try:
        # Load the CSV file
        print("Loading Postures.csv...")
        df = pd.read_csv('Postures.csv')
        print(f"Dataset shape: {df.shape}")
        
        # Show data info
        print("\nDataset Info:")
        print(df.info())
        
        # Replace '?' with 0 instead of NaN for all feature columns
        for col in df.columns[2:]:  # Skip Class and User columns
            df[col] = df[col].replace('?', '0')
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Show class distribution
        print("\nClass distribution:")
        print(df['Class'].value_counts().sort_index())
        
        # Extract features and labels
        X = df.iloc[:, 2:].values.astype(float)  # All columns except Class and User
        y = df['Class'].values.astype(int)
        
        # Reshape X for CNN (samples, timesteps, features)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Unique classes: {np.unique(y)}")
        
        return X, y
        
    except FileNotFoundError:
        print("Error: Postures.csv not found!")
        raise
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def process_h5_files(directory):
    """Load and process H5 files from directory"""
    X = []
    y = []
    for file in os.listdir(directory):
        if file.endswith(".h5"):
            filepath = os.path.join(directory, file)
            with h5py.File(filepath, "r") as f:
                print(f"Processing file: {file}")
                for dataset_name in f.keys():
                    data = f[dataset_name][:]
                    label = file.split(".")[0]  # Extract label from filename
                    X.append(data)
                    y.append(label)
    return X, np.array(y)

def create_model(input_shape, num_classes):
    """Create CNN model"""
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling1D(2),
        Dropout(0.2),
        
        Conv1D(128, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Dropout(0.2),
        
        Conv1D(256, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Dropout(0.3),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    return model

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 4))

    # Accuracy plot
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(history.history['accuracy'], label='Training', color='skyblue')
    ax1.plot(history.history['val_accuracy'], label='Validation', color='lightgreen')
    ax1.set_title('Model Accuracy', color='white')
    ax1.set_xlabel('Epoch', color='white')
    ax1.set_ylabel('Accuracy', color='white')
    ax1.legend(loc='lower right', facecolor='black', labelcolor='white')
    ax1.tick_params(colors='white', which='both')
    plt.setp(ax1.get_xticklabels(), color='white')
    plt.setp(ax1.get_yticklabels(), color='white')
    ax1.set_facecolor('black')

    # Loss plot
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(history.history['loss'], label='Training', color='salmon')
    ax2.plot(history.history['val_loss'], label='Validation', color='gold')
    ax2.set_title('Model Loss', color='white')
    ax2.set_xlabel('Epoch', color='white')
    ax2.set_ylabel('Loss', color='white')
    ax2.legend(loc='upper right', facecolor='black', labelcolor='white')
    ax2.tick_params(colors='white', which='both')
    plt.setp(ax2.get_xticklabels(), color='white')
    plt.setp(ax2.get_yticklabels(), color='white')
    ax2.set_facecolor('black')

    plt.gcf().set_facecolor('black')
    plt.tight_layout()
    plt.savefig('results/training_history.png', facecolor='black', bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_test, y_pred, label_encoder):
    """Plot confusion matrix"""
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    
    plt.figure(figsize=(10, 8))
    plt.gcf().set_facecolor('black')
    
    ax = sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='cubehelix',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    
    ax.set_facecolor('black')
    plt.title('Confusion Matrix', color='white')
    plt.xlabel('Predicted Labels', color='white')
    plt.ylabel('True Labels', color='white')
    
    ax.tick_params(colors='white')
    plt.setp(ax.get_xticklabels(), color="white")
    plt.setp(ax.get_yticklabels(), color="white")
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', facecolor='black', bbox_inches='tight')
    plt.show()

def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load and process data
    print("Loading data from Postures.csv...")
    X, y = load_data()
    
    # Print class distribution before filtering
    print("\nOriginal class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for class_id, count in zip(unique, counts):
        print(f"Class {class_id}: {count} samples")
    
    # Remove classes with too few samples (less than 10 samples)
    min_samples = 10
    valid_classes = []
    for class_id in unique:
        if np.sum(y == class_id) >= min_samples:
            valid_classes.append(class_id)
    
    # Filter data to keep only valid classes
    mask = np.isin(y, valid_classes)
    X = X[mask]
    y = y[mask]
    
    print(f"\nKeeping classes with at least {min_samples} samples:")
    unique, counts = np.unique(y, return_counts=True)
    for class_id, count in zip(unique, counts):
        print(f"Class {class_id}: {count} samples")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_encoded = to_categorical(y_encoded)
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2,
        random_state=42,
        stratify=np.argmax(y_encoded, axis=1)
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=np.argmax(y_train, axis=1)
    )
    
    # Print split sizes
    print("\nData split sizes:")
    print(f"Training:   {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")
    print(f"Test:      {X_test.shape[0]} samples")
    
    # Create and compile model
    input_shape = X_train.shape[1:]
    num_classes = y_encoded.shape[1]
    print(f"\nInput shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    
    model = create_model(input_shape, num_classes)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Model summary
    model.summary()
    
    # Train model
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'\nTest Accuracy: {test_acc:.4f}')
    
    # Save model
    model.save('models/posture_cnn.h5')
    
    # Generate predictions and plot results
    y_pred = model.predict(X_test)
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred, label_encoder)

if __name__ == "__main__":
    main()