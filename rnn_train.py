import numpy as np
import h5py
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Masking
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_data():
    """Load and process data from Postures.csv"""
    try:
        print("Loading Postures.csv...")
        df = pd.read_csv('Postures.csv')
        print(f"Dataset shape: {df.shape}")
        
        # Show data info
        print("\nDataset Info:")
        print(df.info())
        
        # Replace '?' with 0 and convert to float
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
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def pad_2d_sequences(X, max_len=None):
    """Pad sequences to the same length"""
    if max_len is None:
        max_len = max([x.shape[1] for x in X])
    
    X_squeezed = [np.squeeze(seq, axis=0) for seq in X]
    X_padded = np.zeros((len(X_squeezed), max_len, X_squeezed[0].shape[1]))
    
    for i, seq in enumerate(X_squeezed):
        X_padded[i, :seq.shape[0], :] = seq
    
    return X_padded

def create_model(input_shape, num_classes):
    """Create RNN model with LSTM layers"""
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
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
    
    # No need for padding since data is already uniform
    print(f"Data shape: {X.shape}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_encoded = to_categorical(y_encoded)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, 
        test_size=0.4, 
        random_state=13
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.5, 
        random_state=42
    )
    
    # Create and compile model
    input_shape = (X.shape[1], X.shape[2])
    num_classes = y_encoded.shape[1]
    model = create_model(input_shape, num_classes)
    
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
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=1)
    print(f'\nValidation Accuracy: {val_accuracy:.4f}')
    print(f'Validation Loss: {val_loss:.4f}')
    
    # Save model
    model.save('models/dynamic_rnn.h5')
    
    # Generate predictions and plot results
    y_pred = model.predict(X_test)
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred, label_encoder)

if __name__ == "__main__":
    main()
