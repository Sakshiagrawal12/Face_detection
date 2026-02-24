# scripts/train_model.py
"""
Script 03: MobileNetV2 Training
Run this after preprocessing to train the model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import json
import logging
from config import (
    IMG_SIZE, NUM_CLASSES, BATCH_SIZE, 
    INITIAL_LEARNING_RATE, FINE_TUNE_LEARNING_RATE,
    EPOCHS, FINE_TUNE_EPOCHS, BEST_MODEL_PATH, FINAL_MODEL_PATH,
    OUTPUT_PATH, MODELS_PATH, CATEGORIES
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_mobilenetv2_model():
    """Create MobileNetV2 based model"""
    print("\n" + "="*60)
    print("BUILDING MOBILENETV2 MODEL")
    print("="*60)
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu', name='dense_128')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    x = Dense(64, activation='relu', name='dense_64')(x)
    x = Dropout(0.3, name='dropout_2')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax', name='output')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Print model summary
    model.summary()
    
    # Count trainable parameters
    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    print(f"\n📊 Model Statistics:")
    print(f"  • Total parameters: {trainable_params + non_trainable_params:,}")
    print(f"  • Trainable parameters: {trainable_params:,}")
    print(f"  • Non-trainable parameters: {non_trainable_params:,}")
    
    return model, base_model

def plot_training_history(history, save_path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_after_training(model, X_val, y_val):
    """Quick evaluation after training"""
    print("\n" + "="*60)
    print("📊 POST-TRAINING EVALUATION")
    print("="*60)
    
    # Predict on validation set
    y_pred_prob = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"\n📈 Validation Metrics:")
    print(f"  • Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  • Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  • Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  • F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }

def train_mobilenetv2():
    """Train MobileNetV2 model with comprehensive logging"""
    
    print("\n" + "="*60)
    print("🚀 MOBILENETV2 TRAINING PIPELINE")
    print("="*60)
    
    # Create directories
    log_dir = os.path.join(OUTPUT_PATH, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Check if preprocessed data exists
    required_files = ['X_train.npy', 'X_val.npy', 'y_train.npy', 'y_val.npy']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(OUTPUT_PATH, f))]
    
    if missing_files:
        logging.error(f"Missing preprocessed files: {missing_files}")
        print("Please run preprocessing first: python main.py --mode preprocess")
        return False
    
    # Load preprocessed data
    print("\n📂 Loading preprocessed data...")
    X_train = np.load(os.path.join(OUTPUT_PATH, 'X_train.npy'))
    X_val = np.load(os.path.join(OUTPUT_PATH, 'X_val.npy'))
    y_train = np.load(os.path.join(OUTPUT_PATH, 'y_train.npy'))
    y_val = np.load(os.path.join(OUTPUT_PATH, 'y_val.npy'))
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  • Training samples: {len(X_train)}")
    print(f"  • Validation samples: {len(X_val)}")
    print(f"  • Image shape: {X_train[0].shape}")
    print(f"  • Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\n⚖️  Class weights: {class_weight_dict}")
    
    # Enhanced data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Create model
    model, base_model = create_mobilenetv2_model()
    
    # Save model architecture visualization
    tf.keras.utils.plot_model(
        model, 
        to_file=os.path.join(OUTPUT_PATH, 'model_architecture.png'),
        show_shapes=True,
        show_layer_names=True
    )
    
    # Phase 1: Train top layers
    print("\n" + "="*60)
    print("📈 PHASE 1: TRAINING TOP LAYERS")
    print("="*60)
    
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks for phase 1
    callbacks_phase1 = [
        ModelCheckpoint(
            os.path.join(MODELS_PATH, 'phase1_best.h5'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(log_dir, 'phase1'),
            histogram_freq=1,
            write_graph=True
        ),
        CSVLogger(os.path.join(log_dir, 'phase1_training_log.csv'))
    ]
    
    history1 = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks_phase1,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Phase 2: Fine-tuning
    print("\n" + "="*60)
    print("🔧 PHASE 2: FINE-TUNING")
    print("="*60)
    
    # Unfreeze some layers for fine-tuning
    base_model.trainable = True
    
    # Freeze early layers, fine-tune later layers
    # Typically freeze first 100 layers of MobileNetV2
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Count trainable layers
    trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"\n📊 Fine-tuning Configuration:")
    print(f"  • Total base layers: {len(base_model.layers)}")
    print(f"  • Trainable layers: {trainable_layers}")
    print(f"  • Frozen layers: {len(base_model.layers) - trainable_layers}")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks for phase 2
    callbacks_phase2 = [
        ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(MODELS_PATH, 'best_by_precision.h5'),
            monitor='val_precision' if hasattr(tf.keras.metrics, 'Precision') else 'val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(log_dir, 'phase2'),
            histogram_freq=1,
            write_graph=True
        ),
        CSVLogger(os.path.join(log_dir, 'phase2_training_log.csv'))
    ]
    
    # Add precision metric if available
    if hasattr(tf.keras.metrics, 'Precision'):
        metrics = ['accuracy', tf.keras.metrics.Precision(name='precision')]
        model.compile(
            optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=metrics
        )
    
    history2 = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        validation_data=(X_val, y_val),
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks_phase2,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Combine histories
    history = {}
    for key in history1.history.keys():
        history[key] = history1.history[key] + history2.history[key]
    
    # Plot training history
    plot_training_history(
        type('obj', (object,), {'history': history})(),
        os.path.join(OUTPUT_PATH, 'training_history.png')
    )
    
    # Quick evaluation after training
    val_metrics = evaluate_after_training(model, X_val, y_val)
    
    # Save final model
    model.save(FINAL_MODEL_PATH)
    print(f"\n💾 Final model saved to: {FINAL_MODEL_PATH}")
    
    # Save training metadata
    training_info = {
        'model': 'MobileNetV2',
        'input_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'phase1_epochs': len(history1.history['loss']),
        'phase2_epochs': len(history2.history['loss']),
        'total_epochs': len(history['loss']),
        'final_validation_metrics': val_metrics,
        'class_weights': class_weight_dict,
        'training_completed': datetime.datetime.now().isoformat()
    }
    
    with open(os.path.join(OUTPUT_PATH, 'training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=4)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📁 Training logs saved to: {log_dir}")
    print(f"📁 Best model saved to: {BEST_MODEL_PATH}")
    print(f"📁 Final model saved to: {FINAL_MODEL_PATH}")
    print(f"\n📊 Final Validation Metrics:")
    print(f"  • Accuracy:  {val_metrics['accuracy']*100:.2f}%")
    print(f"  • Precision: {val_metrics['precision']*100:.2f}%")
    print(f"  • Recall:    {val_metrics['recall']*100:.2f}%")
    print(f"  • F1-Score:  {val_metrics['f1']*100:.2f}%")
    print("\nNext step: Run 04_evaluate_model.py for detailed evaluation")
    
    return True

if __name__ == "__main__":
    # Check TensorFlow and GPU
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU Available: {len(gpus)} GPU(s)")
        for gpu in gpus:
            print(f"  • {gpu.name}")
    else:
        print("⚠️  No GPU found. Training will be on CPU (slower)")
    
    success = train_mobilenetv2()
    sys.exit(0 if success else 1)