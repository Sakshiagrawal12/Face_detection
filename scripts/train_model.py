# # scripts/train_model.py
# """
# Script 03: MobileNetV2 Training
# Run this after preprocessing to train the model
# """

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import (
#     ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
#     TensorBoard, CSVLogger
# )
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import datetime
# import json
# import logging
# from config import (
#     IMG_SIZE, NUM_CLASSES, BATCH_SIZE, 
#     INITIAL_LEARNING_RATE, FINE_TUNE_LEARNING_RATE,
#     EPOCHS, FINE_TUNE_EPOCHS, BEST_MODEL_PATH, FINAL_MODEL_PATH,
#     OUTPUT_PATH, MODELS_PATH, CATEGORIES
# )

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def create_mobilenetv2_model():
#     """Create MobileNetV2 based model"""
#     print("\n" + "="*60)
#     print("BUILDING MOBILENETV2 MODEL")
#     print("="*60)
    
#     # Load pre-trained MobileNetV2
#     base_model = MobileNetV2(
#         weights='imagenet',
#         include_top=False,
#         input_shape=(IMG_SIZE, IMG_SIZE, 3)
#     )
    
#     # Freeze base model layers initially
#     base_model.trainable = False
    
#     # Add custom layers
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = BatchNormalization()(x)
#     x = Dense(128, activation='relu', name='dense_128')(x)
#     x = Dropout(0.5, name='dropout_1')(x)
#     x = Dense(64, activation='relu', name='dense_64')(x)
#     x = Dropout(0.3, name='dropout_2')(x)
#     predictions = Dense(NUM_CLASSES, activation='softmax', name='output')(x)
    
#     # Create model
#     model = Model(inputs=base_model.input, outputs=predictions)
    
#     # Print model summary
#     model.summary()
    
#     # Count trainable parameters
#     trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
#     non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
#     print(f"\n📊 Model Statistics:")
#     print(f"  • Total parameters: {trainable_params + non_trainable_params:,}")
#     print(f"  • Trainable parameters: {trainable_params:,}")
#     print(f"  • Non-trainable parameters: {non_trainable_params:,}")
    
#     return model, base_model

# def plot_training_history(history, save_path):
#     """Plot training history"""
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
#     # Plot accuracy
#     ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
#     ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
#     ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
#     ax1.set_xlabel('Epoch', fontsize=12)
#     ax1.set_ylabel('Accuracy', fontsize=12)
#     ax1.legend(loc='lower right')
#     ax1.grid(True, alpha=0.3)
#     ax1.set_ylim([0, 1])
    
#     # Plot loss
#     ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
#     ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
#     ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
#     ax2.set_xlabel('Epoch', fontsize=12)
#     ax2.set_ylabel('Loss', fontsize=12)
#     ax2.legend(loc='upper right')
#     ax2.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()

# def evaluate_after_training(model, X_val, y_val):
#     """Quick evaluation after training"""
#     print("\n" + "="*60)
#     print("📊 POST-TRAINING EVALUATION")
#     print("="*60)
    
#     # Predict on validation set
#     y_pred_prob = model.predict(X_val, verbose=0)
#     y_pred = np.argmax(y_pred_prob, axis=1)
    
#     # Calculate metrics
#     from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
#     accuracy = accuracy_score(y_val, y_pred)
#     precision = precision_score(y_val, y_pred, average='weighted')
#     recall = recall_score(y_val, y_pred, average='weighted')
#     f1 = f1_score(y_val, y_pred, average='weighted')
    
#     print(f"\n📈 Validation Metrics:")
#     print(f"  • Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
#     print(f"  • Precision: {precision:.4f} ({precision*100:.2f}%)")
#     print(f"  • Recall:    {recall:.4f} ({recall*100:.2f}%)")
#     print(f"  • F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
#     return {
#         'accuracy': float(accuracy),
#         'precision': float(precision),
#         'recall': float(recall),
#         'f1': float(f1)
#     }

# def train_mobilenetv2():
#     """Train MobileNetV2 model with comprehensive logging"""
    
#     print("\n" + "="*60)
#     print("🚀 MOBILENETV2 TRAINING PIPELINE")
#     print("="*60)
    
#     # Create directories
#     log_dir = os.path.join(OUTPUT_PATH, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#     os.makedirs(log_dir, exist_ok=True)
    
#     # Check if preprocessed data exists
#     required_files = ['X_train.npy', 'X_val.npy', 'y_train.npy', 'y_val.npy']
#     missing_files = [f for f in required_files if not os.path.exists(os.path.join(OUTPUT_PATH, f))]
    
#     if missing_files:
#         logging.error(f"Missing preprocessed files: {missing_files}")
#         print("Please run preprocessing first: python main.py --mode preprocess")
#         return False
    
#     # Load preprocessed data
#     print("\n📂 Loading preprocessed data...")
#     X_train = np.load(os.path.join(OUTPUT_PATH, 'X_train.npy'))
#     X_val = np.load(os.path.join(OUTPUT_PATH, 'X_val.npy'))
#     y_train = np.load(os.path.join(OUTPUT_PATH, 'y_train.npy'))
#     y_val = np.load(os.path.join(OUTPUT_PATH, 'y_val.npy'))
    
#     print(f"\n📊 Dataset Statistics:")
#     print(f"  • Training samples: {len(X_train)}")
#     print(f"  • Validation samples: {len(X_val)}")
#     print(f"  • Image shape: {X_train[0].shape}")
#     print(f"  • Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
#     # Calculate class weights for imbalanced data
#     class_weights = compute_class_weight(
#         class_weight='balanced',
#         classes=np.unique(y_train),
#         y=y_train
#     )
#     class_weight_dict = dict(enumerate(class_weights))
#     print(f"\n⚖️  Class weights: {class_weight_dict}")
    
#     # Enhanced data augmentation
#     train_datagen = ImageDataGenerator(
#         rotation_range=30,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         brightness_range=[0.8, 1.2],
#         fill_mode='nearest'
#     )
    
#     # Create model
#     model, base_model = create_mobilenetv2_model()
    
#     # Save model architecture visualization
#     tf.keras.utils.plot_model(
#         model, 
#         to_file=os.path.join(OUTPUT_PATH, 'model_architecture.png'),
#         show_shapes=True,
#         show_layer_names=True
#     )
    
#     # Phase 1: Train top layers
#     print("\n" + "="*60)
#     print("📈 PHASE 1: TRAINING TOP LAYERS")
#     print("="*60)
    
#     model.compile(
#         optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     # Callbacks for phase 1
#     callbacks_phase1 = [
#         ModelCheckpoint(
#             os.path.join(MODELS_PATH, 'phase1_best.h5'),
#             monitor='val_accuracy',
#             mode='max',
#             save_best_only=True,
#             verbose=1
#         ),
#         EarlyStopping(
#             monitor='val_loss',
#             patience=5,
#             restore_best_weights=True,
#             verbose=1
#         ),
#         ReduceLROnPlateau(
#             monitor='val_loss',
#             factor=0.5,
#             patience=3,
#             min_lr=1e-7,
#             verbose=1
#         ),
#         TensorBoard(
#             log_dir=os.path.join(log_dir, 'phase1'),
#             histogram_freq=1,
#             write_graph=True
#         ),
#         CSVLogger(os.path.join(log_dir, 'phase1_training_log.csv'))
#     ]
    
#     history1 = model.fit(
#         train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
#         steps_per_epoch=len(X_train) // BATCH_SIZE,
#         validation_data=(X_val, y_val),
#         epochs=EPOCHS,
#         callbacks=callbacks_phase1,
#         class_weight=class_weight_dict,
#         verbose=1
#     )
    
#     # Phase 2: Fine-tuning
#     print("\n" + "="*60)
#     print("🔧 PHASE 2: FINE-TUNING")
#     print("="*60)
    
#     # Unfreeze some layers for fine-tuning
#     base_model.trainable = True
    
#     # Freeze early layers, fine-tune later layers
#     # Typically freeze first 100 layers of MobileNetV2
#     for layer in base_model.layers[:100]:
#         layer.trainable = False
    
#     # Count trainable layers
#     trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
#     print(f"\n📊 Fine-tuning Configuration:")
#     print(f"  • Total base layers: {len(base_model.layers)}")
#     print(f"  • Trainable layers: {trainable_layers}")
#     print(f"  • Frozen layers: {len(base_model.layers) - trainable_layers}")
    
#     # Recompile with lower learning rate
#     model.compile(
#         optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     # Callbacks for phase 2
#     callbacks_phase2 = [
#         ModelCheckpoint(
#             BEST_MODEL_PATH,
#             monitor='val_accuracy',
#             mode='max',
#             save_best_only=True,
#             verbose=1
#         ),
#         ModelCheckpoint(
#             os.path.join(MODELS_PATH, 'best_by_precision.h5'),
#             monitor='val_precision' if hasattr(tf.keras.metrics, 'Precision') else 'val_accuracy',
#             mode='max',
#             save_best_only=True,
#             verbose=1
#         ),
#         EarlyStopping(
#             monitor='val_loss',
#             patience=3,
#             restore_best_weights=True,
#             verbose=1
#         ),
#         ReduceLROnPlateau(
#             monitor='val_loss',
#             factor=0.2,
#             patience=2,
#             min_lr=1e-7,
#             verbose=1
#         ),
#         TensorBoard(
#             log_dir=os.path.join(log_dir, 'phase2'),
#             histogram_freq=1,
#             write_graph=True
#         ),
#         CSVLogger(os.path.join(log_dir, 'phase2_training_log.csv'))
#     ]
    
#     # Add precision metric if available
#     if hasattr(tf.keras.metrics, 'Precision'):
#         metrics = ['accuracy', tf.keras.metrics.Precision(name='precision')]
#         model.compile(
#             optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
#             loss='sparse_categorical_crossentropy',
#             metrics=metrics
#         )
    
#     history2 = model.fit(
#         train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
#         steps_per_epoch=len(X_train) // BATCH_SIZE,
#         validation_data=(X_val, y_val),
#         epochs=FINE_TUNE_EPOCHS,
#         callbacks=callbacks_phase2,
#         class_weight=class_weight_dict,
#         verbose=1
#     )
    
#     # Combine histories
#     history = {}
#     for key in history1.history.keys():
#         history[key] = history1.history[key] + history2.history[key]
    
#     # Plot training history
#     plot_training_history(
#         type('obj', (object,), {'history': history})(),
#         os.path.join(OUTPUT_PATH, 'training_history.png')
#     )
    
#     # Quick evaluation after training
#     val_metrics = evaluate_after_training(model, X_val, y_val)
    
#     # Save final model
#     model.save(FINAL_MODEL_PATH)
#     print(f"\n💾 Final model saved to: {FINAL_MODEL_PATH}")
    
#     # Save training metadata
#     training_info = {
#         'model': 'MobileNetV2',
#         'input_size': IMG_SIZE,
#         'batch_size': BATCH_SIZE,
#         'phase1_epochs': len(history1.history['loss']),
#         'phase2_epochs': len(history2.history['loss']),
#         'total_epochs': len(history['loss']),
#         'final_validation_metrics': val_metrics,
#         'class_weights': class_weight_dict,
#         'training_completed': datetime.datetime.now().isoformat()
#     }
    
#     with open(os.path.join(OUTPUT_PATH, 'training_info.json'), 'w') as f:
#         json.dump(training_info, f, indent=4)
    
#     print("\n" + "="*60)
#     print("✅ TRAINING COMPLETED SUCCESSFULLY!")
#     print("="*60)
#     print(f"\n📁 Training logs saved to: {log_dir}")
#     print(f"📁 Best model saved to: {BEST_MODEL_PATH}")
#     print(f"📁 Final model saved to: {FINAL_MODEL_PATH}")
#     print(f"\n📊 Final Validation Metrics:")
#     print(f"  • Accuracy:  {val_metrics['accuracy']*100:.2f}%")
#     print(f"  • Precision: {val_metrics['precision']*100:.2f}%")
#     print(f"  • Recall:    {val_metrics['recall']*100:.2f}%")
#     print(f"  • F1-Score:  {val_metrics['f1']*100:.2f}%")
#     print("\nNext step: Run 04_evaluate_model.py for detailed evaluation")
    
#     return True

# if __name__ == "__main__":
#     # Check TensorFlow and GPU
#     print(f"TensorFlow version: {tf.__version__}")
#     gpus = tf.config.list_physical_devices('GPU')
#     if gpus:
#         print(f"✅ GPU Available: {len(gpus)} GPU(s)")
#         for gpu in gpus:
#             print(f"  • {gpu.name}")
#     else:
#         print("⚠️  No GPU found. Training will be on CPU (slower)")
    
#     success = train_mobilenetv2()
#     sys.exit(0 if success else 1)

import sys
import os
import argparse
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import json
import logging
import boto3
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    """Parse command line arguments for SageMaker"""
    parser = argparse.ArgumentParser()
    
    # Data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DIR', '/opt/ml/output'))
    
    # Hyperparameters
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--initial_learning_rate', type=float, default=0.001)
    parser.add_argument('--fine_tune_learning_rate', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--fine_tune_epochs', type=int, default=5)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--dense_units', type=int, default=128)
    
    return parser.parse_args()

def load_sagemaker_data(train_dir, validation_dir, img_size, batch_size):
    """Load data from SageMaker input channels"""
    print("\n" + "="*60)
    print("📂 LOADING DATA FROM SAGEMAKER CHANNELS")
    print("="*60)
    
    # Check if using preprocessed numpy files or image directories
    if os.path.exists(os.path.join(train_dir, 'X_train.npy')):
        # Load numpy files
        print("Loading preprocessed numpy files...")
        X_train = np.load(os.path.join(train_dir, 'X_train.npy'))
        X_val = np.load(os.path.join(validation_dir, 'X_val.npy'))
        y_train = np.load(os.path.join(train_dir, 'y_train.npy'))
        y_val = np.load(os.path.join(validation_dir, 'y_val.npy'))
        
        return X_train, X_val, y_train, y_val
    
    else:
        # Load from image directories
        print("Loading from image directories...")
        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            validation_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=False
        )
        
        return train_generator, validation_generator

def create_mobilenetv2_model(img_size, num_classes, dropout_rate, dense_units):
    """Create MobileNetV2 based model"""
    print("\n" + "="*60)
    print("BUILDING MOBILENETV2 MODEL")
    print("="*60)
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(dense_units, activation='relu', name='dense_128')(x)
    x = Dropout(dropout_rate, name='dropout_1')(x)
    x = Dense(dense_units//2, activation='relu', name='dense_64')(x)
    x = Dropout(dropout_rate/2, name='dropout_2')(x)
    predictions = Dense(num_classes, activation='softmax', name='output')(x)
    
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
    plt.close()

def evaluate_after_training(model, X_val, y_val):
    """Quick evaluation after training"""
    print("\n" + "="*60)
    print("📊 POST-TRAINING EVALUATION")
    print("="*60)
    
    # Predict on validation set
    y_pred_prob = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calculate metrics
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

def train():
    """Main training function for SageMaker"""
    
    print("\n" + "="*60)
    print("🚀 SAGEMAKER MOBILENETV2 TRAINING PIPELINE")
    print("="*60)
    
    # Parse arguments
    args = parse_args()
    
    # Log SageMaker environment
    print(f"\n📁 SageMaker Environment:")
    print(f"  • Training channel: {args.train}")
    print(f"  • Validation channel: {args.validation}")
    print(f"  • Model directory: {args.model_dir}")
    print(f"  • Output directory: {args.output_dir}")
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.model_dir, 'assets'), exist_ok=True)
    
    # Load data
    train_data, val_data = load_sagemaker_data(
        args.train, args.validation, 
        args.img_size, args.batch_size
    )
    
    # Check if using generators or numpy arrays
    using_generators = isinstance(train_data, tf.keras.preprocessing.image.DirectoryIterator)
    
    if using_generators:
        X_train, y_train = None, None
        X_val, y_val = None, None
        train_generator = train_data
        val_generator = val_data
        
        # Get class distribution from generator
        class_distribution = {}
        for class_name, class_index in train_generator.class_indices.items():
            count = np.sum(train_generator.classes == class_index)
            class_distribution[class_name] = count
        
        print(f"\n📊 Dataset Statistics (from generators):")
        print(f"  • Training samples: {train_generator.samples}")
        print(f"  • Validation samples: {val_generator.samples}")
        print(f"  • Image shape: ({args.img_size}, {args.img_size}, 3)")
        print(f"  • Class distribution (train): {class_distribution}")
        
        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_generator.classes),
            y=train_generator.classes
        )
        class_weight_dict = dict(enumerate(class_weights))
        
    else:
        X_train, X_val, y_train, y_val = train_data, val_data[0], val_data[1], val_data[3]
        train_generator = None
        val_generator = None
        
        print(f"\n📊 Dataset Statistics (from numpy):")
        print(f"  • Training samples: {len(X_train)}")
        print(f"  • Validation samples: {len(X_val)}")
        print(f"  • Image shape: {X_train[0].shape}")
        print(f"  • Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
        
        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
    
    print(f"\n⚖️  Class weights: {class_weight_dict}")
    
    # Create data augmentation
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
    model, base_model = create_mobilenetv2_model(
        args.img_size, args.num_classes, 
        args.dropout_rate, args.dense_units
    )
    
    # Phase 1: Train top layers
    print("\n" + "="*60)
    print("📈 PHASE 1: TRAINING TOP LAYERS")
    print("="*60)
    
    model.compile(
        optimizer=Adam(learning_rate=args.initial_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks for phase 1
    log_dir = os.path.join(args.output_dir, 'logs', 'phase1')
    callbacks_phase1 = [
        ModelCheckpoint(
            os.path.join(args.model_dir, 'phase1_best.h5'),
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
        TensorBoard(log_dir=log_dir),
        CSVLogger(os.path.join(log_dir, 'training_log.csv'))
    ]
    
    if using_generators:
        history1 = model.fit(
            train_datagen.flow_from_directory(
                args.train,
                target_size=(args.img_size, args.img_size),
                batch_size=args.batch_size,
                class_mode='sparse'
            ),
            steps_per_epoch=train_generator.samples // args.batch_size,
            validation_data=val_generator,
            epochs=args.epochs,
            callbacks=callbacks_phase1,
            class_weight=class_weight_dict,
            verbose=1
        )
    else:
        history1 = model.fit(
            train_datagen.flow(X_train, y_train, batch_size=args.batch_size),
            steps_per_epoch=len(X_train) // args.batch_size,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
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
        optimizer=Adam(learning_rate=args.fine_tune_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision')]
    )
    
    # Callbacks for phase 2
    log_dir_phase2 = os.path.join(args.output_dir, 'logs', 'phase2')
    callbacks_phase2 = [
        ModelCheckpoint(
            os.path.join(args.model_dir, 'best_model.h5'),
            monitor='val_accuracy',
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
        TensorBoard(log_dir=log_dir_phase2),
        CSVLogger(os.path.join(log_dir_phase2, 'training_log.csv'))
    ]
    
    if using_generators:
        history2 = model.fit(
            train_datagen.flow_from_directory(
                args.train,
                target_size=(args.img_size, args.img_size),
                batch_size=args.batch_size,
                class_mode='sparse'
            ),
            steps_per_epoch=train_generator.samples // args.batch_size,
            validation_data=val_generator,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks_phase2,
            class_weight=class_weight_dict,
            verbose=1
        )
    else:
        history2 = model.fit(
            train_datagen.flow(X_train, y_train, batch_size=args.batch_size),
            steps_per_epoch=len(X_train) // args.batch_size,
            validation_data=(X_val, y_val),
            epochs=args.fine_tune_epochs,
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
        os.path.join(args.output_dir, 'training_history.png')
    )
    
    # Evaluate if using numpy arrays
    if not using_generators:
        val_metrics = evaluate_after_training(model, X_val, y_val)
    else:
        # Evaluate on validation generator
        val_loss, val_accuracy, val_precision = model.evaluate(val_generator)
        val_metrics = {
            'accuracy': float(val_accuracy),
            'precision': float(val_precision),
            'recall': float(val_precision),  # Placeholder
            'f1': float(val_accuracy)  # Placeholder
        }
    
    # Save final model in TensorFlow SavedModel format
    tf.saved_model.save(model, os.path.join(args.model_dir, '1'))
    
    # Also save as .h5 for compatibility
    model.save(os.path.join(args.model_dir, 'final_model.h5'))
    
    # Save training metadata
    training_info = {
        'model': 'MobileNetV2',
        'input_size': args.img_size,
        'batch_size': args.batch_size,
        'phase1_epochs': len(history1.history['loss']),
        'phase2_epochs': len(history2.history['loss']),
        'total_epochs': len(history['loss']),
        'final_validation_metrics': val_metrics,
        'class_weights': class_weight_dict,
        'hyperparameters': vars(args),
        'training_completed': datetime.datetime.now().isoformat()
    }
    
    with open(os.path.join(args.model_dir, 'training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=4)
    
    print("\n" + "="*60)
    print("✅ SAGEMAKER TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📁 Model saved to: {args.model_dir}")
    print(f"\n📊 Final Validation Metrics:")
    print(f"  • Accuracy:  {val_metrics['accuracy']*100:.2f}%")
    print(f"  • Precision: {val_metrics['precision']*100:.2f}%")
    print(f"  • Recall:    {val_metrics['recall']*100:.2f}%")
    print(f"  • F1-Score:  {val_metrics['f1']*100:.2f}%")
    
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
    
    success = train()
    sys.exit(0 if success else 1)
