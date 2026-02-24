# scripts/preprocessing.py
"""
Script 02: Data Preprocessing
Run this after cleaning to prepare data for training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import class_weight
import joblib
import logging
from collections import Counter
import matplotlib.pyplot as plt
from config import DATASET_PATH, CATEGORIES, IMG_SIZE, OUTPUT_PATH, BATCH_SIZE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_image(img_path):
    """Validate if image can be read and has correct properties"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False, "Cannot read image"
        
        h, w = img.shape[:2]
        if h < 50 or w < 50:
            return False, f"Image too small: {w}x{h}"
        
        if h/w > 3 or w/h > 3:
            return False, f"Extreme aspect ratio: {w}x{h}"
        
        return True, "Valid"
    except Exception as e:
        return False, str(e)

def load_and_preprocess_images():
    """Load images and preprocess them with validation"""
    print("\n" + "="*60)
    print("LOADING AND PREPROCESSING IMAGES")
    print("="*60)
    print(f"Loading from: {DATASET_PATH}")
    
    data = []
    labels = []
    invalid_images = []
    image_stats = {'min_size': float('inf'), 'max_size': 0, 'total_size': 0}
    
    for category in CATEGORIES:
        category_path = os.path.join(DATASET_PATH, category)
        if not os.path.exists(category_path):
            logging.error(f"Category path not found: {category_path}")
            continue
            
        class_num = CATEGORIES.index(category)
        print(f"\n📂 Loading {category} images...")
        
        # Get all image files
        image_files = [f for f in os.listdir(category_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if len(image_files) == 0:
            logging.warning(f"No images found in {category_path}")
            continue
        
        for img_name in tqdm(image_files, desc=f"Processing {category}"):
            img_path = os.path.join(category_path, img_name)
            
            # Validate image
            is_valid, message = validate_image(img_path)
            if not is_valid:
                invalid_images.append({'path': img_path, 'reason': message})
                continue
            
            try:
                # Read and preprocess image
                img = cv2.imread(img_path)
                
                # Update stats
                h, w = img.shape[:2]
                image_stats['min_size'] = min(image_stats['min_size'], min(h, w))
                image_stats['max_size'] = max(image_stats['max_size'], max(h, w))
                image_stats['total_size'] += (h * w)
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                
                # Normalize to [0, 1]
                img = img.astype('float32') / 255.0
                
                data.append(img)
                labels.append(class_num)
                
            except Exception as e:
                logging.error(f"Error processing {img_name}: {e}")
                invalid_images.append({'path': img_path, 'reason': str(e)})
    
    if len(data) == 0:
        raise ValueError("No images were loaded! Please check your dataset.")
    
    # Log invalid images
    if invalid_images:
        print(f"\n⚠️  Found {len(invalid_images)} invalid images:")
        for inv in invalid_images[:5]:  # Show first 5
            print(f"  • {os.path.basename(inv['path'])}: {inv['reason']}")
        if len(invalid_images) > 5:
            print(f"  ... and {len(invalid_images) - 5} more")
    
    # Log image statistics
    print(f"\n📊 Image Statistics:")
    print(f"  • Smallest dimension: {image_stats['min_size']}px")
    print(f"  • Largest dimension: {image_stats['max_size']}px")
    print(f"  • Average size: {image_stats['total_size'] / len(data):.0f}px²")
    
    return np.array(data), np.array(labels), invalid_images

def apply_data_augmentation(X_train, y_train):
    """Apply data augmentation to training set"""
    print("\n🔄 Applying data augmentation...")
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Fit the generator
    datagen.fit(X_train)
    
    print("✅ Data augmentation configured")
    return datagen

def balance_classes(X, y):
    """Balance classes using augmentation if needed"""
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    print(f"\n⚖️  Class distribution before balancing: {class_counts}")
    
    if len(class_counts) == 2:
        minority_class = 0 if counts[0] < counts[1] else 1
        majority_count = max(counts)
        minority_count = min(counts)
        
        if minority_count < majority_count * 0.8:  # If imbalance > 20%
            print(f"⚠️  Class imbalance detected. Minority class has {minority_count} samples")
            print("Consider using class weights during training instead of augmentation")
    
    return X, y, class_counts

def visualize_samples(X, y, num_samples=10, save_path=None):
    """Visualize sample images"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    indices = np.random.choice(len(X), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        axes[i].imshow(X[idx])
        axes[i].set_title(f"Class: {CATEGORIES[y[idx]]}")
        axes[i].axis('off')
    
    plt.suptitle("Sample Images from Dataset", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    """Main preprocessing function"""
    print("\n" + "="*60)
    print("FACE MASK DETECTION - DATA PREPROCESSING")
    print("="*60)
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print("="*60)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Step 1: Load and preprocess images
    try:
        X, y, invalid_images = load_and_preprocess_images()
    except Exception as e:
        logging.error(f"Error loading images: {e}")
        return False
    
    print(f"\n📊 Total images loaded: {len(X)}")
    print(f"Image shape: {X[0].shape}")
    print(f"Pixel range: [{X.min():.3f}, {X.max():.3f}]")
    
    # Step 2: Check class balance
    X, y, class_counts = balance_classes(X, y)
    
    # Step 3: Split dataset
    print("\n📊 Splitting dataset...")
    
    # First split: training vs temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Second split: validation vs test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n📈 Dataset Split:")
    print(f"  • Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  • Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  • Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Step 4: Calculate class weights for imbalanced training
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\n⚖️  Class weights for training: {class_weight_dict}")
    
    # Step 5: Visualize sample images
    print("\n👁️  Visualizing sample images...")
    visualize_samples(
        X_train[:10], y_train[:10], 
        save_path=os.path.join(OUTPUT_PATH, 'sample_images.png')
    )
    
    # Step 6: Save preprocessed data
    print("\n💾 Saving preprocessed data...")
    
    # Save numpy arrays
    np.save(os.path.join(OUTPUT_PATH, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_PATH, 'X_val.npy'), X_val)
    np.save(os.path.join(OUTPUT_PATH, 'X_test.npy'), X_test)
    np.save(os.path.join(OUTPUT_PATH, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_PATH, 'y_val.npy'), y_val)
    np.save(os.path.join(OUTPUT_PATH, 'y_test.npy'), y_test)
    
    # Save label binarizer
    lb = LabelBinarizer()
    lb.fit(y)
    joblib.dump(lb, os.path.join(OUTPUT_PATH, 'label_binarizer.pkl'))
    
    # Save class weights
    joblib.dump(class_weight_dict, os.path.join(OUTPUT_PATH, 'class_weights.pkl'))
    
    # Save preprocessing info
    preprocessing_info = {
        'total_images': len(X),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'image_size': IMG_SIZE,
        'class_distribution': {CATEGORIES[i]: int(count) for i, count in enumerate(np.bincount(y))},
        'class_weights': class_weight_dict,
        'invalid_images': len(invalid_images)
    }
    
    import json
    with open(os.path.join(OUTPUT_PATH, 'preprocessing_info.json'), 'w') as f:
        json.dump(preprocessing_info, f, indent=4)
    
    print("✅ Data saved successfully!")
    print(f"  • Training data: {os.path.join(OUTPUT_PATH, 'X_train.npy')}")
    print(f"  • Validation data: {os.path.join(OUTPUT_PATH, 'X_val.npy')}")
    print(f"  • Test data: {os.path.join(OUTPUT_PATH, 'X_test.npy')}")
    print(f"  • Class weights: {os.path.join(OUTPUT_PATH, 'class_weights.pkl')}")
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📋 Summary:")
    print(f"  • Total images processed: {len(X)}")
    print(f"  • Invalid images skipped: {len(invalid_images)}")
    print(f"  • Training samples: {len(X_train)}")
    print(f"  • Validation samples: {len(X_val)}")
    print(f"  • Test samples: {len(X_test)}")
    print(f"\n📁 All data saved to: {OUTPUT_PATH}")
    print("\nNext step: Run 03_train_model.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)