# main.py
"""
Main entry point for Face Mask Detection System using MobileNetV2
Supports basic and advanced features with multiple execution modes
"""

import os
import sys
import argparse
import logging
import importlib
from typing import Optional
import config

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Project version
__version__ = '2.0.0'

def setup_logging():
    """Setup logging configuration"""
    log_file = os.path.join(config.LOGS_PATH, 'main.log')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

def create_directories():
    """Create all necessary directories if they don't exist"""
    directories = [
        config.MODELS_PATH,
        config.OUTPUT_PATH,
        config.SCREENSHOTS_PATH,
        config.CASCADES_PATH,
        config.LOGS_PATH,
        os.path.join(config.OUTPUT_PATH, 'evaluation_results'),
        os.path.join(config.OUTPUT_PATH, 'training_history'),
    ]
    
    print("\n📁 Creating necessary directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    # Validate paths
    if not config.validate_paths():
        logging.warning("Some paths may not be writable")

def print_banner():
    """Print welcome banner with version info"""
    banner = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           FACE MASK DETECTION SYSTEM                         ║
    ║                                                              ║
    ║                    with MobileNetV2                          ║
    ║                      Version {__version__}                           ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_gpu():
    """Check if GPU is available for TensorFlow"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"\n✅ GPU Available: {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"   • GPU {i}: {gpu.name}")
            return True
        else:
            print("\n⚠️  No GPU found. Training will be on CPU (slower)")
            return False
    except Exception as e:
        print(f"\n⚠️  Could not check GPU: {e}")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = {
        'tensorflow': 'tensorflow',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'tqdm': 'tqdm',
        'PIL': 'pillow'
    }
    
    missing_packages = []
    version_info = {}
    
    print("\n📦 Checking dependencies...")
    for package, pip_name in required_packages.items():
        try:
            module = __import__(package)
            if hasattr(module, '__version__'):
                version_info[package] = module.__version__
            print(f"  ✅ {package:<12} {version_info.get(package, '')}")
        except ImportError:
            missing_packages.append(pip_name)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print("\n❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install dependencies with:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_dataset(dataset_path):
    """Check if dataset exists and has images"""
    from config import CATEGORIES
    
    if not os.path.exists(dataset_path):
        print(f"\n❌ Dataset path not found: {dataset_path}")
        return False
    
    print(f"\n🔍 Checking dataset at: {dataset_path}")
    
    has_images = False
    missing_folders = []
    total_images = 0
    
    for category in CATEGORIES:
        category_path = os.path.join(dataset_path, category)
        if not os.path.exists(category_path):
            missing_folders.append(category)
            print(f"\n❌ Folder not found: {category_path}")
        else:
            images = [f for f in os.listdir(category_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            print(f"\n📊 {category}: {len(images)} images")
            total_images += len(images)
            if len(images) > 0:
                has_images = True
    
    if total_images > 0:
        print(f"\n📈 Total images: {total_images}")
    
    if missing_folders:
        print(f"\n❌ Missing required folders: {missing_folders}")
        print(f"\nExpected structure:")
        print(f"  {dataset_path}/")
        print(f"  ├── with_mask/    (contains mask images)")
        print(f"  └── without_mask/ (contains non-mask images)")
        return False
    
    return has_images

def download_cascades():
    """Download required Haar cascade files if missing"""
    cascade_files = [
        'haarcascade_frontalface_default.xml',
        'haarcascade_nose.xml',
        'haarcascade_mcs_nose.xml'
    ]
    
    missing_files = []
    for cascade_file in cascade_files:
        cascade_path = os.path.join(config.CASCADES_PATH, cascade_file)
        if not os.path.exists(cascade_path):
            missing_files.append(cascade_file)
    
    if missing_files:
        print("\n📥 Some cascade files are missing. Downloading...")
        try:
            import urllib.request
            
            base_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
            
            for cascade_file in missing_files:
                url = base_url + cascade_file
                save_path = os.path.join(config.CASCADES_PATH, cascade_file)
                
                print(f"  Downloading {cascade_file}...")
                urllib.request.urlretrieve(url, save_path)
                print(f"  ✅ Downloaded to {save_path}")
            
            print("\n✅ All cascade files downloaded")
        except Exception as e:
            print(f"\n❌ Failed to download cascades: {e}")
            return False
    
    return True

def print_config_summary(args):
    """Print configuration summary"""
    from config import get_config_summary
    
    print("\n" + "="*60)
    print("📋 CONFIGURATION SUMMARY")
    print("="*60)
    
    summary = get_config_summary()
    for key, value in summary.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n  Mode: {args.mode.upper()}")
    print(f"  Dataset: {config.DATASET_PATH}")
    print("="*60)

def import_script(script_name, function_name):
    """Dynamically import a function from a script"""
    try:
        module = importlib.import_module(f"scripts.{script_name}")
        return getattr(module, function_name)
    except ImportError as e:
        logging.error(f"Could not import {script_name}.{function_name}: {e}")
        return None

def main():
    """Main function"""
    print_banner()
    
    # Setup logging
    logger = setup_logging()
    logger.info("="*60)
    logger.info("Face Mask Detection System Started")
    logger.info("="*60)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Face Mask Detection System with MobileNetV2')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'clean', 'preprocess', 'train', 'evaluate', 
                                'webcam', 'advanced', 'download-cascades'],
                       help='Mode to run the system in')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Path to your dataset folder (containing with_mask/ and without_mask/)')
    parser.add_argument('--version', action='store_true',
                       help='Show version and exit')
    
    args = parser.parse_args()
    
    # Show version and exit
    if args.version:
        print(f"Face Mask Detection System version {__version__}")
        return 0
    
    # Update dataset path if provided
    if args.dataset:
        config.DATASET_PATH = args.dataset
        logger.info(f"Using custom dataset path: {config.DATASET_PATH}")
    else:
        logger.info(f"Using default dataset path: {config.DATASET_PATH}")
    
    # Create all necessary directories
    create_directories()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check GPU
    check_gpu()
    
    # Print configuration
    print_config_summary(args)
    
    # Download cascades if needed
    if args.mode == 'download-cascades':
        success = download_cascades()
        return 0 if success else 1
    
    # Check dataset (except for webcam/advanced modes)
    if args.mode not in ['webcam', 'advanced', 'download-cascades']:
        has_images = check_dataset(config.DATASET_PATH)
        if not has_images:
            logger.error("No valid images found in dataset!")
            
            if args.mode == 'all':
                print("\nCannot run full pipeline without dataset.")
                return 1
            else:
                response = input("\nContinue anyway? (y/n): ")
                if response.lower() != 'y':
                    return 1
    
    # Import and run scripts based on mode
    try:
        if args.mode == 'all' or args.mode == 'clean':
            print("\n" + "="*60)
            print("🧹 STEP 1: DATA CLEANING")
            print("="*60)
            clean_main = import_script('data_cleaning', 'main')
            if clean_main:
                clean_main()
        
        if args.mode == 'all' or args.mode == 'preprocess':
            print("\n" + "="*60)
            print("🔄 STEP 2: DATA PREPROCESSING")
            print("="*60)
            preprocess_main = import_script('preprocessing', 'main')
            if preprocess_main:
                preprocess_main()
        
        if args.mode == 'all' or args.mode == 'train':
            print("\n" + "="*60)
            print("🏋️  STEP 3: MODEL TRAINING (MobileNetV2)")
            print("="*60)
            train_func = import_script('train_model', 'train_mobilenetv2')
            if train_func:
                train_func()
        
        if args.mode == 'all' or args.mode == 'evaluate':
            print("\n" + "="*60)
            print("📊 STEP 4: MODEL EVALUATION")
            print("="*60)
            evaluate_func = import_script('evaluate_model', 'evaluate_model')
            if evaluate_func:
                evaluate_func()
        
        if args.mode == 'webcam':
            print("\n" + "="*60)
            print("🎥 STEP 5: WEBCAM DETECTION (Basic)")
            print("="*60)
            
            # Check if model exists
            model_path = os.path.join(config.MODELS_PATH, 'best_mobilenetv2.h5')
            if not os.path.exists(model_path):
                logger.error(f"Model not found at: {model_path}")
                print("Please train the model first with: python main.py --mode train")
                return 1
            
            # Download cascades if needed
            download_cascades()
            
            detector_class = import_script('webcam_detection', 'WebcamMaskDetector')
            if detector_class:
                detector = detector_class()
                detector.run()
        
        if args.mode == 'advanced':
            print("\n" + "="*60)
            print("🚀 STEP 6: ADVANCED WEBCAM DETECTION")
            print("Features: Multi-Face Tracking | Compliance Score | Mask Types")
            print("="*60)
            
            # Check if model exists
            model_path = os.path.join(config.MODELS_PATH, 'best_mobilenetv2.h5')
            if not os.path.exists(model_path):
                logger.error(f"Model not found at: {model_path}")
                print("Please train the model first with: python main.py --mode train")
                return 1
            
            # Download cascades if needed
            download_cascades()
            
            try:
                from scripts.advanced_mask_detection import AdvancedMaskDetector
                detector = AdvancedMaskDetector()
                detector.run()
            except ImportError as e:
                logger.error(f"Advanced features not available: {e}")
                print("\nAdvanced features require additional files:")
                print("  - utils/face_tracker.py")
                print("  - utils/mask_classifier.py")
                print("  - utils/compliance_dashboard.py")
                return 1
        
        print("\n" + "="*60)
        print("✅ COMPLETED SUCCESSFULLY!")
        print("="*60)
        logger.info("Application completed successfully")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Exiting...")
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())