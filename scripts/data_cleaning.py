# scripts/data_cleaning.py
"""
Script 01: Data Cleaning
Run this first to clean your dataset
"""

import sys
import os
import seaborn as sns
# Add parent directory to path so we can import config and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_cleaner import DataCleaner
from utils.face_detector import FaceDetector
from utils.visualization import Visualizer
from config import DATASET_PATH, CATEGORIES, OUTPUT_PATH

def main():
    """Main function for data cleaning"""
    print("\n" + "="*60)
    print("FACE MASK DETECTION - DATA CLEANING")
    print("="*60)
    print(f"Dataset path: {DATASET_PATH}")
    print("="*60)
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset path not found: {DATASET_PATH}")
        print("Please check your dataset path in config.py")
        return
    
    # Check for required subfolders
    missing_folders = []
    for category in CATEGORIES:
        category_path = os.path.join(DATASET_PATH, category)
        if not os.path.exists(category_path):
            missing_folders.append(category)
    
    if missing_folders:
        print(f"❌ Missing required folders: {missing_folders}")
        print(f"\nPlease create these folders and add your images:")
        for folder in missing_folders:
            print(f"  {os.path.join(DATASET_PATH, folder)}/")
        return
    
    # Initialize components
    cleaner = DataCleaner(dataset_path=DATASET_PATH, categories=CATEGORIES)
    face_detector = FaceDetector()
    visualizer = Visualizer()
    
    # Step 1: Check class balance
    print("\n📊 Step 1: Checking class balance...")
    class_counts = cleaner.check_class_balance()
    
    # Visualize class distribution
    if class_counts:
        visualizer.plot_class_distribution(
            class_counts,
            save_path=os.path.join(OUTPUT_PATH, 'class_distribution.png')
        )
    
    # Step 2: Remove corrupted images
    print("\n🧹 Step 2: Removing corrupted images...")
    corrupted = cleaner.remove_corrupted_images()
    
    # Step 3: Remove duplicates
    print("\n🔄 Step 3: Removing duplicate images...")
    duplicates = cleaner.remove_duplicates()
    
    # Step 4: Analyze image sizes
    print("\n📏 Step 4: Analyzing image sizes...")
    odd_sized = cleaner.analyze_image_sizes()
    
    # Step 5: Validate faces in dataset
    print("\n👤 Step 5: Validating faces in images...")
    no_face = face_detector.validate_dataset_faces(DATASET_PATH, CATEGORIES)
    cleaner.cleaning_report['no_face'] = no_face
    
    # Step 6: Generate final report
    print("\n📝 Step 6: Generating cleaning report...")
    report_path = cleaner.generate_report()
    
    # Print summary
    print("\n" + "="*60)
    print("CLEANING SUMMARY")
    print("="*60)
    print(f"✅ Corrupted images removed: {len(corrupted)}")
    print(f"✅ Duplicate images removed: {len(duplicates)}")
    print(f"⚠️  Odd-sized images flagged: {len(odd_sized)}")
    print(f"⚠️  Images without faces flagged: {len(no_face)}")
    print(f"\n📊 Full report saved to: {report_path}")
    
    print("\n✅ Data cleaning completed!")
    print("Next step: Run 02_preprocessing.py")

# This allows the script to be run directly or imported
if __name__ == "__main__":
    main()