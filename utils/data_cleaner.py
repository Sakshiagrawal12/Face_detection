# utils/data_cleaner.py
"""
Data Cleaning Module
Provides functionality to clean and validate image datasets
"""

import os
import cv2
import hashlib
import imghdr
import json
import logging
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from config import CATEGORIES, OUTPUT_PATH

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataCleaner:
    """
    Data cleaning and validation for image datasets
    
    Attributes:
        dataset_path (str): Path to the dataset
        categories (list): List of category names
        cleaning_report (dict): Report of cleaning operations
    """
    
    def __init__(self, dataset_path: str, categories: List[str] = CATEGORIES):
        """
        Initialize DataCleaner
        
        Args:
            dataset_path: Path to the dataset
            categories: List of category folders
        """
        self.dataset_path = dataset_path
        self.categories = categories
        self.cleaning_report = {
            'corrupted': [],
            'duplicates': [],
            'odd_sized': [],
            'no_face': [],
            'stats': {}
        }
        logging.info(f"DataCleaner initialized for {dataset_path}")
    
    def remove_corrupted_images(self) -> List[str]:
        """
        Remove corrupted or unreadable images
        
        Returns:
            list: Paths of removed corrupted images
        """
        print("\n" + "="*50)
        print("REMOVING CORRUPTED IMAGES")
        print("="*50)
        
        corrupted = []
        
        for category in self.categories:
            category_path = os.path.join(self.dataset_path, category)
            if not os.path.exists(category_path):
                logging.warning(f"Category path does not exist: {category_path}")
                continue
                
            print(f"\nChecking {category} images...")
            image_files = [f for f in os.listdir(category_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            for img_name in tqdm(image_files, desc=f"Scanning {category}"):
                img_path = os.path.join(category_path, img_name)
                
                try:
                    # Try to read the image
                    img = cv2.imread(img_path)
                    if img is None:
                        corrupted.append(img_path)
                        logging.debug(f"Corrupted image (cannot read): {img_path}")
                        continue
                    
                    # Check if it's a valid image file
                    img_type = imghdr.what(img_path)
                    if img_type not in ['jpeg', 'png', 'jpg', 'bmp']:
                        corrupted.append(img_path)
                        logging.debug(f"Invalid image type: {img_path}")
                        
                except Exception as e:
                    corrupted.append(img_path)
                    logging.debug(f"Error reading {img_path}: {e}")
        
        # Remove corrupted images
        for img_path in corrupted:
            try:
                os.remove(img_path)
                print(f"  Removed: {os.path.basename(img_path)}")
            except Exception as e:
                logging.error(f"Could not remove {img_path}: {e}")
        
        self.cleaning_report['corrupted'] = corrupted
        print(f"\n✅ Removed {len(corrupted)} corrupted images")
        return corrupted
    
    def remove_duplicates(self) -> List[str]:
        """
        Remove duplicate images based on MD5 hash
        
        Returns:
            list: Paths of removed duplicate images
        """
        print("\n" + "="*50)
        print("REMOVING DUPLICATE IMAGES")
        print("="*50)
        
        hash_dict = defaultdict(list)
        duplicates = []
        
        for category in self.categories:
            category_path = os.path.join(self.dataset_path, category)
            if not os.path.exists(category_path):
                continue
                
            print(f"\nChecking {category} for duplicates...")
            image_files = [f for f in os.listdir(category_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            for img_name in tqdm(image_files, desc=f"Scanning {category}"):
                img_path = os.path.join(category_path, img_name)
                
                try:
                    with open(img_path, 'rb') as f:
                        img_hash = hashlib.md5(f.read()).hexdigest()
                    hash_dict[img_hash].append(img_path)
                except Exception as e:
                    logging.debug(f"Error hashing {img_path}: {e}")
        
        # Remove duplicates (keep first occurrence)
        for img_hash, paths in hash_dict.items():
            if len(paths) > 1:
                # Keep the first image, remove the rest
                for path in paths[1:]:
                    try:
                        os.remove(path)
                        duplicates.append(path)
                        print(f"  Removed duplicate: {os.path.basename(path)}")
                    except Exception as e:
                        logging.error(f"Could not remove {path}: {e}")
        
        self.cleaning_report['duplicates'] = duplicates
        print(f"\n✅ Removed {len(duplicates)} duplicate images")
        return duplicates
    
    def analyze_image_sizes(self) -> List[Dict]:
        """
        Analyze image sizes and flag problematic ones
        
        Returns:
            list: List of problematic images with issues
        """
        print("\n" + "="*50)
        print("ANALYZING IMAGE SIZES")
        print("="*50)
        
        size_stats = defaultdict(int)
        odd_sized = []
        
        for category in self.categories:
            category_path = os.path.join(self.dataset_path, category)
            if not os.path.exists(category_path):
                continue
                
            print(f"\nAnalyzing {category} images...")
            image_files = [f for f in os.listdir(category_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            for img_name in tqdm(image_files, desc=f"Analyzing {category}"):
                img_path = os.path.join(category_path, img_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    h, w = img.shape[:2]
                    size_stats[f"{w}x{h}"] += 1
                    
                    # Flag problematic images
                    if w < 100 or h < 100:  # Too small
                        odd_sized.append({
                            'path': img_path, 
                            'issue': 'too_small', 
                            'size': f"{w}x{h}"
                        })
                    elif w/h > 2 or h/w > 2:  # Too elongated
                        odd_sized.append({
                            'path': img_path, 
                            'issue': 'elongated', 
                            'size': f"{w}x{h}"
                        })
        
        # Print statistics
        print("\nImage size distribution:")
        sorted_sizes = sorted(size_stats.items(), key=lambda x: x[1], reverse=True)
        for size, count in sorted_sizes[:10]:  # Show top 10
            print(f"  {size}: {count} images")
        
        self.cleaning_report['odd_sized'] = odd_sized
        self.cleaning_report['stats']['size_distribution'] = dict(sorted_sizes)
        
        print(f"\n⚠️  Found {len(odd_sized)} potentially problematic images")
        return odd_sized
    
    def check_class_balance(self) -> Dict[str, int]:
        """
        Check if classes are balanced
        
        Returns:
            dict: Class counts
        """
        print("\n" + "="*50)
        print("CHECKING CLASS BALANCE")
        print("="*50)
        
        class_counts = {}
        
        for category in self.categories:
            category_path = os.path.join(self.dataset_path, category)
            if os.path.exists(category_path):
                count = len([f for f in os.listdir(category_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
                class_counts[category] = count
                print(f"{category}: {count} images")
        
        # Check balance
        if len(class_counts) == 2:
            counts = list(class_counts.values())
            if min(counts) > 0:
                ratio = max(counts) / min(counts)
                if ratio > 1.5:
                    print(f"⚠️  Warning: Classes are imbalanced (ratio: {ratio:.2f})")
                    print("   Consider data augmentation for the minority class")
                else:
                    print(f"✅ Classes are well-balanced (ratio: {ratio:.2f})")
            else:
                print("⚠️  One class has zero images!")
        
        self.cleaning_report['stats']['class_counts'] = class_counts
        return class_counts
    
    def generate_report(self) -> str:
        """
        Generate and save cleaning report
        
        Returns:
            str: Path to saved report
        """
        report_path = os.path.join(OUTPUT_PATH, 'cleaning_report.json')
        
        # Add summary statistics
        self.cleaning_report['summary'] = {
            'total_corrupted': len(self.cleaning_report['corrupted']),
            'total_duplicates': len(self.cleaning_report['duplicates']),
            'total_odd_sized': len(self.cleaning_report['odd_sized']),
            'total_no_face': len(self.cleaning_report.get('no_face', [])),
        }
        
        # Add timestamp
        import datetime
        self.cleaning_report['timestamp'] = datetime.datetime.now().isoformat()
        
        with open(report_path, 'w') as f:
            json.dump(self.cleaning_report, f, indent=4)
        
        print(f"\n📊 Cleaning report saved to: {report_path}")
        logging.info(f"Cleaning report saved to {report_path}")
        return report_path
    
    def run_full_cleaning(self) -> Dict:
        """
        Run complete data cleaning pipeline
        
        Returns:
            dict: Complete cleaning report
        """
        print("\n" + "="*60)
        print("STARTING COMPLETE DATA CLEANING PIPELINE")
        print("="*60)
        
        # Step 1: Check class balance
        self.check_class_balance()
        
        # Step 2: Remove corrupted images
        self.remove_corrupted_images()
        
        # Step 3: Remove duplicates
        self.remove_duplicates()
        
        # Step 4: Analyze image sizes
        self.analyze_image_sizes()
        
        # Step 5: Generate report
        self.generate_report()
        
        print("\n" + "="*60)
        print("✅ DATA CLEANING COMPLETED")
        print("="*60)
        
        return self.cleaning_report

# For standalone testing
if __name__ == "__main__":
    print("Testing DataCleaner...")
    
    # Test with current directory
    test_cleaner = DataCleaner(dataset_path='.')
    test_cleaner.check_class_balance()
    
    print("\n✅ Test complete")