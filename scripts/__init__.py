# scripts/__init__.py
"""
Scripts package for Face Mask Detection System
This file makes the scripts folder a proper Python package
"""

# Import key functions to make them available at package level
from .data_cleaning import main as clean_data
from .preprocessing import main as preprocess_data
from .train_model import train_mobilenetv2
from .evaluate_model import evaluate_model
from .webcam_detection import WebcamMaskDetector

# Define what gets imported with "from scripts import *"
__all__ = [
    'clean_data',
    'preprocess_data', 
    'train_mobilenetv2',
    'evaluate_model',
    'WebcamMaskDetector'
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'Your Name'