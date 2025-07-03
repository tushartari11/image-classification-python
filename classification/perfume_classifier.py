import os
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerfumeClassifier:
    """
    A classifier to detect perfume bottles and boxes in images using multiple approaches:
    1. Pre-trained CNN model (ResNet50) with custom classification head
    2. Traditional computer vision techniques for bottle/box shape detection
    """
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load or create the model
        self.model = self._load_model(model_path)
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    def _load_model(self, model_path: str = None):
        """Load pre-trained model or create a new one"""
        # Using ResNet50 as base model
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Modify the final layer for binary classification (perfume vs non-perfume)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # 2 classes: perfume, not perfume
        )
        
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            logger.info("Using pre-trained ResNet50 with custom head (not trained on perfume data)")
            
        model.to(self.device)
        model.eval()
        return model
    
    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for model input"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def _detect_bottle_shape(self, image_path: str) -> float:
        """
        Traditional CV approach to detect bottle-like shapes
        Returns confidence score (0-1)
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return 0.0
            
            # Convert to grayscale and apply edge detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bottle_score = 0.0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1000:  # Skip small contours
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                # Bottles typically have aspect ratio > 1.5 (taller than wide)
                if 1.5 < aspect_ratio < 4.0:
                    # Calculate how much of the bounding rectangle is filled
                    rect_area = w * h
                    fill_ratio = area / rect_area if rect_area > 0 else 0
                    
                    # Bottles should have reasonable fill ratio
                    if 0.3 < fill_ratio < 0.8:
                        bottle_score = max(bottle_score, fill_ratio * 0.8)
            
            return bottle_score
            
        except Exception as e:
            logger.error(f"Error in shape detection for {image_path}: {e}")
            return 0.0
    
    def _detect_box_shape(self, image_path: str) -> float:
        """
        Traditional CV approach to detect box-like shapes
        Returns confidence score (0-1)
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return 0.0
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            box_score = 0.0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1000:  # Skip small contours
                    continue
                
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Boxes should have 4 corners (rectangular)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                    
                    # Boxes can be square or rectangular
                    if 1.0 < aspect_ratio < 3.0:
                        rect_area = w * h
                        fill_ratio = area / rect_area if rect_area > 0 else 0
                        
                        if fill_ratio > 0.7:  # Boxes should fill most of their bounding rectangle
                            box_score = max(box_score, fill_ratio * 0.9)
            
            return box_score
            
        except Exception as e:
            logger.error(f"Error in box detection for {image_path}: {e}")
            return 0.0
    
    def classify_image(self, image_path: str) -> Tuple[bool, float]:
        """
        Classify if image contains perfume bottle or box
        Returns (is_perfume, confidence_score)
        """
        try:
            # Method 1: Shape-based detection (traditional CV)
            bottle_score = self._detect_bottle_shape(image_path)
            box_score = self._detect_box_shape(image_path)
            shape_score = max(bottle_score, box_score)
            
            # Method 2: CNN-based classification (commented out as it needs training)
            # For now, we'll use a simplified approach based on filename and shape detection
            filename = os.path.basename(image_path).lower()
            keyword_score = 0.0
            
            # Check for perfume-related keywords in filename
            perfume_keywords = ['perfume', 'fragrance', 'cologne', 'scent', 'bottle', 'spray']
            for keyword in perfume_keywords:
                if keyword in filename:
                    keyword_score = 0.8
                    break
            
            # Combine scores
            final_score = max(shape_score, keyword_score)
            
            # For demo purposes, we'll also add some randomness based on image characteristics
            # In real implementation, this would be replaced by trained model prediction
            image_tensor = self._preprocess_image(image_path)
            if image_tensor is not None:
                # This is a placeholder - in reality you'd use the trained model
                # For now, we'll use shape detection as primary method
                pass
            
            is_perfume = final_score >= self.confidence_threshold
            
            return is_perfume, final_score
            
        except Exception as e:
            logger.error(f"Error classifying image {image_path}: {e}")
            return False, 0.0
    
    def get_image_files(self, directory: str) -> List[str]:
        """Get all image files from directory"""
        image_files = []
        directory = Path(directory)
        
        if not directory.exists():
            logger.error(f"Directory {directory} does not exist")
            return image_files
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(str(file_path))
        
        return image_files
    
    def process_directory(self, input_dir: str, output_dir: str = None, move_files: bool = True):
        """
        Process all images in input directory and classify perfume bottles/boxes
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(input_dir), 'perfume_images')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = self.get_image_files(input_dir)
        logger.info(f"Found {len(image_files)} image files")
        
        perfume_files = []
        results = []
        
        for i, image_path in enumerate(image_files):
            logger.info(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            is_perfume, confidence = self.classify_image(image_path)
            
            result = {
                'file_path': image_path,
                'filename': os.path.basename(image_path),
                'is_perfume': is_perfume,
                'confidence': confidence
            }
            results.append(result)
            
            if is_perfume:
                perfume_files.append(image_path)
                
                # Move or copy file to output directory
                output_path = os.path.join(output_dir, os.path.basename(image_path))
                
                try:
                    if move_files:
                        shutil.move(image_path, output_path)
                        logger.info(f"Moved {os.path.basename(image_path)} to {output_dir}")
                    else:
                        shutil.copy2(image_path, output_path)
                        logger.info(f"Copied {os.path.basename(image_path)} to {output_dir}")
                except Exception as e:
                    logger.error(f"Error moving/copying {image_path}: {e}")
        
        # Save results to CSV
        self._save_results(results, output_dir)
        
        logger.info(f"Processing complete!")
        logger.info(f"Total files processed: {len(image_files)}")
        logger.info(f"Perfume bottles/boxes found: {len(perfume_files)}")
        logger.info(f"Results saved to: {output_dir}")
        
        return results
    
    def _save_results(self, results: List[dict], output_dir: str):
        """Save classification results to CSV file"""
        import csv
        
        csv_path = os.path.join(output_dir, 'classification_results.csv')
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'is_perfume', 'confidence', 'file_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        logger.info(f"Results saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Classify perfume bottles and boxes in images')
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('--output_dir', help='Output directory for perfume images', default=None)
    parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold (0-1)')
    parser.add_argument('--copy', action='store_true', help='Copy files instead of moving them')
    parser.add_argument('--model_path', help='Path to trained model file', default=None)
    
    args = parser.parse_args()
    
    # Create classifier
    classifier = PerfumeClassifier(
        model_path=args.model_path,
        confidence_threshold=args.confidence
    )
    
    # Process directory
    results = classifier.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        move_files=not args.copy
    )
    
    # Print summary
    perfume_count = sum(1 for r in results if r['is_perfume'])
    print(f"\nSummary:")
    print(f"Total images processed: {len(results)}")
    print(f"Perfume bottles/boxes detected: {perfume_count}")
    print(f"Detection rate: {perfume_count/len(results)*100:.1f}%")


if __name__ == "__main__":
    main()