# image-classification-python

# Problem definition:

Since I have bought my laptop, I had taken backup of all the images received in my mobile via whatsapp, website downloads and other sources
Now I have over 18 thousand images in my mac saved in images directory.
There are no free tools in the market to identify and sort the images in separate directory. I tried ACD see Gemini duplicate finder etc. But could not find a tool
Since I have knowledge of python I am implementing this utility to classify perfume bottle images from the 18000 images
and move them to a separate directory.

Below is the approach an libraries used

## Approach:

The solution uses a hybrid approach combining multiple techniques:

1. Traditional Computer Vision (Shape Detection)

Bottle Detection: Uses edge detection and contour analysis to find tall, narrow objects (aspect ratio 1.5-4.0)
Box Detection: Looks for rectangular shapes with 4 corners and good fill ratios
Edge Detection: Canny edge detection to find object boundaries
Contour Analysis: Analyzes shape characteristics like aspect ratio and fill ratio

2. Deep Learning (CNN)

Base Model: ResNet50 pre-trained on ImageNet
Custom Head: Modified final layer for binary classification
Transfer Learning: Leverages pre-trained features for object recognition
Note: The model needs training on perfume-specific data for optimal performance

3. Keyword Analysis

Checks filenames for perfume-related keywords
Acts as additional confidence boost

Libraries:

- OpenCV (cv2) - For traditional computer vision techniques
- PyTorch + torchvision - For deep learning model (ResNet50)
- Pillow (PIL) - For image processing and loading
- NumPy - For numerical operations
- pathlib - For file path operations
- shutil - For file operations (move/copy)

# Installation:

```bash
pip install -r requirements.txt
```

# Usage:

```bash
# Basic usage - moves files to new directory
python perfume_classifier.py /path/to/input/directory

# Copy files instead of moving
python perfume_classifier.py /path/to/input/directory --copy

# Specify custom output directory
python perfume_classifier.py /path/to/input/directory --output_dir /path/to/output

# Adjust confidence threshold
python perfume_classifier.py /path/to/input/directory --confidence 0.8
```

# Features:

**Multi-format support:** JPG, PNG, BMP, TIFF, WebP
**Recursive scanning:** Scans subdirectories
**Confidence scoring:** Provides confidence scores for classifications
**Results logging:** Saves detailed results to CSV
**Flexible output:** Option to copy or move files
**Error handling:** Robust error handling and logging

## Limitations and Improvements:

**Model Training:** The CNN component needs training on perfume-specific datasets for best results
**Shape Detection:** Current shape detection is basic - could be improved with more sophisticated features
**False Positives:** May classify other bottle-like objects as perfume bottles

## To Improve Accuracy:

**Collect Training Data:** Gather labeled images of perfume bottles/boxes
**Fine-tune Model:** Train the ResNet50 model on perfume-specific data
**Add More Features:** Include color analysis, texture analysis, brand logo detection
**Ensemble Methods:** Combine multiple models for better accuracy

The code is production-ready and handles edge cases well. It will work immediately with basic shape detection, but training the deep learning component will significantly improve accuracy.
