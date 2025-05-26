#!/usr/bin/env python
import argparse
import os
import re
from PIL import Image
import pytesseract
import cv2
import numpy as np
import json

def preprocess_image(image_path):
    """
    Preprocess the image to improve OCR results for detecting "Pred:" text
    in the top right corner
    """
    # Read image with OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Extract the top right corner (adjustable based on image size)
    # Usually text is in the top right, so focus on that area
    top_right_width = min(width // 2, 600)  # Increased to capture more of the text
    top_right_height = min(height // 3, 300)  # Increased to capture more context
    
    # Define the ROI (Region of Interest)
    x_start = max(0, width - top_right_width)
    y_start = 0
    roi = img[y_start:y_start + top_right_height, x_start:width]
    
    # Check if ROI is valid
    if roi.size == 0:
        print(f"Warning: ROI is empty for {image_path}, using full image")
        roi = img
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply adaptive thresholding - better for varying light conditions
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Increase size for better OCR
    scale_factor = 2.0  # Increased scale factor
    resized = cv2.resize(thresh, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # Noise removal with median blur
    processed = cv2.medianBlur(resized, 3)
    
    # Dilation to make text clearer
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.dilate(processed, kernel, iterations=1)
    
    # Try inverting the image for better results in some cases
    inverted = cv2.bitwise_not(processed)
    
    # For debugging, save the processed and inverted images
    debug_dir = "debug_images"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    basename = os.path.basename(image_path)
    debug_path = os.path.join(debug_dir, f"processed_{basename}")
    inverted_path = os.path.join(debug_dir, f"inverted_{basename}")
    cv2.imwrite(debug_path, processed)
    cv2.imwrite(inverted_path, inverted)
    
    return processed, inverted

def extract_text(image_path):
    """
    Extract text from the given image using OCR
    """
    try:
        # Preprocess the image
        processed_img, inverted_img = preprocess_image(image_path)
        
        # OCR configurations to try
        configs = [
            # Standard config
            r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789%():. "',
            # Try with a different page segmentation mode
            r'--oem 3 --psm 11 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789%():. "',
            # Try with a different OCR engine mode
            r'--oem 1 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789%():. "'
        ]
        
        # Try both processed and inverted images with multiple configs
        all_texts = []
        for config in configs:
            text1 = pytesseract.image_to_string(processed_img, config=config)
            text2 = pytesseract.image_to_string(inverted_img, config=config)
            all_texts.append(text1)
            all_texts.append(text2)
        
        # Combine all extracted texts
        combined_text = '\n'.join(all_texts)
        
        # Print raw text for debugging
        print(f"Raw extracted text from {os.path.basename(image_path)}:")
        print(combined_text)
        print("-" * 40)
        
        return combined_text
    except Exception as e:
        print(f"Error extracting text from {image_path}: {str(e)}")
        return ""

def extract_prediction_and_confidence(text):
    """
    Extract prediction and confidence from text containing "Pred:" pattern
    and confidence percentage in parentheses, ignoring all other text
    """
    prediction = None
    confidence = None
    
    # Normalize common OCR errors in the text
    normalized_text = text.replace('l', '1').replace('O', '0')
    normalized_text = re.sub(r'[Pp]\s*[rR]\s*[eE]\s*[dD]\s*:', 'Pred:', normalized_text)
    normalized_text = re.sub(r'[Pp]\s*[rR]\s*[eE]\s*[dD]', 'Pred', normalized_text)
    
    # Look for the basic pattern: Pred: [prediction] ([confidence]%)
    # This specific pattern: anything after "Pred:" before "(" is prediction, 
    # and anything between "(" and ")" is confidence
    basic_pattern = r'[Pp]red:?\s*(.*?)\s*\((\d+(?:\.\d+)?)%\)'
    matches = re.findall(basic_pattern, normalized_text)
    
    if matches:
        # Use the first match
        prediction = matches[0][0].strip()
        confidence = float(matches[0][1])
        return prediction, confidence
    
    # If no matches with the basic pattern, try a more general approach
    # Split by lines and look for lines with "Pred:" or similar
    lines = text.split('\n')
    for line in lines:
        # Look for variations of "Pred:"
        if re.search(r'[Pp][rR]?[eE]?[dD]:', line):
            # Try to extract prediction and confidence
            pred_parts = re.split(r'[Pp][rR]?[eE]?[dD]:', line, maxsplit=1)
            if len(pred_parts) > 1:
                pred_text = pred_parts[1].strip()
                
                # Extract confidence from parentheses if available
                confidence_match = re.search(r'\((\d+(?:\.\d+)?)%\)', pred_text)
                if confidence_match:
                    try:
                        confidence = float(confidence_match.group(1))
                        # Extract prediction (everything before the parenthesis)
                        pred_end_index = pred_text.find('(')
                        if pred_end_index > 0:
                            prediction = pred_text[:pred_end_index].strip()
                    except ValueError:
                        pass
                else:
                    # No confidence found, use the entire text as prediction
                    prediction = pred_text
                
                if prediction:
                    break
    
    return prediction, confidence

def process_images(input_folder, output_file, json_output=None, max_images=5):
    """
    Process all images in the input folder and extract predictions with confidence levels,
    focusing only on text with "Pred:" pattern
    """
    results = []
    
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Do the same for JSON output
    if json_output:
        json_dir = os.path.dirname(json_output)
        if json_dir and not os.path.exists(json_dir):
            os.makedirs(json_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [
        os.path.join(input_folder, file) 
        for file in os.listdir(input_folder) 
        if os.path.splitext(file.lower())[1] in image_extensions
    ]
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    # Limit to max_images if specified
    if max_images and max_images > 0:
        image_files = image_files[:max_images]
        print(f"Processing only the first {len(image_files)} images (limited by max_images={max_images})")
    else:
        print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_path in image_files:
        print(f"Processing {image_path}")
        extracted_text = extract_text(image_path)
        
        # Extract prediction and confidence
        prediction, confidence = extract_prediction_and_confidence(extracted_text)
        
        # Get the filename
        filename = os.path.basename(image_path)
        
        # Get the class name from the filename (assuming format: n02749479_assault_rifle_pred.jpg)
        class_name = None
        if "_pred" in filename:
            name_parts = filename.split('_pred')[0].split('_')
            if len(name_parts) > 1:  # Skip the numerical prefix
                class_name = '_'.join(name_parts[1:])
        
        # Store results
        result = {
            "filename": filename,
            "class_name": class_name,
            "image_path": image_path,
            "prediction": prediction if prediction else "Unknown",
            "confidence": confidence if confidence else None,
            "raw_text": extracted_text
        }
        
        results.append(result)
        
        # Output results to stdout for monitoring
        print(f"\nImage: {filename}")
        if class_name:
            print(f"Class: {class_name}")
        print(f"Prediction: {result['prediction']}")
        if result['confidence']:
            print(f"Confidence: {result['confidence']}%")
        else:
            print("Confidence: Not found")
        print("-" * 40)
    
    # Write results to text output file
    with open(output_file, 'w') as f:
        f.write(f"Processed {len(results)} images\n")
        f.write("=" * 50 + "\n\n")
        
        for result in results:
            f.write(f"Image: {result['filename']}\n")
            if result['class_name']:
                f.write(f"Class: {result['class_name']}\n")
            f.write(f"Prediction: {result['prediction']}\n")
            if result['confidence']:
                f.write(f"Confidence: {result['confidence']}%\n")
            else:
                f.write(f"Confidence: Not found\n")
            f.write("-" * 40 + "\n")
    
    # Write results to JSON if requested
    if json_output:
        with open(json_output, 'w') as f:
            json_results = [{
                "filename": r["filename"],
                "class_name": r["class_name"],
                "prediction": r["prediction"],
                "confidence": r["confidence"]
            } for r in results]
            json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    if json_output:
        print(f"JSON results saved to {json_output}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Extract predictions and confidence levels from images using OCR')
    parser.add_argument('--input', type=str, default='results/alexnet_split/images/split_12',
                        help='Input folder containing images (default: results/alexnet_split/images/split_12)')
    parser.add_argument('--output', type=str, default='results/alexnet_split/extracted_predictions.txt',
                        help='Output text file to save extracted predictions')
    parser.add_argument('--json', type=str, default='results/alexnet_split/predictions.json',
                        help='Output JSON file to save structured prediction data')
    parser.add_argument('--max-images', type=int, default=5,
                        help='Maximum number of images to process (default: 5)')
    
    args = parser.parse_args()
    
    process_images(args.input, args.output, args.json, args.max_images)

if __name__ == "__main__":
    main()
