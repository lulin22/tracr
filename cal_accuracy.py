#!/usr/bin/env python3
"""
This script processes two files from the alexnet_split experiment:
1. Extracts class names from the 'name' file
2. Extracts predictions from the 'prediction.csv' file
"""

import re
import csv

# File paths
name_file_path = "results/alexnet_split/without HE/name"
prediction_file_path = "results/alexnet_split/without HE/prediction.csv"

def extract_class_names(file_path):
    """
    Extract class names from the name file.
    For each line, get the word after the numbers and before JPEG.
    """
    class_names = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Using regex to extract the part between the last underscore and .JPEG
            match = re.search(r'_([^_]+)\.JPEG', line)
            if match:
                class_name = match.group(1)
                class_names.append(class_name)
    
    return class_names

def extract_predictions(file_path):
    """
    Extract predictions from the prediction.csv file.
    For each line, get the second item (between first and second comma).
    """
    predictions = []
    
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        
        for row in csv_reader:
            if len(row) >= 2:
                predictions.append(row[1])  # Second column (index 1)
    
    return predictions

def accuracy(class_names, predictions):
    correct = 0
    for i in range(len(class_names)):
        if class_names[i] == predictions[i] or class_names[i] in predictions[i] or predictions[i] in class_names[i]:
            correct += 1
        else:
            with open("check_results.txt", "a") as f:
                f.write(f"\n{class_names[i]} | {predictions[i]}")
    return correct / len(class_names)

def main():

    # File paths
    name_file_path = "results/alexnet_split/with HE/name"
    prediction_file_path = "results/alexnet_split/with HE/prediction.csv"

    # Extract class names from name file
    class_names = extract_class_names(name_file_path)
    print(f"Extracted {len(class_names)} class names")
    
    # Extract predictions from prediction.csv file
    predictions = extract_predictions(prediction_file_path)
    print(f"Extracted {len(predictions)} predictions")
    
    # Print first 5 examples of each list
    print("\nFirst 5 class names:")
    print(class_names[:5])
    
    print("\nFirst 5 predictions:")
    print(predictions[:5])
    
    accuracy_result = accuracy(class_names, predictions)
    print(f"Accuracy: {accuracy_result}")
    
    """"# Optionally save the processed lists to new files
    with open("processed_class_names.txt", "w") as f:
        f.write(str(class_names))
    
    with open("processed_predictions.txt", "w") as f:
        f.write(str(predictions))
    
    print("\nProcessed lists have been saved to 'processed_class_names.txt' and 'processed_predictions.txt'")
    """

if __name__ == "__main__":
    main() 