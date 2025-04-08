#!/bin/bash

# Script to prepare and train YOLOv5 with URL-based dataset
# 
# Prerequisites:
# 1. Clone YOLOv5 repository: git clone https://github.com/ultralytics/yolov5.git
# 2. Install dependencies: pip install -r yolov5/requirements.txt
# 3. Put your annotation files in a directory
#
# Usage:
# ./train_yolo_url.sh --imagesets "145 146 147" --annotations-dir /path/to/annotations --output-dir datasets/url_dataset

# Default values
IMAGESETS=""
ANNOTATIONS_DIR=""
OUTPUT_DIR="datasets/url_dataset"
VERIFY_URLS=false
TRAIN_BATCH=8
TRAIN_EPOCHS=50
LOCAL_MODE=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --imagesets)
      IMAGESETS="$2"
      shift 2
      ;;
    --annotations-dir)
      ANNOTATIONS_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --verify-urls)
      VERIFY_URLS=true
      shift
      ;;
    --batch)
      TRAIN_BATCH="$2"
      shift 2
      ;;
    --epochs)
      TRAIN_EPOCHS="$2"
      shift 2
      ;;
    --local)
      LOCAL_MODE=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check required arguments
if [ -z "$IMAGESETS" ]; then
  echo "Error: --imagesets is required"
  exit 1
fi

if [ -z "$ANNOTATIONS_DIR" ]; then
  echo "Error: --annotations-dir is required"
  exit 1
fi

# Ensure YOLOv5 repository exists
if [ ! -d "yolov5" ]; then
  echo "YOLOv5 repository not found, cloning..."
  git clone https://github.com/ultralytics/yolov5.git
fi

# Create URL-based dataset
echo "Creating URL-based dataset..."
python url-yolo-converter.py --imagesets $IMAGESETS --annotations-dir "$ANNOTATIONS_DIR" --output-dir "$OUTPUT_DIR" $([ "$VERIFY_URLS" = true ] && echo "--verify-urls")

if [ $? -ne 0 ]; then
  echo "Error creating dataset"
  exit 1
fi

# Create custom model config
echo "Creating custom model config..."
python yolo-dataset-utils.py --command model_config --output-dir "$OUTPUT_DIR" --num-classes 2

if [ "$LOCAL_MODE" = true ]; then
  # Create local version of dataset
  LOCAL_DIR="${OUTPUT_DIR}_local"
  echo "Creating local dataset at $LOCAL_DIR..."
  python yolo-dataset-utils.py --command local_yaml --url-dataset "$OUTPUT_DIR" --output-dir "$LOCAL_DIR"
  
  # Download a few test images
  echo "Downloading a few test images..."
  python yolo-dataset-utils.py --command download --url-file "$OUTPUT_DIR/train.txt" --output-dir "$LOCAL_DIR/images/train" --max-images 10
  python yolo-dataset-utils.py --command download --url-file "$OUTPUT_DIR/val.txt" --output-dir "$LOCAL_DIR/images/val" --max-images 10
  
  echo "Local test dataset created at $LOCAL_DIR"
  echo "You can train with this small local dataset using:"
  echo "cd yolov5 && python train.py --img 640 --batch $TRAIN_BATCH --epochs $TRAIN_EPOCHS --data ../$LOCAL_DIR/dataset.yaml --cfg ../$OUTPUT_DIR/yolov5s_custom.yaml --weights yolov5s.pt"
  exit 0
fi

# Train YOLOv5 with URL-based dataset
echo "Starting training..."
cd yolov5
python train.py --img 640 --batch $TRAIN_BATCH --epochs $TRAIN_EPOCHS --data ../$OUTPUT_DIR/dataset.yaml --cfg ../$OUTPUT_DIR/yolov5s_custom.yaml --weights yolov5s.pt

echo "Training complete! Results are in yolov5/runs/train/"