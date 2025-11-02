# 🚀 ANPR System - Quick Start Guide

Get up and running with the ANPR system in under 5 minutes!

## 📦 Deployment Package Setup

### Step 1: Extract the Package
```bash
# Extract the deployment package
tar -xzf anpr-system-deployment.tar.gz
cd anpr-system-deployment
```

### Step 2: One-Click Installation
```bash
# Run the automated installer
chmod +x install.sh
./install.sh

# Activate the environment (created by installer)
source anpr_env/bin/activate  
# or source activate_anpr.sh
```

### Step 3: Test Installation
```bash
# Test CLI
anpr --help

# Test web interface
anpr-web
# Open http://localhost:8501 in your browser
```

## 🎯 Quick Test

### Process a Video
```bash
# Basic video processing
anpr --input your_video.mp4 --output result.mp4 --csv plates.csv

# With custom settings
anpr --input your_video.mp4 \
     --output result.mp4 \
     --csv plates.csv \
     --frame-skip 5 \
     --min-conf 0.5
```

### Launch Web Interface
```bash
# Start the web application
anpr-web

# Open http://localhost:8501 in your browser
```

## 🐳 Docker Alternative

```bash
# Using Docker Compose
docker-compose up web

# Or run CLI directly
docker-compose run cli anpr --input video.mp4 --output result.mp4
```

## 📊 What You'll Get

- **Output Video**: Annotated video with bounding boxes around detected plates
- **CSV File**: List of detected license plates with confidence scores
- **Real-time Processing**: Live feedback during processing

## 🆘 Need Help?

- **Full Documentation**: See README.md (included in package)
- **Issues**: [GitHub Issues](https://github.com/Tkvmaster/ANPR-System/issues)

## 🎉 Success!

If you see the help message when running `anpr --help`, you're all set! The ANPR system is ready to process your videos and images.