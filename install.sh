#!/bin/bash

# ANPR System Installation Script
# One-click installation for non-technical users

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BLUE}"
    echo "🚗================================================🚗"
    echo "   ANPR System - Installation Script"
    echo "   Automatic Number Plate Recognition"
    echo "🚗================================================🚗"
    echo -e "${NC}"
}

print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --clean      Perform clean install (purge pip cache)"
    echo "  --uninstall  Remove ANPR System completely"
    echo "  --help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Normal installation"
    echo "  $0 --clean           # Clean installation (no cache)"
    echo "  $0 --uninstall       # Remove ANPR System"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Parse optional flags
CLEAN_INSTALL=false
UNINSTALL=false
PIP_NO_CACHE=""
for arg in "$@"; do
    if [ "$arg" = "--clean" ]; then
        CLEAN_INSTALL=true
        PIP_NO_CACHE="--no-cache-dir"
    elif [ "$arg" = "--uninstall" ]; then
        UNINSTALL=true
    elif [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
        print_help
        exit 0
    fi
done

# Uninstall ANPR System
uninstall_anpr() {
    print_info "Uninstalling ANPR System..."
    
    # Deactivate virtual environment if active
    if [ -n "$VIRTUAL_ENV" ]; then
        print_info "Deactivating virtual environment..."
        deactivate 2>/dev/null || true
    fi
    
    # Remove virtual environment
    if [ -d "anpr_env" ]; then
        print_info "Removing virtual environment..."
        rm -rf anpr_env
        print_success "Virtual environment removed"
    fi
    
    # Remove YOLOv5 directory
    if [ -d "yolov5" ]; then
        print_info "Removing YOLOv5 directory..."
        rm -rf yolov5
        print_success "YOLOv5 directory removed"
    fi
    
    # Remove activation script
    if [ -f "activate_anpr.sh" ]; then
        print_info "Removing activation script..."
        rm -f activate_anpr.sh
        print_success "Activation script removed"
    fi
    
    # Remove desktop shortcut
    if [ -f ~/Desktop/ANPR-System.desktop ]; then
        print_info "Removing desktop shortcut..."
        rm -f ~/Desktop/ANPR-System.desktop
        print_success "Desktop shortcut removed"
    fi
    
    # Uninstall package if installed
    if pip list | grep -q "anpr-system"; then
        print_info "Uninstalling ANPR package..."
        pip uninstall anpr-system -y 2>/dev/null || true
        print_success "ANPR package uninstalled"
    fi
    
    print_success "🎉 ANPR System uninstalled successfully!"
    print_info "You can now run './install.sh' to reinstall"
}

# Check system requirements
check_requirements() {
    print_info "Checking system requirements..."
    
    # Check Python
    if ! command_exists python3; then
        print_error "Python 3 is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    
    # Check Python version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    major=$(echo $python_version | cut -d. -f1)
    minor=$(echo $python_version | cut -d. -f2)
    
    if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 8 ]); then
        print_error "Python 3.8+ is required. Found: $python_version"
        exit 1
    fi
    
    print_success "Python $python_version found"
    
    # Check pip
    if ! command_exists pip3; then
        print_error "pip3 is not installed. Please install pip first."
        exit 1
    fi
    
    print_success "pip3 found"
    
    # Check git
    if ! command_exists git; then
        print_error "Git is not installed. Please install Git first."
        exit 1
    fi
    
    print_success "Git found"
}

# Install system dependencies (Ubuntu/Debian)
install_system_deps() {
    if command_exists apt-get; then
        print_info "Installing system dependencies..."
        sudo apt-get update
        sudo apt-get install -y \
            python3-venv \
            python3-dev \
            build-essential \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender-dev \
            libgomp1 \
            libgl1-mesa-glx \
            wget \
            curl
        print_success "System dependencies installed"
    else
        print_warning "Skipping system dependencies (not Ubuntu/Debian)"
    fi
}

# Create virtual environment
create_venv() {
    print_info "Creating virtual environment..."
    
    if [ -d "anpr_env" ]; then
        print_warning "Virtual environment already exists. Removing..."
        rm -rf anpr_env
    fi
    
    python3 -m venv anpr_env
    source anpr_env/bin/activate
    
    print_success "Virtual environment created"
}

# Install Python dependencies
install_python_deps() {
    print_info "Installing Python dependencies..."
    
    # Optional clean install
    if $CLEAN_INSTALL; then
        print_info "Performing clean install (purging pip cache)..."
        pip cache purge || true
    fi
    
    # Upgrade pip
    pip install $PIP_NO_CACHE --upgrade pip
    
    # Detect platform and use appropriate requirements file
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        print_info "Detected macOS - using CPU-only requirements (MPS acceleration will be detected during installation)"
        pip install $PIP_NO_CACHE -r requirements-macos.txt
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        print_info "Detected Linux - using GPU-enabled requirements"
        pip install $PIP_NO_CACHE -r requirements-linux.txt
    else
        # Fallback to generic requirements
        print_info "Using generic requirements"
        pip install $PIP_NO_CACHE -r requirements.txt
    fi
    
    # Install YOLOv5 requirements to prevent auto-updates during runtime
    print_info "Installing YOLOv5 requirements..."
    pip install $PIP_NO_CACHE "protobuf<4.21.3" ipython || true
    
    # Install pillow with correct constraint first (before streamlit)
    pip install $PIP_NO_CACHE "pillow>=7.1.0,<12.0.0"
    
    # Install additional web dependencies
    pip install $PIP_NO_CACHE streamlit streamlit-option-menu plotly
    
    # Force pillow to correct version after streamlit (in case it upgraded)
    pip install $PIP_NO_CACHE --force-reinstall "pillow>=7.1.0,<12.0.0" 2>/dev/null || pip install $PIP_NO_CACHE "pillow>=7.1.0,<12.0.0"
    
    # Install ANPR package in development mode
    print_info "Installing ANPR package..."
    if [ -f "setup.py" ]; then
        # Force reinstall to ensure entry points are properly generated
        pip install $PIP_NO_CACHE -e . --force-reinstall
        print_success "ANPR package installed"
    else
        print_warning "setup.py not found - skipping package installation"
        print_info "You may need to install the package manually later"
    fi
    
    print_success "Python dependencies installed"
}

# Clone YOLOv5
clone_yolov5() {
    print_info "Cloning YOLOv5 framework..."
    
    if [ -d "yolov5" ]; then
        print_warning "YOLOv5 already exists. Removing..."
        rm -rf yolov5
    fi
    
    git clone https://github.com/rkuo2000/yolov5.git
    print_success "YOLOv5 cloned"
    
    # Apply PyTorch security fixes
    print_info "Applying YOLOv5 PyTorch security fixes..."
    if [ -f "yolov5_pytorch_security_fix.patch" ]; then
        cd yolov5
        if git apply --check ../yolov5_pytorch_security_fix.patch 2>/dev/null; then
            git apply ../yolov5_pytorch_security_fix.patch
            print_success "YOLOv5 PyTorch security fixes applied successfully"
        else
            print_warning "YOLOv5 patches already applied or incompatible"
        fi
        cd ..
    else
        print_warning "YOLOv5 patch file not found, skipping security fixes"
        print_info "Note: You may need to manually apply PyTorch security fixes"
    fi
}

# Create directories
create_directories() {
    print_info "Creating necessary directories..."
    
    mkdir -p test_data results
    print_success "Directories created"
}

# Test installation
test_installation() {
    print_info "Testing installation..."
    
    # Test imports
    python3 -c "
import torch
import cv2
import pandas
from paddleocr import PaddleOCR
import norfair
print('✅ All dependencies imported successfully')
"
    
    # Test MPS support on Apple Silicon
    python3 -c "
import torch
if torch.backends.mps.is_available():
    print('✅ MPS (Metal) acceleration available for Apple Silicon')
    print('   YOLOv5 detection will use hardware acceleration')
elif torch.cuda.is_available():
    print('✅ CUDA acceleration available')
    print('   YOLOv5 detection will use GPU acceleration')
else:
    print('ℹ️  Using CPU for YOLOv5 detection')
    print('   Consider upgrading PyTorch for hardware acceleration')
"
    
    # Test ANPR system
    python3 -c "
try:
    from anpr_system import ANPRSystem
    print('✅ ANPR System imported successfully')
except ImportError:
    print('⚠️  ANPR System package not found - this is normal if setup.py is missing')
    print('   You can install it manually later with: pip install -e .')
"
    
    print_success "Installation test passed"
}

# Create activation script
create_activation_script() {
    print_info "Creating activation script..."
    
    cat > activate_anpr.sh << 'EOF'
#!/bin/bash
# ANPR System Activation Script

echo "🚗 Activating ANPR System..."
source anpr_env/bin/activate

echo "✅ ANPR System activated!"
echo ""
echo "Available commands:"
echo "  anpr --help              # CLI help"
echo "  anpr-web                 # Web interface"
echo "  python anpr-system.py    # Direct script"
echo ""
echo "Hardware acceleration:"
echo "  anpr --device auto       # Auto-detect (MPS/CUDA/CPU)"
echo "  anpr --device mps        # Force MPS (Apple Silicon)"
echo "  anpr --device cuda       # Force CUDA (Linux)"
echo "  anpr --device cpu        # Force CPU"
echo ""
echo "To deactivate: deactivate"
EOF
    
    chmod +x activate_anpr.sh
    print_success "Activation script created"
}

# Create desktop shortcut (Linux)
create_desktop_shortcut() {
    if [ -n "$XDG_CURRENT_DESKTOP" ]; then
        print_info "Creating desktop shortcut..."
        
        cat > ~/Desktop/ANPR-System.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=ANPR System
Comment=Automatic Number Plate Recognition
Exec=$PWD/activate_anpr.sh
Icon=$PWD/utility_files/anpr_logo.jpg
Terminal=true
Categories=Development;
EOF
        
        chmod +x ~/Desktop/ANPR-System.desktop
        print_success "Desktop shortcut created"
    fi
}

# Main installation function
main() {
    print_header
    
    # Handle uninstall
    if $UNINSTALL; then
        uninstall_anpr
        exit 0
    fi
    
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        print_error "Please do not run this script as root"
        exit 1
    fi
    
    # Check requirements
    check_requirements
    
    # Install system dependencies
    install_system_deps
    
    # Create virtual environment
    create_venv
    
    # Install Python dependencies
    install_python_deps
    
    # Clone YOLOv5
    clone_yolov5
    
    # Create directories
    create_directories
    
    # Test installation
    test_installation
    
    # Create activation script
    create_activation_script
    
    # Create desktop shortcut
    create_desktop_shortcut
    
    # Final message
    echo ""
    print_success "🎉 ANPR System installed successfully!"
    echo ""
    print_info "To start using ANPR System:"
    echo "  1. Run: source activate_anpr.sh"
    echo "  2. Or: source anpr_env/bin/activate"
    echo ""
    print_info "Available commands:"
    echo "  anpr --help              # CLI help"
    echo "  anpr-web                 # Web interface"
    echo "  python anpr-system.py    # Direct script"
    echo ""
    print_info "Web interface will be available at: http://localhost:8501"
    echo ""
    print_warning "Note: First run may take longer due to model downloads"
}

# Run main function
main "$@"
