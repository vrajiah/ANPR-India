#!/bin/bash

# ANPR System Deployment Script
# Creates a clean deployment package for fresh machine setup

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}"
    echo "🚗================================================🚗"
    echo "   ANPR System - Deployment Package Creator"
    echo "   Preparing for fresh machine setup"
    echo "🚗================================================🚗"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Main function
main() {
    print_header
    
    # Create deployment directory
    DEPLOY_DIR="anpr-system-deployment"
    print_info "Creating deployment package: $DEPLOY_DIR"
    
    if [ -d "$DEPLOY_DIR" ]; then
        print_warning "Deployment directory exists. Removing..."
        rm -rf "$DEPLOY_DIR"
    fi
    
    mkdir -p "$DEPLOY_DIR"
    
    # Copy essential files
    print_info "Copying essential files..."
    
    # Core package
    cp -r anpr_system "$DEPLOY_DIR/"
    
    # Configuration files
    cp setup.py "$DEPLOY_DIR/"
    cp pyproject.toml "$DEPLOY_DIR/"
    cp requirements.txt "$DEPLOY_DIR/"
    cp requirements-macos.txt "$DEPLOY_DIR/" 2>/dev/null || print_warning "macOS requirements not found"
    cp requirements-linux.txt "$DEPLOY_DIR/" 2>/dev/null || print_warning "Linux requirements not found"
    
    # Documentation
    cp README.md "$DEPLOY_DIR/"
    
    # Scripts
    cp install.sh "$DEPLOY_DIR/"
    chmod +x "$DEPLOY_DIR/install.sh"
    
    # YOLOv5 patches
    cp yolov5_pytorch_security_fix.patch "$DEPLOY_DIR/" 2>/dev/null || print_warning "YOLOv5 patch file not found"
    
    # Docker files
    cp Dockerfile "$DEPLOY_DIR/"
    cp docker-compose.yml "$DEPLOY_DIR/"
    
    # Supporting directories (production essentials only)
    cp -r runs "$DEPLOY_DIR/"
    cp -r utility_files "$DEPLOY_DIR/"
    
    # Note: notebooks/ and legacy/ excluded from production deployment
    # - notebooks/ are for development only
    # - legacy/ contains old script for reference only
    
    # Create archive
    print_info "Creating deployment archive..."
    tar -czf "anpr-system-deployment.tar.gz" "$DEPLOY_DIR"
    
    # Show results
    print_success "Deployment package created successfully!"
    echo ""
    print_info "Package contents:"
    ls -la "$DEPLOY_DIR"
    echo ""
    print_info "Archive created: anpr-system-deployment.tar.gz"
    print_info "Size: $(du -sh anpr-system-deployment.tar.gz | cut -f1)"
    echo ""
    print_info "To deploy on fresh machine:"
    echo "1. Transfer anpr-system-deployment.tar.gz to target machine"
    echo "2. Extract: tar -xzf anpr-system-deployment.tar.gz"
    echo "3. Follow QUICK_START.md (alongside .gz file) for setup instructions"
    echo ""
    print_success "🎉 Ready for deployment!"
}

# Run main function
main "$@"
