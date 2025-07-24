#!/usr/bin/env python3
"""
Build script for PDF outline extraction system.

This script handles the complete build process:
1. Model preparation and quantization
2. Training MLP classifier
3. Validation of model outputs
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"   Error: {e.stderr.strip()}")
        return False


def check_requirements():
    """Check if all required directories and files exist."""
    print("🔍 Checking requirements...")
    
    required_dirs = ['src', 'proto_labels', 'models']
    required_files = [
        'src/prepare_models.py',
        'src/pdf_outline.py',
        'src/validate_output.py',
        'requirements.txt',
        'Dockerfile'
    ]
    
    missing = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            print(f"📁 Created directory: {dir_name}")
    
    for file_name in required_files:
        if not os.path.exists(file_name):
            missing.append(file_name)
    
    if missing:
        print(f"❌ Missing required files: {missing}")
        return False
    
    print("✅ All requirements satisfied")
    return True


def setup_environment():
    """Set up Python environment and install dependencies."""
    print("🔧 Setting up environment...")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if not in_venv:
        print("⚠️  Not in a virtual environment. Consider using venv or conda.")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    return True


def prepare_models():
    """Prepare and train models."""
    print("🤖 Preparing models...")
    
    # Ensure proto_labels directory has some content
    proto_labels_dir = "proto_labels"
    if not os.listdir(proto_labels_dir):
        print("⚠️  No prototype labels found. The system will use synthetic data.")
    
    # Run model preparation
    cmd = "python src/prepare_models.py --model-dir models --proto-labels proto_labels"
    if not run_command(cmd, "Preparing and training models"):
        return False
    
    # Check model outputs
    expected_files = [
        "models/minilm_quantized",
        "models/mlp_head.pt",
        "models/model_metadata.json"
    ]
    
    missing_models = []
    for model_file in expected_files:
        if not os.path.exists(model_file):
            missing_models.append(model_file)
    
    if missing_models:
        print(f"❌ Missing model files: {missing_models}")
        return False
    
    # Check model size
    total_size = 0
    for root, dirs, files in os.walk("models"):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    
    total_size_mb = total_size / (1024 * 1024)
    print(f"📊 Total model size: {total_size_mb:.1f} MB")
    
    if total_size_mb > 200:
        print("⚠️  Warning: Model size exceeds 200MB limit")
    else:
        print("✅ Model size within limits")
    
    return True


def test_extraction():
    """Test PDF extraction on sample files."""
    print("🧪 Testing PDF extraction...")
    
    # Create test directories
    test_input = "test_input"
    test_output = "test_output"
    os.makedirs(test_input, exist_ok=True)
    os.makedirs(test_output, exist_ok=True)
    
    # Check if we have sample PDFs
    sample_pdfs = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    if not sample_pdfs:
        print("⚠️  No sample PDFs found for testing")
        return True
    
    # Copy a sample PDF to test directory
    import shutil
    test_pdf = sample_pdfs[0]
    shutil.copy(test_pdf, os.path.join(test_input, test_pdf))
    
    # Run extraction
    cmd = f"python src/pdf_outline.py --input {test_input} --output {test_output} --models models"
    if not run_command(cmd, f"Testing extraction on {test_pdf}"):
        return False
    
    # Check output
    expected_output = os.path.join(test_output, test_pdf.replace('.pdf', '.json'))
    if os.path.exists(expected_output):
        print(f"✅ Test extraction successful: {expected_output}")
        
        # Show sample output
        try:
            import json
            with open(expected_output, 'r') as f:
                result = json.load(f)
            print(f"   Title: {result.get('title', 'N/A')}")
            print(f"   Headings: {len(result.get('outline', []))}")
        except Exception as e:
            print(f"   Warning: Could not parse output JSON: {e}")
        
        return True
    else:
        print("❌ Test extraction failed - no output generated")
        return False


def build_docker():
    """Build Docker image."""
    print("🐳 Building Docker image...")
    
    # Build image
    cmd = "docker build --platform linux/amd64 -t pdf-outline-extractor:latest ."
    if not run_command(cmd, "Building Docker image"):
        return False
    
    # Check image size
    cmd = "docker images pdf-outline-extractor:latest --format 'table {{.Size}}'"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            size_line = result.stdout.strip().split('\n')[-1]
            print(f"📊 Docker image size: {size_line}")
    except:
        print("⚠️  Could not determine Docker image size")
    
    return True


def validate_setup():
    """Validate the complete setup."""
    print("✅ Validating setup...")
    
    validation_checks = [
        ("Models directory exists", os.path.exists("models")),
        ("MLP model exists", os.path.exists("models/mlp_head.pt")),
        ("Quantized MiniLM exists", os.path.exists("models/minilm_quantized")),
        ("Metadata exists", os.path.exists("models/model_metadata.json")),
        ("Main script exists", os.path.exists("src/pdf_outline.py")),
        ("Validation script exists", os.path.exists("src/validate_output.py")),
        ("Dockerfile exists", os.path.exists("Dockerfile")),
    ]
    
    all_passed = True
    for check_name, check_result in validation_checks:
        status = "✅" if check_result else "❌"
        print(f"   {status} {check_name}")
        if not check_result:
            all_passed = False
    
    return all_passed


def main():
    """Main build process."""
    print("🚀 Starting PDF Outline Extractor build process...")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Check requirements
    if not check_requirements():
        print("❌ Build failed: Missing requirements")
        sys.exit(1)
    
    # Step 2: Setup environment
    if not setup_environment():
        print("❌ Build failed: Environment setup failed")
        sys.exit(1)
    
    # Step 3: Prepare models
    if not prepare_models():
        print("❌ Build failed: Model preparation failed")
        sys.exit(1)
    
    # Step 4: Test extraction
    if not test_extraction():
        print("❌ Build failed: Extraction test failed")
        sys.exit(1)
    
    # Step 5: Build Docker (optional)
    docker_available = subprocess.run("docker --version", shell=True, capture_output=True).returncode == 0
    if docker_available:
        if not build_docker():
            print("⚠️  Docker build failed, but continuing...")
    else:
        print("⚠️  Docker not available, skipping Docker build")
    
    # Step 6: Final validation
    if not validate_setup():
        print("❌ Build failed: Validation failed")
        sys.exit(1)
    
    # Success
    build_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("🎉 BUILD SUCCESSFUL!")
    print(f"⏱️  Total build time: {build_time:.1f} seconds")
    print("\n📋 Next steps:")
    print("1. Test with your PDFs: python src/pdf_outline.py --single your_file.pdf")
    print("2. Validate outputs: python src/validate_output.py --predicted output --ground-truth ground_truth")
    print("3. Run via Docker: docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output pdf-outline-extractor")
    print("\n✨ Happy PDF processing!")


if __name__ == "__main__":
    main()