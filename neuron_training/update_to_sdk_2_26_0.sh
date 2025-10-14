#!/bin/bash
# AWS Neuron SDK 2.26.0 Update Script
# Release Date: 09/18/2025
# Only updates packages that are already installed

set -e

echo "Updating to AWS Neuron SDK 2.26.0 (09/18/2025)"
echo "This script only updates packages that are already installed."
echo ""

# Update package lists
echo "Updating package lists..."
sudo apt-get update

# System packages - only upgrade what's already installed
echo "Updating system packages (if installed)..."
sudo apt install --only-upgrade -y --allow-downgrades --allow-change-held-packages \
  aws-neuronx-collectives=2.28.27.0* \
  aws-neuronx-dkms=2.24.7.0* \
  aws-neuronx-gpsimd-customop-lib=0.18.0.0* \
  aws-neuronx-gpsimd-tools=0.18.0.0* \
  aws-neuronx-k8-plugin=2.28.4.0* \
  aws-neuronx-k8-scheduler=2.28.4.0* \
  aws-neuronx-oci-hook=2.12.36.0* \
  aws-neuronx-runtime-lib=2.28.23.0* \
  aws-neuronx-tools=2.26.14.0*

# Python packages - only upgrade packages that are already installed
echo "Updating Python packages (if installed)..."

# Check which packages are installed and upgrade only those
declare -a packages=(
  "aws-neuronx-runtime-discovery==2.9"
  "libneuronxla==2.2.12677.0"
  "neuronx-cc==2.21.18209.0"
  "neuronx-distributed==0.15.22404"
  "neuronx-distributed-training==1.6.0"
  "neuronx-distributed-inference==0.6.10598"
  "tensorboard-plugin-neuronx==2.0.837.0"
  "tensorflow-neuronx==2.10.1.2.1.0"
  "torch-neuronx==2.8.0.2.10.13553"
  "transformers-neuronx==0.13.1315"
  "jax-neuronx==0.6.2.1.0.6446"
)

for package_spec in "${packages[@]}"; do
  package_name="${package_spec%%==*}"
  # Check if package is installed (pip show returns 0 if installed)
  if pip show "$package_name" >/dev/null 2>&1; then
    echo "Upgrading $package_name..."
    pip install --upgrade --extra-index-url=https://pip.repos.neuron.amazonaws.com "$package_spec"
  fi
done

echo ""
echo "AWS Neuron SDK 2.26.0 update complete!"
echo ""
echo "Important notes for SDK 2.26.0:"
echo "• End-of-support for Transformers NeuronX library"
echo "• PyTorch 2.8, JAX 0.6.2, and Python 3.11 support added"
echo "• Driver versions >2.21 only support non-Inf1 instances"
echo "• For Inf1 users: use Neuron driver 2.21 or below"
echo ""
echo "To verify your installation:"
echo "  python3 neuron_detector.py --verbose"
echo ""

# Uncomment the sections below for FRESH INSTALLATION (installs ALL packages):

# echo "Fresh installation - installing all SDK 2.26.0 system packages..."
# sudo apt install -y \
#   aws-neuronx-collectives=2.28.27.0* \
#   aws-neuronx-dkms=2.24.7.0* \
#   aws-neuronx-gpsimd-customop-lib=0.18.0.0* \
#   aws-neuronx-gpsimd-tools=0.18.0.0* \
#   aws-neuronx-k8-plugin=2.28.4.0* \
#   aws-neuronx-k8-scheduler=2.28.4.0* \
#   aws-neuronx-oci-hook=2.12.36.0* \
#   aws-neuronx-runtime-lib=2.28.23.0* \
#   aws-neuronx-tools=2.26.14.0*

# echo "Fresh installation - installing all SDK 2.26.0 Python packages..."
# pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com \
#   aws-neuronx-runtime-discovery==2.9 \
#   libneuronxla==2.2.12677.0 \
#   neuronx-cc==2.21.18209.0 \
#   neuronx-distributed==0.15.22404 \
#   neuronx-distributed-training==1.6.0 \
#   neuronx-distributed-inference==0.6.10598 \
#   tensorboard-plugin-neuronx==2.0.837.0 \
#   tensorflow-neuronx==2.10.1.2.1.0 \
#   torch-neuronx==2.8.0.2.10.13553 \
#   transformers-neuronx==0.13.1315 \
#   jax-neuronx==0.6.2.1.0.6446