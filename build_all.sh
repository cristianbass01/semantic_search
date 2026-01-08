#!/bin/bash

# Define the base directory as current directory
BASE_DIR="."
PROJECT_NAME="search"
VERSION="0.0.11"

eval $(minikube docker-env)

# Loop through each subdirectory
for dir in "$BASE_DIR"/*; do
    if [ -d "$dir" ]; then
        echo "Checking for Dockerfile in $dir"
        
        # Check if Dockerfile exists
        if [ -f "$dir/Dockerfile" ]; then
            echo "Building and pushing Docker image in $dir"
            
            # Build the Docker image with the specified tag format
            IMAGE_NAME="cristianbassotto/${PROJECT_NAME}-$(basename "$dir" | sed 's/_/-/g'):${VERSION}"
            docker build -f $dir/Dockerfile -t "$IMAGE_NAME" .
            
            # Push the Docker image
            # docker push "$IMAGE_NAME"
        else
            echo "No Dockerfile found in $dir, skipping."
        fi
    fi
done