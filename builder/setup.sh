#!/bin/bash

# Update package lists
apt-get update

# Install espeak-ng
apt-get install -y espeak-ng

# Install ffmpeg
apt-get install -y ffmpeg

# Clean up
apt-get clean
rm -rf /var/lib/apt/lists/*
