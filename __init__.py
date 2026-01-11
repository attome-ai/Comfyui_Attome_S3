"""
Attome S3 - ComfyUI Custom Nodes for AWS S3 Integration

This package provides nodes for loading and saving various media types
(text, image, audio, video) to and from AWS S3 storage.

Nodes:
    - AttomeS3Config: Configure S3 credentials and bucket
    - AttomeS3LoadText / AttomeS3SaveText: Text file operations
    - AttomeS3LoadImage / AttomeS3SaveImage: Image operations (PNG, JPEG, WEBP)
    - AttomeS3LoadAudio / AttomeS3SaveAudio: Audio operations (WAV, MP3, FLAC, OGG)
    - AttomeS3LoadVideo / AttomeS3SaveVideo: Video operations (MP4)

Requirements:
    - boto3
    - Pillow
    - torch
    - numpy
    - torchaudio (for audio operations)
    - opencv-python (for video operations)

Installation:
    1. Clone or copy this folder to ComfyUI/custom_nodes/attome_s3/
    2. Install requirements: pip install boto3 torchaudio opencv-python
    3. Restart ComfyUI

Usage:
    1. Add "Attome S3 Config" node and enter your AWS credentials
    2. Connect the s3_config output to any Load/Save node
    3. Specify the S3 key (path) for the file you want to load/save
"""

from .nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Version info
__version__ = "1.0.1"
__author__ = "Attome"
