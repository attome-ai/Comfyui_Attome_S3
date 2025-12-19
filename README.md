# ComfyUI Attome S3 Nodes

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that enable seamless loading and saving of media files (images, videos, audio, text) directly from/to AWS S3 or S3-compatible storage services.

![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Nodes-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **S3 Configuration Node** - Centralized AWS credentials management
- **Image Operations** - Load/Save PNG, JPEG, WEBP with quality settings
- **Video Operations** - Load/Save MP4 with frame extraction options
- **Audio Operations** - Load/Save WAV, MP3, FLAC, OGG formats
- **Text Operations** - Load/Save text files with encoding support
- **S3-Compatible** - Works with AWS S3, Rustfs, MinIO, DigitalOcean Spaces, Backblaze B2, etc.

## Nodes

| Node | Description |
|------|-------------|
| **Attome S3 Config** | Configure AWS credentials, region, bucket, and optional endpoint URL |
| **Attome S3 Load Image** | Load image from S3, outputs IMAGE + MASK tensors |
| **Attome S3 Save Image** | Save image to S3 (PNG/JPEG/WEBP, configurable quality) |
| **Attome S3 Load Video** | Load video from S3, outputs frames as IMAGE batch + FPS |
| **Attome S3 Save Video** | Save image batch as video to S3 (configurable FPS/codec) |
| **Attome S3 Load Audio** | Load audio from S3, outputs AUDIO dict |
| **Attome S3 Save Audio** | Save audio to S3 (WAV/MP3/FLAC/OGG) |
| **Attome S3 Load Text** | Load text file from S3 |
| **Attome S3 Save Text** | Save text content to S3 |

## Installation


### Method 1: Portable ComfyUI (Windows)

```bash
cd ComfyUI_windows_portable\ComfyUI\custom_nodes
git clone https://github.com/attome-ai/Comfyui_Attome_S3.git
cd Comfyui_Attome_S3
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

## Requirements

- Python 3.10+
- ComfyUI
- boto3
- Pillow (usually pre-installed with ComfyUI)
- torch (usually pre-installed with ComfyUI)
- numpy (usually pre-installed with ComfyUI)
- torchaudio (for audio operations)
- opencv-python (for video operations)

## Usage

### Basic Workflow

1. Add **Attome S3 Config** node
2. Enter your AWS credentials:
   - `aws_access_key_id`: Your AWS access key
   - `aws_secret_access_key`: Your AWS secret key
   - `region_name`: AWS region (e.g., `us-east-1`)
   - `bucket_name`: Your S3 bucket name
   - `endpoint_url`: (Optional) Custom endpoint for S3-compatible services
3. Connect the `s3_config` output to any Load/Save node
4. Specify the S3 key (path) for your file

### Example: Load Image from S3

```
[Attome S3 Config] --> [Attome S3 Load Image] --> [Your Workflow]
                            |
                            +-- s3_key: "images/input.png"
```

### Example: Save Generated Image to S3

```
[Your Workflow] --> [Attome S3 Save Image] <-- [Attome S3 Config]
                            |
                            +-- s3_key: "outputs/result.png"
                            +-- format: "PNG"
                            +-- quality: 95
```

### Using with S3-Compatible Services

For MinIO, DigitalOcean Spaces, Backblaze B2, or other S3-compatible services, set the `endpoint_url` parameter:

| Service | Endpoint URL Example |
|---------|---------------------|
| MinIO | `http://localhost:9000` |
| DigitalOcean Spaces | `https://nyc3.digitaloceanspaces.com` |
| Backblaze B2 | `https://s3.us-west-000.backblazeb2.com` |
| Cloudflare R2 | `https://<account_id>.r2.cloudflarestorage.com` |

## Node Parameters

### Attome S3 Config

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| aws_access_key_id | STRING | "" | AWS access key ID |
| aws_secret_access_key | STRING | "" | AWS secret access key |
| region_name | STRING | "us-east-1" | AWS region |
| bucket_name | STRING | "" | S3 bucket name |
| endpoint_url | STRING | "" | (Optional) Custom S3 endpoint |

### Attome S3 Save Image

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| s3_key | STRING | "path/to/output.png" | S3 object key (path) |
| format | ENUM | "PNG" | Output format: PNG, JPEG, WEBP |
| quality | INT | 95 | Quality for JPEG/WEBP (1-100) |

### Attome S3 Load Video

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| s3_key | STRING | "path/to/video.mp4" | S3 object key (path) |
| frame_start | INT | 0 | Starting frame index |
| frame_count | INT | 0 | Number of frames (0 = all) |
| skip_frames | INT | 0 | Skip N frames between extractions |

### Attome S3 Save Video

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| s3_key | STRING | "path/to/output.mp4" | S3 object key (path) |
| fps | FLOAT | 24.0 | Frames per second |
| codec | ENUM | "mp4v" | Video codec: mp4v, avc1, XVID |

### Attome S3 Save Audio

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| s3_key | STRING | "path/to/output.wav" | S3 object key (path) |
| format | ENUM | "wav" | Output format: wav, mp3, flac, ogg |

## Security Best Practices

1. **Never commit credentials** - Use environment variables or AWS IAM roles
2. **Use least privilege** - Create an IAM user with only S3 permissions needed
3. **Enable bucket versioning** - Protect against accidental overwrites
4. **Use server-side encryption** - Enable SSE-S3 or SSE-KMS on your bucket

### Recommended IAM Policy

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject"
            ],
            "Resource": "arn:aws:s3:::your-bucket-name/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket"
            ],
            "Resource": "arn:aws:s3:::your-bucket-name"
        }
    ]
}
```

## Troubleshooting

### "No module named 'boto3'"

```bash
pip install boto3
```

### "No module named 'cv2'" (Video nodes)

```bash
pip install opencv-python
```

### "No module named 'torchaudio'" (Audio nodes)

```bash
pip install torchaudio
```

### "Access Denied" errors

1. Verify your AWS credentials are correct
2. Check the IAM user has the required S3 permissions
3. Ensure the bucket name and region are correct
4. For S3-compatible services, verify the endpoint URL

### Nodes not appearing in ComfyUI

1. Restart ComfyUI completely
2. Check the console for import errors
3. Verify all dependencies are installed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - The amazing node-based Stable Diffusion UI
- [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) - AWS SDK for Python

## Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Search [existing issues](https://github.com/attome-ai/ComfyUI-Attome-S3/issues)
3. Open a new issue with:
   - ComfyUI version
   - Python version
   - Full error message
   - Steps to reproduce
