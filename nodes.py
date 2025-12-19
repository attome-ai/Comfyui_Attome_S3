"""
Attome S3 Nodes for ComfyUI
Load and save text, video, image, and audio resources from/to AWS S3
"""

import os
import io
import tempfile
import numpy as np
from PIL import Image
import torch
import boto3
from botocore.exceptions import ClientError, NoCredentialsError


# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

def load_env_config():
    """Load default S3 configuration from env.txt file."""
    env_path = os.path.join(os.path.dirname(__file__), "env.txt")
    config = {
        "aws_access_key_id": "",
        "aws_secret_access_key": "",
        "region_name": "us-east-1",
        "bucket_name": "",
        "endpoint_url": "",
    }
    
    if os.path.exists(env_path):
        try:
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith("#"):
                        continue
                    
                    # Parse KEY=VALUE format
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if key == "AWS_ACCESS_KEY_ID":
                            config["aws_access_key_id"] = value
                        elif key == "AWS_SECRET_ACCESS_KEY":
                            config["aws_secret_access_key"] = value
                        elif key == "REGION_NAME":
                            config["region_name"] = value
                        elif key == "BUCKET_NAME":
                            config["bucket_name"] = value
                        elif key == "ENDPOINT_URL":
                            config["endpoint_url"] = value
        except Exception as e:
            print(f"Warning: Failed to load env.txt: {e}")
    
    return config


# ============================================================================
# S3 CONFIG NODE
# ============================================================================

class AttomeS3Config:
    """
    S3 Configuration node - provides AWS S3 credentials and settings
    to other Attome S3 nodes.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # Load defaults from env.txt if available
        env_defaults = load_env_config()
        
        return {
            "required": {
                "aws_access_key_id": ("STRING", {
                    "default": env_defaults["aws_access_key_id"],
                    "multiline": False,
                }),
                "aws_secret_access_key": ("STRING", {
                    "default": env_defaults["aws_secret_access_key"],
                    "multiline": False,
                }),
                "region_name": ("STRING", {
                    "default": env_defaults["region_name"],
                    "multiline": False,
                }),
                "bucket_name": ("STRING", {
                    "default": env_defaults["bucket_name"],
                    "multiline": False,
                }),
            },
            "optional": {
                "endpoint_url": ("STRING", {
                    "default": env_defaults["endpoint_url"],
                    "multiline": False,
                }),
            }
        }

    RETURN_TYPES = ("S3_CONFIG",)
    RETURN_NAMES = ("s3_config",)
    FUNCTION = "create_config"
    CATEGORY = "Attome/S3"

    def create_config(self, aws_access_key_id, aws_secret_access_key,
                      region_name, bucket_name, endpoint_url=""):
        config = {
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "region_name": region_name,
            "bucket_name": bucket_name,
            "endpoint_url": endpoint_url if endpoint_url else None,
        }
        return (config,)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_s3_client(s3_config):
    """Create and return an S3 client from config."""
    client_kwargs = {
        "aws_access_key_id": s3_config["aws_access_key_id"],
        "aws_secret_access_key": s3_config["aws_secret_access_key"],
        "region_name": s3_config["region_name"],
    }
    if s3_config.get("endpoint_url"):
        client_kwargs["endpoint_url"] = s3_config["endpoint_url"]

    return boto3.client("s3", **client_kwargs)


def validate_s3_key(s3_key):
    """Check if s3_key is valid (not empty or whitespace)."""
    return s3_key and s3_key.strip() != ""


def download_from_s3(s3_config, s3_key):
    """Download a file from S3 and return its bytes."""
    client = get_s3_client(s3_config)
    bucket = s3_config["bucket_name"]

    response = client.get_object(Bucket=bucket, Key=s3_key)
    return response["Body"].read()


def upload_to_s3(s3_config, s3_key, data, content_type=None):
    """Upload data to S3."""
    client = get_s3_client(s3_config)
    bucket = s3_config["bucket_name"]

    extra_args = {}
    if content_type:
        extra_args["ContentType"] = content_type

    if isinstance(data, str):
        data = data.encode("utf-8")

    client.put_object(Bucket=bucket, Key=s3_key, Body=data, **extra_args)
    return f"s3://{bucket}/{s3_key}"


# ============================================================================
# TEXT NODES
# ============================================================================

class AttomeS3LoadText:
    """Load text content from S3."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "s3_config": ("S3_CONFIG",),
                "s3_key": ("STRING", {
                    "default": "path/to/file.txt",
                    "multiline": False,
                }),
            },
            "optional": {
                "encoding": ("STRING", {
                    "default": "utf-8",
                    "multiline": False,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "load_text"
    CATEGORY = "Attome/S3"

    def load_text(self, s3_config, s3_key, encoding="utf-8"):
        # Skip S3 loading if s3_key is empty
        if not validate_s3_key(s3_key):
            return ("",)
        
        data = download_from_s3(s3_config, s3_key)
        text = data.decode(encoding)
        return (text,)


class AttomeS3SaveText:
    """Save text content to S3."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "s3_config": ("S3_CONFIG",),
                "text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                }),
                "s3_key": ("STRING", {
                    "default": "path/to/output.txt",
                    "multiline": False,
                }),
            },
            "optional": {
                "encoding": ("STRING", {
                    "default": "utf-8",
                    "multiline": False,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("s3_uri",)
    FUNCTION = "save_text"
    CATEGORY = "Attome/S3"
    OUTPUT_NODE = True

    def save_text(self, s3_config, text, s3_key, encoding="utf-8"):
        data = text.encode(encoding)
        uri = upload_to_s3(s3_config, s3_key, data, content_type="text/plain")
        return (uri,)


# ============================================================================
# IMAGE NODES
# ============================================================================

class AttomeS3LoadImage:
    """Load image from S3. Returns ComfyUI-compatible image tensor."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "s3_config": ("S3_CONFIG",),
                "s3_key": ("STRING", {
                    "default": "path/to/image.png",
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image"
    CATEGORY = "Attome/S3"

    def load_image(self, s3_config, s3_key):
        # Skip S3 loading if s3_key is empty - return empty image
        if not validate_s3_key(s3_key):
            # Return a 512x512 black image and mask (proper size to avoid division errors)
            empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 512, 512), dtype=torch.float32)
            return (empty_image, empty_mask)
        
        data = download_from_s3(s3_config, s3_key)

        # Load image using PIL
        img = Image.open(io.BytesIO(data))

        # Handle different modes
        if img.mode == "I":
            img = img.point(lambda i: i * (1 / 255))

        has_alpha = "A" in img.mode

        # Convert to RGB/RGBA
        if img.mode not in ["RGB", "RGBA"]:
            img = img.convert("RGBA" if has_alpha else "RGB")

        # Convert to numpy array and normalize
        img_np = np.array(img).astype(np.float32) / 255.0

        # Create image tensor (ComfyUI format: BHWC)
        if has_alpha or img.mode == "RGBA":
            # Extract RGB and alpha separately
            image = torch.from_numpy(img_np[..., :3])[None,]
            mask = torch.from_numpy(img_np[..., 3])[None,]
            mask = 1.0 - mask  # Invert mask (ComfyUI convention)
        else:
            image = torch.from_numpy(img_np)[None,]
            mask = torch.zeros((1, img_np.shape[0], img_np.shape[1]),
                              dtype=torch.float32)

        return (image, mask)


class AttomeS3SaveImage:
    """Save image to S3."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "s3_config": ("S3_CONFIG",),
                "image": ("IMAGE",),
                "s3_key": ("STRING", {
                    "default": "path/to/output.png",
                    "multiline": False,
                }),
            },
            "optional": {
                "format": (["PNG", "JPEG", "WEBP"], {
                    "default": "PNG",
                }),
                "quality": ("INT", {
                    "default": 95,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("s3_uri",)
    FUNCTION = "save_image"
    CATEGORY = "Attome/S3"
    OUTPUT_NODE = True

    def save_image(self, s3_config, image, s3_key, format="PNG", quality=95):
        # Convert from ComfyUI tensor format (BHWC) to PIL
        # Take first image in batch
        img_np = image[0].cpu().numpy()
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

        img = Image.fromarray(img_np)

        # Save to bytes buffer
        buffer = io.BytesIO()
        save_kwargs = {}

        if format == "JPEG":
            img = img.convert("RGB")  # JPEG doesn't support alpha
            save_kwargs["quality"] = quality
            content_type = "image/jpeg"
        elif format == "WEBP":
            save_kwargs["quality"] = quality
            content_type = "image/webp"
        else:  # PNG
            content_type = "image/png"

        img.save(buffer, format=format, **save_kwargs)
        buffer.seek(0)

        uri = upload_to_s3(s3_config, s3_key, buffer.read(),
                          content_type=content_type)
        return (uri,)


# ============================================================================
# AUDIO NODES
# ============================================================================

class AttomeS3LoadAudio:
    """Load audio from S3. Returns raw audio bytes and metadata."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "s3_config": ("S3_CONFIG",),
                "s3_key": ("STRING", {
                    "default": "path/to/audio.wav",
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load_audio"
    CATEGORY = "Attome/S3"

    def load_audio(self, s3_config, s3_key):
        # Skip S3 loading if s3_key is empty - return empty audio
        if not validate_s3_key(s3_key):
            try:
                import torchaudio
                # Return empty waveform (1 channel, 1024 samples - proper size to avoid processing errors)
                empty_waveform = torch.zeros((1, 1, 1024), dtype=torch.float32)
                return ({"waveform": empty_waveform, "sample_rate": 44100},)
            except ImportError:
                return ({"waveform": b"", "sample_rate": 44100, "raw": True},)
        
        data = download_from_s3(s3_config, s3_key)

        # Create a temp file to work with audio libraries
        ext = os.path.splitext(s3_key)[1] or ".wav"

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            # Try to load with torchaudio if available
            import torchaudio
            waveform, sample_rate = torchaudio.load(tmp_path)
            audio = {
                "waveform": waveform.unsqueeze(0),  # Add batch dimension
                "sample_rate": sample_rate,
            }
        except ImportError:
            # Fallback: return raw bytes
            audio = {
                "waveform": data,
                "sample_rate": 44100,  # Default assumption
                "raw": True,
            }
        finally:
            os.unlink(tmp_path)

        return (audio,)


class AttomeS3SaveAudio:
    """Save audio to S3."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "s3_config": ("S3_CONFIG",),
                "audio": ("AUDIO",),
                "s3_key": ("STRING", {
                    "default": "path/to/output.wav",
                    "multiline": False,
                }),
            },
            "optional": {
                "format": (["wav", "mp3", "flac", "ogg"], {
                    "default": "wav",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("s3_uri",)
    FUNCTION = "save_audio"
    CATEGORY = "Attome/S3"
    OUTPUT_NODE = True

    def save_audio(self, s3_config, audio, s3_key, format="wav"):
        content_types = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "flac": "audio/flac",
            "ogg": "audio/ogg",
        }

        # Handle raw bytes
        if isinstance(audio, dict) and audio.get("raw"):
            data = audio["waveform"]
        else:
            # Use torchaudio to save
            import torchaudio

            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]

            # Remove batch dimension if present
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)

            with tempfile.NamedTemporaryFile(suffix=f".{format}",
                                             delete=False) as tmp:
                tmp_path = tmp.name

            try:
                torchaudio.save(tmp_path, waveform, sample_rate, format=format)
                with open(tmp_path, "rb") as f:
                    data = f.read()
            finally:
                os.unlink(tmp_path)

        uri = upload_to_s3(s3_config, s3_key, data,
                          content_type=content_types.get(format, "audio/wav"))
        return (uri,)


# ============================================================================
# VIDEO NODES
# ============================================================================

class AttomeS3LoadVideo:
    """Load video from S3. Returns video frames as image batch."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "s3_config": ("S3_CONFIG",),
                "s3_key": ("STRING", {
                    "default": "path/to/video.mp4",
                    "multiline": False,
                }),
            },
            "optional": {
                "frame_start": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                }),
                "frame_count": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                }),
                "skip_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("frames", "frame_count", "fps")
    FUNCTION = "load_video"
    CATEGORY = "Attome/S3"

    def load_video(self, s3_config, s3_key, frame_start=0,
                   frame_count=0, skip_frames=0):
        # Skip S3 loading if s3_key is empty - return empty video
        if not validate_s3_key(s3_key):
            # Return a single 512x512 black frame (proper size to avoid division errors)
            empty_frames = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (empty_frames, 0, 24.0)
        
        data = download_from_s3(s3_config, s3_key)

        ext = os.path.splitext(s3_key)[1] or ".mp4"

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            import cv2

            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Set start frame
            if frame_start > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

            frames = []
            frame_idx = 0
            max_frames = frame_count if frame_count > 0 else total_frames

            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames if needed
                if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    continue

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                frame_idx += 1

            cap.release()

            if not frames:
                raise ValueError("No frames could be extracted from video")

            # Stack frames and convert to tensor (BHWC format)
            frames_np = np.stack(frames).astype(np.float32) / 255.0
            frames_tensor = torch.from_numpy(frames_np)

        finally:
            os.unlink(tmp_path)

        return (frames_tensor, len(frames), fps)


class AttomeS3SaveVideo:
    """Save video (image batch) to S3."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "s3_config": ("S3_CONFIG",),
                "frames": ("IMAGE",),
                "s3_key": ("STRING", {
                    "default": "path/to/output.mp4",
                    "multiline": False,
                }),
            },
            "optional": {
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                }),
                "codec": (["mp4v", "avc1", "XVID"], {
                    "default": "mp4v",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("s3_uri",)
    FUNCTION = "save_video"
    CATEGORY = "Attome/S3"
    OUTPUT_NODE = True

    def save_video(self, s3_config, frames, s3_key, fps=24.0, codec="mp4v"):
        import cv2

        # Convert tensor to numpy
        frames_np = frames.cpu().numpy()
        frames_np = (frames_np * 255).clip(0, 255).astype(np.uint8)

        # Get dimensions
        num_frames, height, width, _ = frames_np.shape

        ext = os.path.splitext(s3_key)[1] or ".mp4"

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

            for frame in frames_np:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()

            with open(tmp_path, "rb") as f:
                data = f.read()
        finally:
            os.unlink(tmp_path)

        uri = upload_to_s3(s3_config, s3_key, data, content_type="video/mp4")
        return (uri,)


# ============================================================================
# NODE MAPPINGS
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "AttomeS3Config": AttomeS3Config,
    "AttomeS3LoadText": AttomeS3LoadText,
    "AttomeS3SaveText": AttomeS3SaveText,
    "AttomeS3LoadImage": AttomeS3LoadImage,
    "AttomeS3SaveImage": AttomeS3SaveImage,
    "AttomeS3LoadAudio": AttomeS3LoadAudio,
    "AttomeS3SaveAudio": AttomeS3SaveAudio,
    "AttomeS3LoadVideo": AttomeS3LoadVideo,
    "AttomeS3SaveVideo": AttomeS3SaveVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AttomeS3Config": "Attome S3 Config",
    "AttomeS3LoadText": "Attome S3 Load Text",
    "AttomeS3SaveText": "Attome S3 Save Text",
    "AttomeS3LoadImage": "Attome S3 Load Image",
    "AttomeS3SaveImage": "Attome S3 Save Image",
    "AttomeS3LoadAudio": "Attome S3 Load Audio",
    "AttomeS3SaveAudio": "Attome S3 Save Audio",
    "AttomeS3LoadVideo": "Attome S3 Load Video",
    "AttomeS3SaveVideo": "Attome S3 Save Video",
}
