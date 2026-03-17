import torch
import os
import sys

# Add sam2 to path
sys.path.append(os.path.join(os.getcwd(), "sam2"))

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = "sam2/checkpoints/sam2.1_hiera_small.pt"
    model_cfg = os.path.join("configs", "sam2.1", "sam2.1_hiera_s.yaml")
    
    # Change CWD to sam2 to load config correctly if needed, 
    # but build_sam2 usually looks relative to the sam2 package or current dir.
    # The config is actually in sam2/sam2/configs/... or similar depending on how it's installed.
    # SAM 2.1 configs are usually at sam2/sam2/configs/sam2.1/sam2.1_hiera_s.yaml
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("CUDA not available, using CPU")
        device = "cpu"

    # Official config path in repo
    model_cfg_path = "sam2.1/sam2.1_hiera_s.yaml" # Relative to sam2 package config root
    
    # Re-build path for clarity
    predictor = SAM2ImagePredictor(build_sam2(model_cfg_path, checkpoint, device=device))
    print("SAM 2.1 Small model loaded successfully!")
    print("Ready for inference.")

except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure sam2 is installed correctly.")
except Exception as e:
    print(f"Error: {e}")
