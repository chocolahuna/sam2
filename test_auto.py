import sys
import os
import cv2
import numpy as np
import uuid
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
from sam_service import get_sam_service

# Use the latest project
projects_dir = "projects"
project_id = "191f7df7-fb25-451e-848e-d9134a9190c1" # User's screen shows ID starting with 191f7df7
project_dirs = os.listdir(projects_dir)
for d in project_dirs:
    if d.startswith("191f7df7"):
        project_id = d
        break

image_path = os.path.join(projects_dir, project_id, "source", "りのLinex1.png")
if not os.path.exists(image_path):
    # Try just listing the first project's image
    project_id = project_dirs[-1]
    source_dir = os.path.join(projects_dir, project_id, "source")
    image_name = os.listdir(source_dir)[0]
    image_path = os.path.join(source_dir, image_name)

print("Testing image:", image_path)

sam = get_sam_service()
# Dump preprocessed image
img = sam._preprocess_image(image_path)

# Downscale for faster testing and thicker lines relative to grid
max_dim = 1024
if max(img.size) > max_dim:
    ratio = max_dim / max(img.size)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    print(f"Resized image for test to {new_size}")

img_np = np.array(img)
print("Image mean:", img_np.mean())
print("Image shape:", img_np.shape)

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
custom_mask_generator = SAM2AutomaticMaskGenerator(
    model=sam.model,
    points_per_side=32, 
    pred_iou_thresh=0.5,
    stability_score_thresh=0.5,
    min_mask_region_area=100
)

masks_custom = custom_mask_generator.generate(img_np)
print(f"Generated {len(masks_custom)} masks with custom params and resized image.")

if len(masks_custom) > 0:
    print(f"First mask area: {masks_custom[0]['area']}")
