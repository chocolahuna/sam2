import os
import shutil
import uuid
import json
from typing import List, Optional, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np
from PIL import Image

# Import services
from sam_service import get_sam_service
from line_engine import get_line_engine

app = FastAPI(title="SAM 2 PaintBase API")

# Enable CORS for Electron
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Workspace and Projects
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "projects"))
os.makedirs(PROJECTS_DIR, exist_ok=True)

# Mount projects dir for static file access (preview masks etc.)
app.mount("/projects", StaticFiles(directory=PROJECTS_DIR), name="projects")

@app.get("/projects")
async def list_projects():
    projects = []
    if not os.path.exists(PROJECTS_DIR):
        return []
    for project_id in os.listdir(PROJECTS_DIR):
        project_path = os.path.join(PROJECTS_DIR, project_id)
        json_path = os.path.join(project_path, "project.json")
        if os.path.isdir(project_path) and os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                projects.append({
                    "id": data["id"],
                    "name": data["name"],
                    "width": data.get("width"),
                    "height": data.get("height"),
                    "parts_count": len(data.get("parts", []))
                })
    return projects

# Schemas
class ProjectCreate(BaseModel):
    name: str

class PointPrompt(BaseModel):
    project_id: str
    points: List[List[float]]
    labels: List[int]

class BoxPrompt(BaseModel):
    project_id: str
    box: List[float] # [x1, y1, x2, y2]

class UnderpaintParams(BaseModel):
    project_id: str
    line_threshold: float = 0.45
    fill_internal_lines: bool = True

@app.post("/project/create")
async def create_project(file: UploadFile = File(...)):
    project_id = str(uuid.uuid4())
    project_path = os.path.join(PROJECTS_DIR, project_id)
    os.makedirs(project_path, exist_ok=True)
    
    source_dir = os.path.join(project_path, "source")
    os.makedirs(source_dir, exist_ok=True)
    
    file_path = os.path.join(source_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Initialize SAM 2 with this image
    sam = get_sam_service()
    width, height = sam.set_image(file_path)
    
    project_data = {
        "id": project_id,
        "name": file.filename,
        "source_image": f"source/{file.filename}",
        "width": width,
        "height": height,
        "parts": []
    }
    
    with open(os.path.join(project_path, "project.json"), "w") as f:
        json.dump(project_data, f, indent=2)
        
    return project_data

class AutoSegmentRequest(BaseModel):
    project_id: str

@app.post("/sam/auto-segment")
async def auto_segment(request: AutoSegmentRequest):
    project_id = request.project_id
    project_path = os.path.join(PROJECTS_DIR, project_id)
    with open(os.path.join(project_path, "project.json"), "r") as f:
        project = json.load(f)
    
    source_path = os.path.join(project_path, project["source_image"])
    sam = get_sam_service()
    masks = sam.generate_auto_masks(source_path)
    
    # Save masks to files
    masks_dir = os.path.join(project_path, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    results = []
    for i, m in enumerate(masks):
        mask_id = f"auto_{i}"
        mask_filename = f"{mask_id}.png"
        mask_path = os.path.join(masks_dir, mask_filename)
        
        # Convert boolean mask to image
        mask_img = (m["segmentation"] * 255).astype(np.uint8)
        is_success, buffer = cv2.imencode(".png", mask_img)
        if is_success:
            with open(mask_path, "wb") as f:
                f.write(buffer)
        
        results.append({
            "id": mask_id,
            "mask_path": f"masks/{mask_filename}",
            "bbox": m["bbox"], # [x, y, w, h]
            "area": float(m["area"]),
            "predicted_iou": float(m["predicted_iou"])
        })
    
    # Update project.json
    project["parts"] = results
    with open(os.path.join(project_path, "project.json"), "w") as f:
        json.dump(project, f, indent=2)
        
    return results

@app.post("/sam/predict-points")
async def predict_points(prompt: PointPrompt):
    project_id = prompt.project_id
    project_path = os.path.join(PROJECTS_DIR, project_id)
    with open(os.path.join(project_path, "project.json"), "r") as f:
        project = json.load(f)
    
    source_path = os.path.join(project_path, project["source_image"])
    sam = get_sam_service()
    
    # Force setting image to ensure embeddings are ready
    sam.set_image(source_path)
    
    mask, score = sam.predict_by_points(prompt.points, prompt.labels)
    
    # Save the new mask as a temporary preview
    masks_dir = os.path.join(project_path, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    mask_filename = "preview.png"
    mask_path = os.path.join(masks_dir, mask_filename)
    
    mask_img = (mask * 255).astype(np.uint8)
    is_success, buffer = cv2.imencode(".png", mask_img)
    if is_success:
        with open(mask_path, "wb") as f:
            f.write(buffer)
    
    # Calculate simple bbox and area
    y_indices, x_indices = np.where(mask)
    if len(x_indices) > 0:
        x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
        y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        area = int(np.sum(mask))
    else:
        bbox = [0, 0, 0, 0]
        area = 0

    preview_part = {
        "id": "preview",
        "mask_path": f"masks/{mask_filename}",
        "bbox": bbox,
        "area": area,
        "predicted_iou": float(score)
    }
    
    # Return current parts PLUS the preview
    return project["parts"] + [preview_part]

class LassoPrompt(BaseModel):
    project_id: str
    points: List[List[float]]
    label: int # 1 for positive (add), 0 for negative (subtract)
    target_part_id: Optional[str] = None # Support modifying existing part

@app.post("/sam/predict-lasso")
async def predict_lasso(prompt: LassoPrompt):
    project_id = prompt.project_id
    project_path = os.path.join(PROJECTS_DIR, project_id)
    with open(os.path.join(project_path, "project.json"), "r") as f:
        project = json.load(f)
        
    width = project.get("width", 1024)
    height = project.get("height", 1024)
    
    masks_dir = os.path.join(project_path, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    # Check if there is already a preview or target part we are modifying
    base_mask = np.zeros((height, width), dtype=np.uint8)
    
    if prompt.target_part_id:
        target_path = None
        for p in project["parts"]:
            if p["id"] == prompt.target_part_id:
                target_path = os.path.join(project_path, p["mask_path"])
                break
        if target_path and os.path.exists(target_path):
            mask_array = np.fromfile(target_path, dtype=np.uint8)
            base_mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
    
    # Or start from the current preview if it exists and we're just adding more to it
    preview_path = os.path.join(masks_dir, "preview.png")
    if prompt.target_part_id is None and os.path.exists(preview_path):
        preview_array = np.fromfile(preview_path, dtype=np.uint8)
        base_mask = cv2.imdecode(preview_array, cv2.IMREAD_GRAYSCALE)
    if base_mask is None:
        base_mask = np.zeros((height, width), dtype=np.uint8)

    # Draw the polygon
    pts = np.array(prompt.points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    poly_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [pts], 255)
    
    # Combine masks based on label
    if prompt.label == 1:
        # Add (bitwise OR)
        new_mask = cv2.bitwise_or(base_mask, poly_mask)
    else:
        # Subtract (bitwise AND with NOT)
        new_mask = cv2.bitwise_and(base_mask, cv2.bitwise_not(poly_mask))
    
    # Save back to preview using numpy
    is_success, buffer = cv2.imencode(".png", new_mask)
    if is_success:
        with open(preview_path, "wb") as f:
            f.write(buffer)
    
    # Calculate simple bbox and area
    y_indices, x_indices = np.where(new_mask > 0)
    if len(x_indices) > 0:
        x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
        y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        area = int(np.sum(new_mask > 0))
    else:
        bbox = [0, 0, 0, 0]
        area = 0

    preview_part = {
        "id": "preview",
        "mask_path": "masks/preview.png",
        "bbox": bbox,
        "area": area,
        "predicted_iou": 1.0 # Manual editing is always highest confidence
    }
    
    # Remove existing preview part if any before returning
    final_parts = [p for p in project["parts"] if p["id"] != "preview"]
    
    # We don't save preview to project.json, we just return it appended
    return final_parts + [preview_part]

class CommitPreviewRequest(BaseModel):
    project_id: str
    target_part_id: Optional[str] = None

@app.post("/sam/commit-preview")
async def commit_preview(request: CommitPreviewRequest):
    project_id = request.project_id
    project_path = os.path.join(PROJECTS_DIR, project_id)
    with open(os.path.join(project_path, "project.json"), "r") as f:
        project = json.load(f)
        
    masks_dir = os.path.join(project_path, "masks")
    preview_path = os.path.join(masks_dir, "preview.png")
    
    if not os.path.exists(preview_path):
        return project["parts"]
        
    mask_id = f"manual_{uuid.uuid4().hex[:8]}"
    mask_filename = f"{mask_id}.png"
    final_path = os.path.join(masks_dir, mask_filename)
    
    # Move preview file to final filename
    shutil.move(preview_path, final_path)
    
    # Calculate area and bbox for the final structure using numpy
    final_array = np.fromfile(final_path, dtype=np.uint8)
    mask_img = cv2.imdecode(final_array, cv2.IMREAD_GRAYSCALE)
    y_indices, x_indices = np.where(mask_img > 0)
    if len(x_indices) > 0:
        x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
        y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        area = int(np.sum(mask_img > 0))
    else:
        bbox = [0, 0, 0, 0]
        area = 0
        
    new_part = {
        "id": mask_id,
        "mask_path": f"masks/{mask_filename}",
        "bbox": bbox,
        "area": area,
        "predicted_iou": 1.0 # Manual masks have highest assumed IOU
    }
    
    if request.target_part_id:
        # Overwrite existing part while maintaining order
        found = False
        for i, p in enumerate(project["parts"]):
            if p["id"] == request.target_part_id:
                # Optionally, we could delete the old mask image file here
                # old_mask_path = os.path.join(project_path, p["mask_path"])
                # if os.path.exists(old_mask_path): os.remove(old_mask_path)
                
                # Keep the original ID so frontend selection doesn't break
                new_part["id"] = p["id"]
                project["parts"][i] = new_part
                found = True
                break
        if not found:
             project["parts"].append(new_part)
    else:
        project["parts"].append(new_part)
        
    with open(os.path.join(project_path, "project.json"), "w") as f:
        json.dump(project, f, indent=2)
        
    return project["parts"]

@app.delete("/project/{project_id}")
async def delete_project(project_id: str):
    """Deletes a project folder and all its contents."""
    project_path = os.path.join(PROJECTS_DIR, project_id)
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="Project not found")
    
    try:
        shutil.rmtree(project_path)
        return {"status": "success", "message": f"Project {project_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")

@app.get("/project/{project_id}/export-full")
async def export_full_project(project_id: str):
    """Exports the entire project folder as a ZIP file."""
    import io
    import zipfile
    import urllib.parse
    import traceback
    from fastapi.responses import StreamingResponse

    project_path = os.path.join(PROJECTS_DIR, project_id)
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="Project not found")
    
    try:
        with open(os.path.join(project_path, "project.json"), "r", encoding="utf-8") as f:
            project_data = json.load(f)
        
        name = project_data.get('name', project_id)
        # Remove characters that might be unsafe for filenames
        safe_name = "".join([c for c in name if c not in '<>:"/\\|?*'])
        zip_filename = f"paintbase_full_{safe_name}.zip"
        
        # Use quote for the filename to support Unicode in headers
        # Use both filename and filename* for maximum compatibility
        quoted_filename = urllib.parse.quote(zip_filename)
        content_disposition = f"attachment; filename=\"{quoted_filename}\"; filename*=UTF-8''{quoted_filename}"

        io_buf = io.BytesIO()
        with zipfile.ZipFile(io_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(project_path):
                for file in files:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, project_path)
                    zf.write(abs_path, arcname=rel_path)
        
        # Must seek(0) after closing the ZipFile context
        io_buf.seek(0)
        return StreamingResponse(
            io_buf,
            media_type="application/zip",
            headers={
                "Access-Control-Expose-Headers": "Content-Disposition",
                "Content-Disposition": content_disposition
            }
        )
    except Exception as e:
        print(f"Full export error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/project/import")
async def import_project(file: UploadFile = File(...)):
    """Imports a project from a ZIP file."""
    import zipfile
    import io
    import uuid
    from fastapi import UploadFile, File
    
    # Generate a new project ID for the imported project to avoid collisions 
    # OR try to preserve the original ID if found in project.json
    
    content = await file.read()
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        # Check for project.json
        if "project.json" not in zf.namelist():
            raise HTTPException(status_code=400, detail="Invalid project ZIP: project.json not found")
        
        with zf.open("project.json") as f:
            project_data = json.loads(f.read().decode("utf-8"))
            
        original_id = project_data.get("id")
        if not original_id:
            original_id = str(uuid.uuid4())
            
        project_path = os.path.join(PROJECTS_DIR, original_id)
        
        # If ID already exists, generate a new one
        if os.path.exists(project_path):
            new_id = str(uuid.uuid4())
            project_path = os.path.join(PROJECTS_DIR, new_id)
            project_data["id"] = new_id
            
        os.makedirs(project_path, exist_ok=True)
        zf.extractall(project_path)
        
        # Re-save project.json in case ID was changed
        with open(os.path.join(project_path, "project.json"), "w") as f:
            json.dump(project_data, f, indent=2)
            
    return project_data

@app.get("/project/{project_id}")
async def get_project(project_id: str):
    project_path = os.path.join(PROJECTS_DIR, project_id)
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="Project not found")
        
    with open(os.path.join(project_path, "project.json"), "r") as f:
        return json.load(f)

@app.get("/project/{project_id}/export")
async def export_project(project_id: str):
    """
    Exports all mask parts in their original resolution as a ZIP file.
    """
    import zipfile
    import io
    import traceback
    
    try:
        project_path = os.path.join(PROJECTS_DIR, project_id)
        if not os.path.exists(project_path):
            raise HTTPException(status_code=404, detail="Project not found")
            
        with open(os.path.join(project_path, "project.json"), "r") as f:
            project = json.load(f)
            
        source_path = os.path.join(project_path, project["source_image"])
        
        # Get original image dimensions using numpy
        orig_array = np.fromfile(source_path, dtype=np.uint8)
        if orig_array.size == 0:
            raise HTTPException(status_code=500, detail=f"Source image not found: {source_path}")
        orig_img = cv2.imdecode(orig_array, cv2.IMREAD_COLOR)
        if orig_img is None:
            raise HTTPException(status_code=500, detail=f"Source image not found: {source_path}")
        orig_h, orig_w = orig_img.shape[:2]
        
        # Create in-memory zip
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for idx, part in enumerate(project.get("parts", [])):
                if part["id"] == "preview":
                    continue # Skip uncommitted preview
                    
                mask_path = os.path.join(project_path, part["mask_path"])
                if not os.path.exists(mask_path):
                    continue
                    
                # Read working resolution mask using numpy
                mask_array = np.fromfile(mask_path, dtype=np.uint8)
                mask_img = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
                if mask_img is None:
                    continue
                
                # Resize back to original resolution
                # Use INTER_NEAREST to keep sharp edges (binary mask)
                resized_mask = cv2.resize(mask_img, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                
                # Convert back to bytes for zip
                is_success, buffer = cv2.imencode(".png", resized_mask)
                if is_success:
                    # E.g. part_001_auto_0.png
                    filename = f"part_{idx+1:03d}_{part['id']}.png"
                    zip_file.writestr(filename, buffer.tobytes())
                    
        zip_buffer.seek(0)
        
        headers = {
            'Content-Disposition': f'attachment; filename="paintbase_{project_id}.zip"'
        }
        from fastapi.responses import StreamingResponse
        return StreamingResponse(iter([zip_buffer.getvalue()]), media_type="application/zip", headers=headers)
        
    except HTTPException:
        raise
    except Exception as e:
        print("Export failed with exception:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/project/{project_id}/refine")
async def refine_project_masks(project_id: str):
    """
    Refines all existing masks in a project to fit line art boundaries.
    """
    from line_engine import get_line_engine
    engine = get_line_engine()
    
    project_path = os.path.join(PROJECTS_DIR, project_id)
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="Project not found")
        
    with open(os.path.join(project_path, "project.json"), "r") as f:
        project = json.load(f)
        
    source_path = os.path.join(project_path, project["source_image"])
    
    # Load source image
    source_array = np.fromfile(source_path, dtype=np.uint8)
    source_img = cv2.imdecode(source_array, cv2.IMREAD_COLOR)
    if source_img is None:
        raise HTTPException(status_code=500, detail="Failed to load source image for refinement")
    
    # Load all masks
    parts_masks = {}
    for part in project.get("parts", []):
        if part["id"] == "preview": continue
        
        mask_path = os.path.join(project_path, part["mask_path"])
        if os.path.exists(mask_path):
            mask_array = np.fromfile(mask_path, dtype=np.uint8)
            mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                parts_masks[part["id"]] = mask
                
    if not parts_masks:
        return project["parts"]
        
    # Run refinement
    try:
        refined_masks = engine.refine_masks(source_img, parts_masks)
        
        # Save back refined masks
        for pid, refined_mask in refined_masks.items():
            # Find the original part to get its path
            for part in project["parts"]:
                if part["id"] == pid:
                    abs_mask_path = os.path.join(project_path, part["mask_path"])
                    engine.save_image_unicode(abs_mask_path, refined_mask)
                    
                    # Update bbox and area in part metadata
                    y_indices, x_indices = np.where(refined_mask > 0)
                    if len(x_indices) > 0:
                        x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
                        y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
                        part["bbox"] = [x_min, y_min, x_max - x_min, y_max - y_min]
                        part["area"] = int(np.sum(refined_mask > 0))
                    break
        
        # Save updated project metadata
        with open(os.path.join(project_path, "project.json"), "w") as f:
            json.dump(project, f, indent=2)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Refinement error: {str(e)}")
        
    return project["parts"]

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
