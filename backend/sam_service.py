import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2

# Sam core is now in backend/sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SAM2Service:
    def __init__(self):
        # Force CPU for stability during initial setup, as RTX 50 series might have compatibility issues with current PyTorch
        self.device = "cpu" 
        print(f"DEBUG: Forcing SAM 2.1 to {self.device} mode for compatibility.")
        
        # Paths are now relative to backend directory
        base_dir = os.path.dirname(__file__)
        self.checkpoint = os.path.join(base_dir, "checkpoints", "sam2.1_hiera_small.pt")
        # Config path needs to be either absolute or handled by sam2 package. 
        # Since we are in backend/, and sam2 is a package in backend/, sam2.configs should resolve.
        # However, build_sam2 often expects the path relative to the sam2 PACKAGE root or absolute.
        self.model_cfg = os.path.join(base_dir, "sam2", "configs", "sam2.1", "sam2.1_hiera_s.yaml")
        
        print(f"DEBUG: Loading SAM 2.1 Small model on {self.device}...")
        try:
            self.model = build_sam2(self.model_cfg, self.checkpoint, device=self.device)
            self.predictor = SAM2ImagePredictor(self.model)
            self.mask_generator = SAM2AutomaticMaskGenerator(self.model)
            print("DEBUG: SAM 2.1 Small model loaded successfully on CPU.")
        except Exception as e:
            print(f"ERROR: Failed to load SAM 2 model: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def _preprocess_image(self, image_path):
        """画像を読み込み、透過なら白背景に貼り付けて RGB に変換する"""
        import io
        with open(image_path, "rb") as f:
            image = Image.open(io.BytesIO(f.read()))
            image.load() # Preload to memory before BytesIO closes
            
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            # Alpha チャンネルがある場合は白背景に合成
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            background.paste(image, mask=image.split()[3]) # 3 is alpha
            return background
        return image.convert("RGB")

    def set_image(self, image_path):
        """主画像をセットし、埋め込みを計算する"""
        image = self._preprocess_image(image_path)
        image_np = np.array(image)
        self.predictor.set_image(image_np)
        return image.size # (width, height)

    def predict_by_points(self, points, labels):
        """点指定によるマスク生成"""
        # points: [[x, y], ...], labels: [1, 0, ...] (1: positive, 0: negative)
        masks, scores, logits = self.predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array(labels),
            multimask_output=True,
        )
        # スコアが最も高いものを返す (score index 0 usually for multi-mask)
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]

    def predict_by_box(self, box):
        """矩形指定によるマスク生成 (box: [x1, y1, x2, y2])"""
        masks, scores, logits = self.predictor.predict(
            box=np.array(box),
            multimask_output=False,
        )
        return masks[0], scores[0]

    def generate_auto_masks(self, image_path):
        """自動候補生成 (ダウンスケールして高速・高感度抽出)"""
        print(f"DEBUG: Generating automatic masks for {image_path}")
        image = self._preprocess_image(image_path)
        orig_w, orig_h = image.size
        
        # 最大1024pxにダウンスケール (線画の相対的な太さを増し、抽出速度を上げる)
        max_dim = 1024
        scale = 1.0
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        image_np = np.array(image)
        print(f"DEBUG: Resized image shape for generation: {image_np.shape}")
        
        # 線画向けのチューニングを施した Mask Generator を作成
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        custom_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=64, # 高密度グリッド (4096点)
            pred_iou_thresh=0.5, # 線画のためIoU閾値と安定性閾値を大幅に緩和
            stability_score_thresh=0.5,
            min_mask_region_area=100 # 小さすぎるノイズを除去
        )
        
        masks = custom_generator.generate(image_np)
        print(f"DEBUG: Generated {len(masks)} masks.")
        
        # マスクを元の解像度にスケールアップ
        if scale != 1.0:
            for m in masks:
                seg_uint8 = (m["segmentation"].astype(np.uint8)) * 255
                seg_up = cv2.resize(seg_uint8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                m["segmentation"] = seg_up > 0
                x, y, w, h = m["bbox"]
                m["bbox"] = [x / scale, y / scale, w / scale, h / scale]
                m["area"] = int(np.sum(m["segmentation"]))
        
        # 面積が小さい順にソート (パーツを見つけやすくするため)
        masks.sort(key=lambda x: x["area"])
        return masks

sam_service = None

def get_sam_service():
    global sam_service
    if sam_service is None:
        sam_service = SAM2Service()
    return sam_service
