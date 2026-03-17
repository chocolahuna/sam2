import cv2
import numpy as np
import os

class LineAwareUnderpaintEngine:
    def __init__(self):
        pass

    def extract_line_art(self, image_np, threshold=0.7):
        """Extracts line art region from image. 0 is paper, 255 is line."""
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # ぼかしを入れてノイズを軽減
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # しきい値以下（暗い部分）を線とする
        _, binary = cv2.threshold(blurred, int(255 * threshold), 255, cv2.THRESH_BINARY_INV)
        return binary

    def refine_masks(self, source_img, parts_masks):
        """
        Refines multiple masks to fit the line art using Watershed.
        source_img: Original RGB image (numpy)
        parts_masks: Dictionary of {part_id: binary_mask_uint8}
        """
        if not parts_masks:
            return {}

        h, w = source_img.shape[:2]
        line_mask = self.extract_line_art(source_img)
        
        # Markers for watershed: 0 is unknown, 1..N are part seeds
        markers = np.zeros((h, w), dtype=np.int32)
        
        part_ids = list(parts_masks.keys())
        # We need to handle background / unmasked areas too, 
        # otherwise everything will be filled by one of the parts.
        
        # Combined mask of all parts
        all_parts_mask = np.zeros((h, w), dtype=np.uint8)
        
        for i, pid in enumerate(part_ids):
            mask = parts_masks[pid]
            all_parts_mask = cv2.bitwise_or(all_parts_mask, mask)
            
            # Seed is the area of the mask that is NOT on a line
            # Shrink the mask for seeds to ensure they are "safe" internal points
            kernel = np.ones((5,5), np.uint8)
            eroded = cv2.erode(mask, kernel, iterations=1)
            seed_area = cv2.bitwise_and(eroded, cv2.bitwise_not(line_mask))
            markers[seed_area > 0] = i + 1
            
        # Outside background seed
        # Areas that are far from any part and not on lines
        kernel_bg = np.ones((11,11), np.uint8)
        dilated_all = cv2.dilate(all_parts_mask, kernel_bg, iterations=3)
        background_seed = cv2.bitwise_not(cv2.bitwise_or(dilated_all, line_mask))
        markers[background_seed > 0] = len(part_ids) + 1
            
        # Perform Watershed
        if len(source_img.shape) == 2:
            color_img = cv2.cvtColor(source_img, cv2.COLOR_GRAY2BGR)
        else:
            # OpenCV wants BGR
            color_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)
            
        cv2.watershed(color_img, markers)
        
        # markers == -1 are boundaries, others are 1..N
        new_masks = {}
        for i, pid in enumerate(part_ids):
            # The refined mask includes its watershed region
            refined = (markers == i + 1).astype(np.uint8) * 255
            
            # 境界線 (-1) も含める（膨張1回分相当）
            # これをしないと線の中央に白い隙間ができる場合がある
            kernel_edge = np.ones((3,3), np.uint8)
            refined = cv2.dilate(refined, kernel_edge, iterations=1)
            
            # 線画と重なっている部分のみを最終的なマスクとして返すか、
            # もしくは元々のパーツ範囲＋拡張分として返す。
            # 下塗りエンジンとしては「線画の範囲まで埋める」のが目的なので、
            # dilated された refined をそのまま返すのが一般的
            new_masks[pid] = refined
            
        return new_masks

    def save_image_unicode(self, path, img):
        """Helper to save images with Unicode paths on Windows"""
        is_success, buffer = cv2.imencode(".png", img)
        if is_success:
            with open(path, "wb") as f:
                f.write(buffer)
            return True
        return False

line_engine = LineAwareUnderpaintEngine()

def get_line_engine():
    return line_engine
