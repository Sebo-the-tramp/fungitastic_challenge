import os
import cv2
import numpy as np


def read_segments(segment_path):
    segments = []
    class_ids = []

    if os.path.exists(segment_path):
        with open(segment_path, "r") as f:
            lines = f.readlines()

            for line in lines:
                parts = line.strip().split(" ")
                class_id = int(parts[0])
                points = parts[1:]
                polygon = [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]
                segments.append(polygon)
                class_ids.append(class_id)
        
        return segments, class_ids
        # return torch.from_numpy(np.array(segment)).unsqueeze(0)  # Add channel dimension
    else:
        print(f"Segment not found for {segment_path}")
        return [], None
    
def polygon_to_mask(polygons, img_width, img_height):
    """
    Converts a list of polygons into a single binary mask.
    segments: List of polygons, where each polygon is a list of (x, y) tuples.
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for polygon in polygons:
        # Scale the normalized (0-1) coordinates to actual pixel values
        # Reshaping to (-1, 1, 2) is the safest format for all cv2 versions
        pixel_coords = np.array([[x * img_width, y * img_height] for x, y in polygon], dtype=np.int32)
        pixel_coords = pixel_coords.reshape((-1, 1, 2))
        
        # Fill polygons ONE BY ONE to prevent the even-odd overlap bug
        cv2.fillPoly(mask, [pixel_coords], 255)
    
    return mask