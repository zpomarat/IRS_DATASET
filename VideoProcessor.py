from ultralytics import YOLO
import cv2
import torch
import numpy as np


class VideoProcessor:
    """
    Process experimental videos.

    Features:
        - Face detection and anonymization using YOLOv8
        - Video cropping around the push phase
        - Synchronization with force signals
    """

    def __init__(self, yolo_model_path: str, conf: float = 0.1, device: str = None):
        """
        Initialize the VideoProcessor.

        Args:
            yolo_model_path: Path to the YOLOv8 face detection model weights.
            conf: Confidence threshold for face detection (default: 0.1).
                  Low value is intentional to minimize missed detections.
            device: Compute device ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.model = YOLO(yolo_model_path)
        self.conf = conf
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"VideoProcessor initialized on {self.device}")

    def _pixelate(self, face_roi: np.ndarray, blocks: int = 10) -> np.ndarray:
        """
        Anonymize a face region using pixelation.

        Downscales then upscales the region to destroy facial details.

        Args:
            face_roi: Cropped face region as a numpy array (H, W, C).
            blocks: Number of pixels in the downscaled image.
                    Lower values produce stronger pixelation (default: 10).

        Returns:
            Pixelated face region as a numpy array (H, W, C).
        """
        h, w = face_roi.shape[:2]
        small = cv2.resize(face_roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def blur_faces(self, input_path: str, output_path: str,
                   detect_every: int = 10, padding: int = 30,
                   pixelate_blocks: int = 10):
        """
        Detect and anonymize faces in a video using YOLOv8 + pixelation.

        Face detection runs every `detect_every` frames for efficiency.
        Bounding boxes from the last detection are reused on intermediate frames.
        Anonymization is applied within an elliptical mask for natural-looking results.
        Audio is implicitly removed since OpenCV only writes video frames.

        Args:
            input_path: Path to the input video file.
            output_path: Path to save the anonymized output video (MP4).
            detect_every: Run YOLO every N frames (default: 10).
            padding: Extra pixels added around detected bounding boxes (default: 30).
            pixelate_blocks: Pixelation strength — lower = stronger (default: 10).
        """
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing {input_path} — {total_frames} frames")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        last_boxes = []  # bounding boxes from the last YOLO detection
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run face detection every `detect_every` frames
            if frame_count % detect_every == 0:
                results = self.model(frame, conf=self.conf)[0]
                last_boxes = results.boxes.xyxy

            # Apply anonymization using the last known bounding boxes
            for box in last_boxes:
                x1, y1, x2, y2 = map(int, box)

                # Expand bounding box with padding
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                y2 = min(frame.shape[0], y2 + padding)

                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue

                # Pixelate the face region
                anonymized = self._pixelate(face_roi, blocks=pixelate_blocks)

                # Apply pixelation within an elliptical mask for rounded edges
                mask = np.zeros(face_roi.shape[:2], dtype=np.uint8)
                center = (face_roi.shape[1] // 2, face_roi.shape[0] // 2)
                axes = (face_roi.shape[1] // 2, face_roi.shape[0] // 2)
                cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

                mask_3d = cv2.merge([mask, mask, mask])
                face_roi_anonymized = np.where(mask_3d > 0, anonymized, face_roi)
                frame[y1:y2, x1:x2] = face_roi_anonymized

            out.write(frame)
            frame_count += 1

        print(f"Done — {frame_count} frames written to {output_path}")
        cap.release()
        out.release()