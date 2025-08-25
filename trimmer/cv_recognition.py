import cv2
import numpy as np
from collections import deque

def detect_and_crop_faces(input_video_path, output_video_path, min_segment_duration=1.0, buffer_time=0.5, output_size=512, zoom_factor=1.5, show_confidence=True, confidence_threshold=0.6):
    """
    Mendeteksi wajah dalam video dan membuat output video square yang di-crop dan zoom pada wajah dengan confidence heatmap.
    
    Parameters:
    - input_video_path: path ke video input
    - output_video_path: path untuk video output
    - min_segment_duration: durasi minimum segmen dalam detik
    - buffer_time: waktu buffer sebelum dan sesudah deteksi wajah (dalam detik)
    - output_size: ukuran output video (square, default 512x512)
    - zoom_factor: faktor zoom pada wajah (default 1.5)
    - show_confidence: tampilkan confidence level dan heatmap
    """
    
    # Load face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    print(f"Output will be: {output_size}x{output_size} square format")
    
    # Calculate buffer frames
    buffer_frames = int(buffer_time * fps)
    min_segment_frames = int(min_segment_duration * fps)
    
    # First pass: detect all frames with faces and their confidence scores
    print("Scanning video for faces...")
    face_data = []  # Store (frame_number, face_rect, center_point, confidence_score)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with confidence scores using detectMultiScale3
        faces, reject_levels, level_weights = face_cascade.detectMultiScale3(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=3,  # Lower for more sensitivity to get confidence scores
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
            outputRejectLevels=True
        )
        
        # If faces detected, store the largest face with highest confidence
        if len(faces) > 0:
            # Calculate confidence scores based on level_weights and reject_levels
            confidences = []
            for i in range(len(faces)):
                if i < len(level_weights):
                    # Normalize confidence score (0-1)
                    confidence = min(1.0, max(0.0, level_weights[i] / 10.0))
                else:
                    confidence = 0.5  # Default confidence
                confidences.append(confidence)
            
            # Find face with highest confidence or largest size if confidences are similar
            best_face_idx = 0
            if len(confidences) > 1:
                # Combine confidence and size for best selection
                scores = []
                for i, (face, conf) in enumerate(zip(faces, confidences)):
                    x, y, w, h = face
                    size_score = (w * h) / (width * height)  # Normalize by frame size
                    combined_score = conf * 0.7 + size_score * 0.3
                    scores.append(combined_score)
                best_face_idx = np.argmax(scores)
            
            x, y, w, h = faces[best_face_idx]
            confidence = confidences[best_face_idx]
            
            if confidence > confidence_threshold:
            # Calculate center point of the face
                center_x = x + w // 2
                center_y = y + h // 2
                face_data.append((frame_count, faces[best_face_idx], (center_x, center_y), confidence))
        
        frame_count += 1
        
        # Progress indicator
        if frame_count % (fps * 10) == 0:  # Every 10 seconds
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    print(f"Found faces in {len(face_data)} frames out of {total_frames}")
    
    if not face_data:
        print("No faces detected in the video!")
        cap.release()
        return
    
    # Calculate confidence statistics
    confidences = [data[3] for data in face_data]
    avg_confidence = np.mean(confidences)
    min_confidence = np.min(confidences)
    max_confidence = np.max(confidences)
    
    print(f"Confidence stats - Avg: {avg_confidence:.3f}, Min: {min_confidence:.3f}, Max: {max_confidence:.3f}")
    
    # Create segments with buffer time
    segments = []
    face_frames = [data[0] for data in face_data]
    
    if face_frames:
        # Group consecutive frames into segments
        current_segment_start = max(0, face_frames[0] - buffer_frames)
        current_segment_end = face_frames[0]
        
        for i in range(1, len(face_frames)):
            # If frames are close enough, extend current segment
            if face_frames[i] - face_frames[i-1] <= buffer_frames * 2:
                current_segment_end = face_frames[i]
            else:
                # End current segment and start new one
                segments.append((current_segment_start, min(current_segment_end + buffer_frames, total_frames - 1)))
                current_segment_start = max(0, face_frames[i] - buffer_frames)
                current_segment_end = face_frames[i]
        
        # Add the last segment
        segments.append((current_segment_start, min(current_segment_end + buffer_frames, total_frames - 1)))
    
    # Filter out segments that are too short
    segments = [(start, end) for start, end in segments if (end - start) >= min_segment_frames]
    
    print(f"Created {len(segments)} segments:")
    for i, (start, end) in enumerate(segments):
        duration = (end - start) / fps
        print(f"  Segment {i+1}: frames {start}-{end} ({duration:.2f}s)")
    
    if not segments:
        print("No segments meet minimum duration requirement!")
        cap.release()
        return
    
    # Second pass: create square cropped output video with confidence visualization
    print("Creating square cropped output video with confidence visualization...")
    
    # Define codec and create VideoWriter for square output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_size, output_size))
    
    total_output_frames = 0
    
    # Create face data lookup for quick access
    face_dict = {data[0]: (data[1], data[2], data[3]) for data in face_data}
    
    for segment_idx, (start_frame, end_frame) in enumerate(segments):
        print(f"Processing segment {segment_idx + 1}/{len(segments)}...")
        
        # Set video position to start of segment
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Calculate average center position for this segment for smoother tracking
        segment_centers = []
        segment_confidences = []
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx in face_dict:
                segment_centers.append(face_dict[frame_idx][1])
                segment_confidences.append(face_dict[frame_idx][2])
        
        # Smooth center tracking using moving average
        smooth_centers = smooth_face_tracking(segment_centers, window_size=5)
        smooth_confidences = smooth_confidence_tracking(segment_confidences, window_size=3)
        center_idx = 0
        
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get face position and confidence for this frame
            confidence = 0.0
            if frame_idx in face_dict:
                face_rect, face_center, confidence = face_dict[frame_idx]
                center_x, center_y = face_center
                x, y, w, h = face_rect
            else:
                # Use smoothed center and confidence for frames without detected faces
                if center_idx < len(smooth_centers):
                    center_x, center_y = smooth_centers[center_idx]
                    confidence = smooth_confidences[center_idx] if center_idx < len(smooth_confidences) else 0.3
                else:
                    # Use last known position
                    center_x, center_y = smooth_centers[-1] if smooth_centers else (width//2, height//2)
                    confidence = smooth_confidences[-1] if smooth_confidences else 0.3
                
                # Estimate face size based on previous detections
                w = h = max(60, min(width, height) // 8)  # Default face size estimation
                x, y = center_x - w//2, center_y - h//2
            
            center_idx += 1
            
            # Calculate crop area around face with zoom
            face_size = max(w, h)
            crop_size = int(face_size * zoom_factor)
            
            # Ensure minimum crop size
            crop_size = max(crop_size, output_size // 4)
            
            # Calculate crop boundaries
            crop_x1 = max(0, center_x - crop_size // 2)
            crop_y1 = max(0, center_y - crop_size // 2)
            crop_x2 = min(width, crop_x1 + crop_size)
            crop_y2 = min(height, crop_y1 + crop_size)
            
            # Adjust if crop goes beyond frame boundaries
            if crop_x2 - crop_x1 < crop_size:
                crop_x1 = max(0, crop_x2 - crop_size)
            if crop_y2 - crop_y1 < crop_size:
                crop_y1 = max(0, crop_y2 - crop_size)
            
            # Crop the frame
            cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Resize to square output format
            if cropped_frame.size > 0:
                # Resize maintaining aspect ratio and center crop to square
                square_frame = resize_to_square(cropped_frame, output_size)
                
                if show_confidence:
                    # Add confidence visualization
                    square_frame = add_confidence_visualization(
                        square_frame, 
                        confidence, 
                        frame_idx in face_dict,
                        output_size
                    )
                    
                    # Add face detection rectangle with confidence color coding
                    if frame_idx in face_dict:
                        adj_x = int((x - crop_x1) * (output_size / (crop_x2 - crop_x1)))
                        adj_y = int((y - crop_y1) * (output_size / (crop_y2 - crop_y1)))
                        adj_w = int(w * (output_size / (crop_x2 - crop_x1)))
                        adj_h = int(h * (output_size / (crop_y2 - crop_y1)))
                        
                        # Color based on confidence (red=low, yellow=medium, green=high)
                        rect_color = get_confidence_color(confidence)
                        thickness = max(2, int(4 * confidence))
                        
                        # Draw rectangle if it's within bounds
                        if 0 <= adj_x < output_size and 0 <= adj_y < output_size:
                            cv2.rectangle(square_frame, (adj_x, adj_y), 
                                        (min(adj_x + adj_w, output_size), min(adj_y + adj_h, output_size)), 
                                        rect_color, thickness)
                
                out.write(square_frame)
                total_output_frames += 1
            else:
                # If crop failed, create a black frame with low confidence indicator
                black_frame = np.zeros((output_size, output_size, 3), dtype=np.uint8)
                if show_confidence:
                    black_frame = add_confidence_visualization(black_frame, 0.0, False, output_size)
                out.write(black_frame)
                total_output_frames += 1
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    output_duration = total_output_frames / fps
    original_duration = total_frames / fps
    compression_ratio = (1 - output_duration / original_duration) * 100
    
    print(f"\nProcessing complete!")
    print(f"Original video: {original_duration:.2f}s ({width}x{height})")
    print(f"Output video: {output_duration:.2f}s ({output_size}x{output_size})")
    print(f"Compression: {compression_ratio:.1f}%")
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"Output saved as: {output_video_path}")

def get_confidence_color(confidence):
    """
    Get color based on confidence level using heatmap colors
    Red (low) -> Yellow (medium) -> Green (high)
    """
    if confidence < 0.3:
        # Red for low confidence
        return (0, 0, 255)
    elif confidence < 0.6:
        # Interpolate between red and yellow
        ratio = (confidence - 0.3) / 0.3
        return (0, int(255 * ratio), int(255 * (1 - ratio)))
    elif confidence < 0.8:
        # Interpolate between yellow and green
        ratio = (confidence - 0.6) / 0.2
        return (0, 255, int(255 * ratio))
    else:
        # Green for high confidence
        return (0, 255, 0)

def add_confidence_visualization(frame, confidence, has_detection, output_size):
    """
    Add confidence bar and heatmap overlay to the frame
    """
    # Create a copy to avoid modifying original
    vis_frame = frame.copy()
    
    # Add confidence bar at the top
    bar_height = 20
    bar_width = output_size - 40
    bar_x = 20
    bar_y = 10
    
    # Background bar (dark gray)
    cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    
    # Confidence level bar with heatmap colors
    if confidence > 0:
        fill_width = int(bar_width * confidence)
        confidence_color = get_confidence_color(confidence)
        cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), confidence_color, -1)
    
    # Border for the bar
    cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)
    
    # Add confidence text
    conf_text = f"Confidence: {confidence:.3f}"
    detection_text = "DETECTED" if has_detection else "TRACKING"
    
    # Text color based on confidence
    text_color = (255, 255, 255) if confidence > 0.5 else (200, 200, 200)
    
    cv2.putText(vis_frame, conf_text, (bar_x, bar_y + bar_height + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.putText(vis_frame, detection_text, (bar_x, bar_y + bar_height + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, get_confidence_color(confidence), 1)
    
    return vis_frame

def resize_to_square(image, size):
    """
    Resize image to square format by center cropping and scaling
    """
    h, w = image.shape[:2]
    
    # If already square, just resize
    if h == w:
        return cv2.resize(image, (size, size))
    
    # Center crop to square
    if h > w:
        # Portrait - crop top and bottom
        crop_size = w
        start_y = (h - crop_size) // 2
        cropped = image[start_y:start_y + crop_size, :]
    else:
        # Landscape - crop left and right
        crop_size = h
        start_x = (w - crop_size) // 2
        cropped = image[:, start_x:start_x + crop_size]
    
    # Resize to target size
    return cv2.resize(cropped, (size, size))

def smooth_face_tracking(centers, window_size=5):
    """
    Apply moving average smoothing to face center positions for stable tracking
    """
    if not centers:
        return []
    
    smoothed = []
    for i in range(len(centers)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(centers), i + window_size // 2 + 1)
        
        # Calculate average position
        avg_x = sum(center[0] for center in centers[start_idx:end_idx]) / (end_idx - start_idx)
        avg_y = sum(center[1] for center in centers[start_idx:end_idx]) / (end_idx - start_idx)
        
        smoothed.append((int(avg_x), int(avg_y)))
    
    return smoothed

def smooth_confidence_tracking(confidences, window_size=3):
    """
    Apply moving average smoothing to confidence scores
    """
    if not confidences:
        return []
    
    smoothed = []
    for i in range(len(confidences)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(confidences), i + window_size // 2 + 1)
        
        # Calculate average confidence
        avg_conf = sum(confidences[start_idx:end_idx]) / (end_idx - start_idx)
        smoothed.append(avg_conf)
    
    return smoothed

def preview_face_crop_with_confidence(input_video_path, output_size=512, zoom_factor=1.5, max_preview_time=30):
    """
    Preview function to see face cropping with confidence visualization in real-time
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(input_video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(max_preview_time * fps) if max_preview_time else float('inf')
    
    print("Preview mode with confidence - Press 'q' to quit, 'space' to pause/resume")
    print("Left window: original, Right window: cropped square with confidence")
    
    frame_count = 0
    paused = False
    
    while frame_count < max_frames:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
        
        if 'frame' in locals():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with confidence
            faces, reject_levels, level_weights = face_cascade.detectMultiScale3(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=3,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
                outputRejectLevels=True
            )
            
            # Show original with detection
            original_display = frame.copy()
            
            confidence = 0.0
            if len(faces) > 0:
                # Get confidence for best face
                if len(level_weights) > 0:
                    confidence = min(1.0, max(0.0, level_weights[0] / 10.0))
                
                # Get largest face
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                x, y, w, h = largest_face
                
                # Draw rectangle with confidence color
                rect_color = get_confidence_color(confidence)
                cv2.rectangle(original_display, (x, y), (x+w, y+h), rect_color, 2)
                
                # Calculate center and crop area
                center_x, center_y = x + w // 2, y + h // 2
                face_size = max(w, h)
                crop_size = int(face_size * zoom_factor)
                crop_size = max(crop_size, output_size // 4)
                
                # Calculate crop boundaries
                crop_x1 = max(0, center_x - crop_size // 2)
                crop_y1 = max(0, center_y - crop_size // 2)
                crop_x2 = min(width, crop_x1 + crop_size)
                crop_y2 = min(height, crop_y1 + crop_size)
                
                # Show crop area on original
                cv2.rectangle(original_display, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 255), 1)
                
                # Create cropped square version with confidence
                cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                if cropped_frame.size > 0:
                    square_frame = resize_to_square(cropped_frame, output_size)
                    square_frame = add_confidence_visualization(square_frame, confidence, True, output_size)
                    cv2.imshow('Cropped Square with Confidence', square_frame)
            
            # Add info to original
            cv2.putText(original_display, f'Faces: {len(faces)} | Conf: {confidence:.3f}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, get_confidence_color(confidence), 2)
            cv2.putText(original_display, f'Frame: {frame_count}', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Original with Confidence Detection', original_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space bar to pause/resume
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    input_video = 'CINA.mp4'
    output_video = 'CINA_faces_square_confidence.mp4'
    
    # Uncomment to preview face cropping with confidence first
    # preview_face_crop_with_confidence(input_video, output_size=512, zoom_factor=1.5, max_preview_time=30)
    
    # Process the video with square face-focused output and confidence visualization
    detect_and_crop_faces(
        input_video_path=input_video,
        output_video_path=output_video,
        min_segment_duration=2.0,    # Minimum 2 seconds per segment
        buffer_time=0.5,             # 0.5 seconds buffer before/after face detection
        output_size=512,             # 512x512 square output
        zoom_factor=1.5,             # 1.5x zoom on detected face
        show_confidence=True         # Show confidence visualization
    )