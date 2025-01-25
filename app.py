# import os
# import cv2
# import face_recognition
# import tempfile
# import streamlit as st
# import hashlib
# import numpy as np
# from PIL import Image
# import imagehash  # Ensure this library is imported
# import subprocess
# from video_processing import detect_fake_video
# import os


# # Video frame extraction function
# def frame_extract(path):
#     """Extract frames from a video file."""
#     vidObj = cv2.VideoCapture(path)
#     success = True
#     while success:
#         success, image = vidObj.read()
#         if success:
#             yield image

# # Function to create face-cropped videos
# def create_face_videos(file_path, out_dir):
#     """Process a video file and create a face-cropped video."""
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)

#     out_path = os.path.join(out_dir, "processed_video.mp4")
#     frames = []
#     out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (112, 112))

#     for idx, frame in enumerate(frame_extract(file_path)):
#         if idx <= 150:  # Limit frames for faster processing
#             frames.append(frame)
#             if len(frames) == 4:  # Process frames in batches of 4
#                 all_faces = []
#                 for frm in frames:
#                     rgb_frame = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
#                     face_locations = face_recognition.face_locations(rgb_frame)
#                     all_faces.extend(face_locations)

#                 if not all_faces:
#                     print("No faces detected in frames")

#                 for (top, right, bottom, left) in all_faces:
#                     for i in range(len(frames)):
#                         try:
#                             face_image = frames[i][top:bottom, left:right]
#                             if face_image.size == 0:
#                                 continue
#                             face_image = cv2.resize(face_image, (112, 112))

#                             # Write the face-cropped frame to video
#                             out.write(face_image)
#                         except Exception as e:
#                             print(f"Error processing frame {i}: {e}")
#                 frames = []

#     out.release()
#     if os.path.exists(out_path):
#         print(f"Finished processing video: {out_path}")
#     else:
#         print(f"Failed to save processed video: {out_path}")

#     return out_path

# # Extract background hashes function
# def extract_background_hashes(video_path, out_dir, max_frames=5):
#     """Extract background hashes from a video, processing only a limited number of frames."""
#     print(f"Extracting background hashes from {video_path}...")
#     cap = cv2.VideoCapture(video_path)
#     background_hashes = []
#     frame_count = 0

#     # Create a background subtractor object
#     backSub = cv2.createBackgroundSubtractorMOG2()

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         # Apply background subtraction
#         foreground_mask = backSub.apply(frame)
#         background = cv2.bitwise_and(frame, frame, mask=~foreground_mask)

#         # Convert background to PIL Image and compute hash
#         background_pil = Image.fromarray(background)
#         background_hash = imagehash.phash(background_pil)
#         background_hashes.append(background_hash)

#         if frame_count % 100 == 0:
#             print(f"Processed {frame_count} frames")

#         # Stop if we've processed the maximum number of frames
#         if len(background_hashes) >= max_frames:
#             break

#     cap.release()
#     print(f"Extracted {len(background_hashes)} background hashes from {video_path}.")

#     # Save background hashes to a text file
#     with open(os.path.join(out_dir, "background_hashes.txt"), 'w') as bg_log:
#         for bg_hash in background_hashes:
#             bg_log.write(str(bg_hash) + '\n')

# # Extract face hashes function
# def extract_face_hashes(video_path, out_dir, max_frames=5):
#     """Extract face hashes from a video, processing only a limited number of frames."""
#     print(f"Extracting face hashes from {video_path}...")
#     cap = cv2.VideoCapture(video_path)
#     face_hashes = []
#     frame_count = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         # Detect faces in the frame
#         face_locations = face_recognition.face_locations(frame)

#         if face_locations:
#             print(f"Found {len(face_locations)} faces in frame {frame_count}")

#         for face_location in face_locations:
#             top, right, bottom, left = face_location
#             face_image = frame[top:bottom, left:right]
#             face_pil = Image.fromarray(face_image)
#             face_hash = imagehash.phash(face_pil)
#             face_hashes.append(face_hash)

#         if frame_count % 100 == 0:
#             print(f"Processed {frame_count} frames")

#         # Stop if we've processed the maximum number of frames
#         if len(face_hashes) >= max_frames:
#             break

#     cap.release()
#     print(f"Extracted {len(face_hashes)} face hashes from {video_path}.")

#     # Save face hashes to a text file
#     with open(os.path.join(out_dir, "face_hashes.txt"), 'w') as face_log:
#         for face_hash in face_hashes:
#             face_log.write(str(face_hash) + '\n')

# # Function to extract metadata using ffmpeg
# def extract_metadata(video_path, out_dir):
#     """Extract metadata from a video file using FFmpeg and save it to a text file."""
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)

#     metadata_file_path = os.path.join(out_dir, "metadata.txt")
    
#     command = [
#         'ffmpeg',
#         '-i', video_path,
#         '-f', 'ffmetadata',
#         'pipe:1'
#     ]

#     # Execute the command and capture output
#     result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
#     # Check for errors in the stderr
#     if result.returncode != 0:
#         st.error("Error extracting metadata.")
#         return None

#     # The metadata is returned in the stdout
#     metadata = result.stdout

#     # Write metadata to a text file
#     with open(metadata_file_path, 'w') as metadata_file:
#         metadata_file.write(metadata)
    
#     return metadata_file_path

# # Example usage: Call the function with a prediction (label, confidence, additional info)
# # upload_hashes((1, 95.5, None))  # Example call for a real video


# # Streamlit main function for interface
# def main():
#     st.title("Face Detection and Video Processing with Hashing")
#     if "processed_video_path" not in st.session_state:
#         st.session_state.processed_video_path = None

#     uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

#     if uploaded_file is not None:
#         with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#             tmp_file.write(uploaded_file.read())
#             video_path = tmp_file.name

#         # Process the video and display results
#         if st.button("Analyze"):
#             st.write("Processing video, detecting faces, and generating hashes...")

#             processed_video_path = create_face_videos(video_path, 'Processed_Videos/')
#             st.session_state.processed_video_path = processed_video_path  # Store the processed video path

#             st.write(f"Processed video saved at: {processed_video_path}")

#             # Extract background and face hashes
#             background_count = extract_background_hashes(processed_video_path, 'Processed_Videos/', max_frames=20)
#             st.write(f"Extracted background hashes and saved to 'Processed_Videos/background_hashes.txt'.")

#             face_count = extract_face_hashes(processed_video_path, 'Processed_Videos/', max_frames=20)
#             st.write(f"Extracted face hashes and saved to 'Processed_Videos/face_hashes.txt'.")

#             # metadata_file_path = extract_metadata(st.session_state.processed_video_path, 'Processed_Videos/')
#             # st.write(f"Metadata extracted and saved to {metadata_file_path}")
#             # Optionally, display the first frame
#             cap = cv2.VideoCapture(processed_video_path)
#             ret, frame = cap.read()
#             if ret:
#                 st.image(frame, caption="First frame of the processed video")
#             cap.release()
#             prediction = detect_fake_video(st.session_state.processed_video_path)
#             output = "REAL" if prediction[0] == 1 else "FAKE"
#             confidence = prediction[1]
#             st.write(f"Prediction: {output} with {confidence:.2f}% confidence")
#             cap.release()
           
               
# if __name__ == "__main__":
#     main()


#BETTER CODE DO NOT DELETE
# import os
# import cv2
# import face_recognition
# import tempfile
# import subprocess
# from flask import Flask, request, render_template
# from video_processing import detect_fake_video

# app = Flask(__name__)

# # Create the output directory if it doesn't exist
# os.makedirs('static/Processed_Videos', exist_ok=True)

# # Video frame extraction function
# def frame_extract(path):
#     """Extract frames from a video file."""
#     vidObj = cv2.VideoCapture(path)
#     success = True
#     while success:
#         success, image = vidObj.read()
#         if success:
#             yield image

# # Function to create face-cropped videos
# def create_face_videos(file_path):
#     """Process a video file and create a face-cropped video."""
#     out_path = os.path.join('static/Processed_Videos', "processed_video.mp4")
#     frames = []
#     out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (112, 112))

#     for idx, frame in enumerate(frame_extract(file_path)):
#         if idx <= 150:  # Limit frames for faster processing
#             frames.append(frame)
#             if len(frames) == 4:  # Process frames in batches of 4
#                 all_faces = []
#                 for frm in frames:
#                     rgb_frame = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
#                     face_locations = face_recognition.face_locations(rgb_frame)
#                     all_faces.extend(face_locations)

#                 if not all_faces:
#                     print("No faces detected in frames")

#                 for (top, right, bottom, left) in all_faces:
#                     for i in range(len(frames)):
#                         try:
#                             face_image = frames[i][top:bottom, left:right]
#                             if face_image.size == 0:
#                                 continue
#                             face_image = cv2.resize(face_image, (112, 112))

#                             # Write the face-cropped frame to video
#                             out.write(face_image)
#                         except Exception as e:
#                             print(f"Error processing frame {i}: {e}")
#                 frames = []

#     out.release()
#     return out_path

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'video_file' not in request.files:
#         return "No file uploaded.", 400

#     video_file = request.files['video_file']
#     if video_file.filename == '':
#         return "No file selected.", 400

#     # Save the uploaded video file to a temporary location
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
#         video_file.save(tmp_file.name)
#         video_path = tmp_file.name

#     # Process the video
#     processed_video_path = create_face_videos(video_path)

#     # Detect fake video
#     prediction = detect_fake_video(processed_video_path)
#     output = "REAL" if prediction[0] == 1 else "FAKE"
#     confidence = prediction[1]

#     return render_template('result.html', 
#                            prediction=output, 
#                            confidence=confidence,
#                            processed_video_path=processed_video_path)

# if __name__ == "__main__":
#     app.run(debug=True)
import os
import cv2
import face_recognition
import tempfile
import subprocess
import imagehash
from PIL import Image
from flask import Flask, request, render_template
from video_processing import detect_fake_video

from pymediainfo import MediaInfo

app = Flask(__name__)

# Create output directories if they don't exist
os.makedirs('static/Processed_Videos', exist_ok=True)

def frame_extract(path):
    """Extract frames from a video file."""
    vidObj = cv2.VideoCapture(path)
    success = True
    while success:
        success, image = vidObj.read()
        if success:
            yield image

def create_face_videos(file_path):
    """Process a video file and create a face-cropped video."""
    out_path = os.path.join('static/Processed_Videos', "processed_video.mp4")
    frames = []
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (112, 112))

    for idx, frame in enumerate(frame_extract(file_path)):
        if idx <= 150:  # Limit frames for faster processing
            frames.append(frame)
            if len(frames) == 4:  # Process frames in batches of 4
                all_faces = []
                for frm in frames:
                    rgb_frame = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    all_faces.extend(face_locations)

                if not all_faces:
                    print("No faces detected in frames")

                for (top, right, bottom, left) in all_faces:
                    for i in range(len(frames)):
                        try:
                            face_image = frames[i][top:bottom, left:right]
                            if face_image.size == 0:
                                continue
                            face_image = cv2.resize(face_image, (112, 112))
                            out.write(face_image)  # Write the face-cropped frame to video
                        except Exception as e:
                            print(f"Error processing frame {i}: {e}")
                frames = []

    out.release()
    return out_path

def extract_background_hashes(video_path, out_dir, max_frames=20):
    """Extract background hashes from a video."""
    print(f"Extracting background hashes from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    background_hashes = []
    frame_count = 0
    backSub = cv2.createBackgroundSubtractorMOG2()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        foreground_mask = backSub.apply(frame)
        background = cv2.bitwise_and(frame, frame, mask=~foreground_mask)
        background_pil = Image.fromarray(background)
        background_hash = imagehash.phash(background_pil)
        background_hashes.append(background_hash)

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")

        if len(background_hashes) >= max_frames:
            break

    cap.release()
    print(f"Extracted {len(background_hashes)} background hashes.")

    with open(os.path.join(out_dir, "background_hashes.txt"), 'w') as bg_log:
        for bg_hash in background_hashes:
            bg_log.write(str(bg_hash) + '\n')

def extract_face_hashes(video_path, out_dir, max_frames=20):
    """Extract face hashes from a video."""
    print(f"Extracting face hashes from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    face_hashes = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        face_locations = face_recognition.face_locations(frame)

        if face_locations:
            print(f"Found {len(face_locations)} faces in frame {frame_count}")

        for top, right, bottom, left in face_locations:
            face_image = frame[top:bottom, left:right]
            face_pil = Image.fromarray(face_image)
            face_hash = imagehash.phash(face_pil)
            face_hashes.append(face_hash)

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")

        if len(face_hashes) >= max_frames:
            break

    cap.release()
    print(f"Extracted {len(face_hashes)} face hashes.")

    with open(os.path.join(out_dir, "face_hashes.txt"), 'w') as face_log:
        for face_hash in face_hashes:
            face_log.write(str(face_hash) + '\n')
    
def extract_first_frame(video_path):
    # Load the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    
    success, frame = cap.read()  # Read the first frame
    if success:
        # Save the first frame as an image
        first_frame_path = 'static/first_frame.jpg'  # Save in static folder
        cv2.imwrite(first_frame_path, frame)
        cap.release()
        return first_frame_path
    else:
        cap.release()
        raise Exception("Could not extract the first frame from the video.")
def extract_video_metadata(video_path):
    """Extract metadata from a video file using pymediainfo."""
    media_info = MediaInfo.parse(video_path)
    metadata = {}
    
    for track in media_info.tracks:
        if track.track_type == "General":
            for key, value in track.to_data().items():
                if value:  # Only store non-empty values
                    metadata[key] = value
    
    return metadata

def save_metadata(metadata, out_dir):
    """Save metadata to a file."""
    metadata_path = os.path.join(out_dir, "video_metadata.txt")
    with open(metadata_path, 'w') as metadata_file:
        for key, value in metadata.items():
            metadata_file.write(f"{key}: {value}\n")
    return metadata_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video_file' not in request.files:
        return "No file uploaded.", 400

    video_file = request.files['video_file']
    if video_file.filename == '':
        return "No file selected.", 400

    # Save the uploaded video to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        video_file.save(tmp_file.name)
        video_path = tmp_file.name

    # Create an output directory for the video hashes
    video_hash_dir = os.path.join('static/Processed_Videos', 'hashes')
    os.makedirs(video_hash_dir, exist_ok=True)

    # Extract background and face hashes
    extract_background_hashes(video_path, video_hash_dir)
    extract_face_hashes(video_path, video_hash_dir)

    # Process the video
    processed_video_path = create_face_videos(video_path)

    # Detect if the video is fake or real
    prediction = detect_fake_video(processed_video_path)
    print(prediction)
    output = "REAL" if prediction[0] == 1 else "FAKE"
    confidence = prediction[1]
    face_hashes_path = os.path.join(video_hash_dir, 'face_hashes.txt')
    background_hashes_path = os.path.join(video_hash_dir, 'background_hashes.txt')

    # Read face hashes
    with open(face_hashes_path, 'r') as file:
        face_hashes = [line.strip() for line in file.readlines()]

    # Read background hashes
    with open(background_hashes_path, 'r') as file:
        background_hashes = [line.strip() for line in file.readlines()]
    print(face_hashes)
    print(background_hashes)
    first_frame_path = extract_first_frame(video_path)
    metadata = extract_video_metadata(video_path)
    metadata_path = save_metadata(metadata, video_hash_dir)
    with open(metadata_path, 'r') as file:
        metadata = [line.strip() for line in file.readlines()]


    return render_template('result.html',
                           prediction=output,
                           confidence=confidence,
                           processed_video_path=processed_video_path,
                           face_hashes=face_hashes,
                           background_hashes=background_hashes,
                           first_frame_path=first_frame_path)

    # return render_template('result.html', 
    #                        prediction=output, 
    #                        confidence=confidence,
    #                        processed_video_path=processed_video_path)

if __name__ == "__main__":
    app.run(debug=True)
# import os
# import cv2
# import face_recognition
# import tempfile
# import subprocess
# import imagehash
# from PIL import Image
# from flask import Flask, request, render_template
# from video_processing import detect_fake_video
# import exifread
# import hashlib
# from pymediainfo import MediaInfo
# import json


# app = Flask(__name__)

# # Create output directories if they don't exist
# os.makedirs('static/Processed_Videos', exist_ok=True)

# def hash_string(data):
#     """Generate SHA256 hash of a string."""
#     return hashlib.sha256(data.encode()).hexdigest()

# def extract_exif_data(video_path):
#     """Extract EXIF data from a video file."""
#     with open(video_path, 'rb') as f:
#         tags = exifread.process_file(f)
#     return tags

# def frame_extract(path):
#     """Extract frames from a video file."""
#     vidObj = cv2.VideoCapture(path)
#     success = True
#     while success:
#         success, image = vidObj.read()
#         if success:
#             yield image

# def create_face_videos(file_path):
#     """Process a video file and create a face-cropped video."""
#     out_path = os.path.join('static/Processed_Videos', "processed_video.mp4")
#     frames = []
#     out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (112, 112))

#     for idx, frame in enumerate(frame_extract(file_path)):
#         if idx <= 150:  # Limit frames for faster processing
#             frames.append(frame)
#             if len(frames) == 4:  # Process frames in batches of 4
#                 all_faces = []
#                 for frm in frames:
#                     rgb_frame = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
#                     face_locations = face_recognition.face_locations(rgb_frame)
#                     all_faces.extend(face_locations)

#                 if not all_faces:
#                     print("No faces detected in frames")

#                 for (top, right, bottom, left) in all_faces:
#                     for i in range(len(frames)):
#                         try:
#                             face_image = frames[i][top:bottom, left:right]
#                             if face_image.size == 0:
#                                 continue
#                             face_image = cv2.resize(face_image, (112, 112))
#                             out.write(face_image)  # Write the face-cropped frame to video
#                         except Exception as e:
#                             print(f"Error processing frame {i}: {e}")
#                 frames = []

#     out.release()
#     return out_path

# def extract_background_hashes(video_path, out_dir, max_frames=20):
#     """Extract background hashes from a video."""
#     print(f"Extracting background hashes from {video_path}...")
#     cap = cv2.VideoCapture(video_path)
#     background_hashes = []
#     frame_count = 0
#     backSub = cv2.createBackgroundSubtractorMOG2()

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         foreground_mask = backSub.apply(frame)
#         background = cv2.bitwise_and(frame, frame, mask=~foreground_mask)
#         background_pil = Image.fromarray(background)
#         background_hash = imagehash.phash(background_pil)
#         background_hashes.append(background_hash)

#         if frame_count % 100 == 0:
#             print(f"Processed {frame_count} frames")

#         if len(background_hashes) >= max_frames:
#             break

#     cap.release()
#     print(f"Extracted {len(background_hashes)} background hashes.")

#     with open(os.path.join(out_dir, "background_hashes.txt"), 'w') as bg_log:
#         for bg_hash in background_hashes:
#             bg_log.write(str(bg_hash) + '\n')

# def extract_face_hashes(video_path, out_dir, max_frames=20):
#     """Extract face hashes from a video."""
#     print(f"Extracting face hashes from {video_path}...")
#     cap = cv2.VideoCapture(video_path)
#     face_hashes = []
#     frame_count = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         face_locations = face_recognition.face_locations(frame)

#         if face_locations:
#             print(f"Found {len(face_locations)} faces in frame {frame_count}")

#         for top, right, bottom, left in face_locations:
#             face_image = frame[top:bottom, left:right]
#             face_pil = Image.fromarray(face_image)
#             face_hash = imagehash.phash(face_pil)
#             face_hashes.append(face_hash)

#         if frame_count % 100 == 0:
#             print(f"Processed {frame_count} frames")

#         if len(face_hashes) >= max_frames:
#             break

#     cap.release()
#     print(f"Extracted {len(face_hashes)} face hashes.")

#     with open(os.path.join(out_dir, "face_hashes.txt"), 'w') as face_log:
#         for face_hash in face_hashes:
#             face_log.write(str(face_hash) + '\n')

# def extract_first_frame(video_path):
#     # Load the video using OpenCV
#     cap = cv2.VideoCapture(video_path)

#     success, frame = cap.read()  # Read the first frame
#     if success:
#         # Save the first frame as an image
#         first_frame_path = 'static/first_frame.jpg'  # Save in static folder
#         cv2.imwrite(first_frame_path, frame)
#         cap.release()
#         return first_frame_path
#     else:
#         cap.release()
#         raise Exception("Could not extract the first frame from the video.")

# def extract_video_metadata(video_path):
#     """Extract metadata from a video file using pymediainfo."""
#     media_info = MediaInfo.parse(video_path)
#     metadata = {}
    
#     for track in media_info.tracks:
#         if track.track_type == "General":
#             for key, value in track.to_data().items():
#                 if value:  # Only store non-empty values
#                     metadata[key] = value
    
#     return metadata

# def save_metadata(metadata, out_dir):
#     """Save metadata to a file."""
#     metadata_path = os.path.join(out_dir, "video_metadata.txt")
#     with open(metadata_path, 'w') as metadata_file:
#         for key, value in metadata.items():
#             metadata_file.write(f"{key}: {value}\n")
#     return metadata_path
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'video_file' not in request.files:
#         return "No file uploaded.", 400

#     video_file = request.files['video_file']
#     if video_file.filename == '':
#         return "No file selected.", 400

#     # Save the uploaded video to a temporary location
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
#         video_file.save(tmp_file.name)
#         video_path = tmp_file.name

#     # Create an output directory for the video hashes
#     video_hash_dir = os.path.join('static/Processed_Videos', 'hashes')
#     os.makedirs(video_hash_dir, exist_ok=True)

#     # Extract video metadata
#     metadata = extract_video_metadata(video_path)
    
#     # Save metadata to a file
#     metadata_path = save_metadata(metadata, video_hash_dir)

#     # Extract background and face hashes
#     extract_background_hashes(video_path, video_hash_dir)
#     extract_face_hashes(video_path, video_hash_dir)

#     # Process the video
#     processed_video_path = create_face_videos(video_path)

#     # Detect if the video is fake or real
#     prediction = detect_fake_video(processed_video_path)
#     print(prediction)
#     output = "REAL" if prediction[0] == 1 else "FAKE"
#     confidence = prediction[1]
#     face_hashes_path = os.path.join(video_hash_dir, 'face_hashes.txt')
#     background_hashes_path = os.path.join(video_hash_dir, 'background_hashes.txt')
#     metadata_path = os.path.join(video_hash_dir, 'video_metadata.txt')

#     # Read face hashes
#     with open(face_hashes_path, 'r') as file:
#         face_hashes = [line.strip() for line in file.readlines()]

#     # Read background hashes
#     with open(background_hashes_path, 'r') as file:
#         background_hashes = [line.strip() for line in file.readlines()]
#     with open(metadata_path, 'r') as file:
#         metadata = [line.strip() for line in file.readlines()]
#     first_frame_path = extract_first_frame(video_path)
#     print(metadata)

#     return render_template('result.html',
#                            prediction=output,
#                            confidence=confidence,
#                            processed_video_path=processed_video_path,
#                            face_hashes=json.dumps(face_hashes),
#                            background_hashes=json.dumps(background_hashes),
#                            metadata=json.dumps(metadata),
#                            first_frame_path=first_frame_path,
#                            )  # Include the path to the metadata file in the result

# if __name__ == "__main__":
#     app.run(debug=True)