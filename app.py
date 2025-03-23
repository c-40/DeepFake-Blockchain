# import os
# import cv2
# import face_recognition
# import tempfile
# import subprocess
# import imagehash
# from PIL import Image
# from flask import Flask, request, render_template
# from video_processing import detect_fake_video

# from pymediainfo import MediaInfo

# app = Flask(__name__)

# # Create output directories if they don't exist
# os.makedirs('static/Processed_Videos', exist_ok=True)
# def frame_extract(video_path):
#     """Extract frames from the video."""
#     cap = cv2.VideoCapture(video_path)
#     frames = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)

#     cap.release()
#     return frames

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

# def generate_video_hash(video_path):
#     """Generate perceptual hashes for each frame in the video."""
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return []

#     # List to store pHashes of each frame
#     frame_hashes = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break  # End of video

#         # Convert the frame to grayscale
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Convert the frame (numpy array) to a PIL image
#         pil_image = Image.fromarray(gray_frame)

#         # Generate the pHash of the frame
#         p_hash = imagehash.phash(pil_image)
#         frame_hashes.append(str(p_hash))

#     cap.release()
#     return frame_hashes

# def save_video_hashes(video_hashes, out_dir):
#     """Save the generated video hashes into a text file."""
#     hash_file_path = os.path.join(out_dir, "video_hashes.txt")
#     with open(hash_file_path, 'w') as hash_file:
#         for hash_value in video_hashes:
#             hash_file.write(f"{hash_value}\n")
#     return hash_file_path
    
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

#     # Create an output directory for the processed video if it doesn't exist
#     processed_video_dir = os.path.join('static/Processed_Videos')
#     os.makedirs(processed_video_dir, exist_ok=True)

#     # Create the face-cropped video
#     processed_video_path = create_face_videos(video_path)

#     # Generate video hashes
#     video_hashes = generate_video_hash(processed_video_path)

#     # Create an output directory for the video hashes
#     video_hash_dir = os.path.join('static/Processed_Videos', 'hashes')
#     os.makedirs(video_hash_dir, exist_ok=True)

#     # Save the generated hashes to a file
#     hash_file_path = save_video_hashes(video_hashes, video_hash_dir)

#     # Detect if the video is fake or real
#     prediction = detect_fake_video(processed_video_path)
#     print(prediction)
#     output = "REAL" if prediction[0] == 1 else "FAKE"
#     confidence = prediction[1]

#     # Return the processed video path and hashes
#     return render_template('result.html',
#                            prediction=output,
#                            confidence=confidence,
#                            processed_video_path=processed_video_path,
#                            hash_file_path=hash_file_path)


#     # return render_template('result.html', 
#     #                        prediction=output, 
#     #                        confidence=confidence,
#     #                        processed_video_path=processed_video_path)

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
from flask import render_template, request, url_for


app = Flask(__name__)

# Create output directories if they don't exist
os.makedirs('static/Processed_Videos', exist_ok=True)

def frame_extract(video_path, max_frames=20):
    """Extract up to max_frames from the video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Extract the frames up to the specified max_frames
    while True:
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        frames.append(frame)

    cap.release()
    return frames

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

def generate_video_hash(video_path, max_frames=20):
    """Generate perceptual hashes for the first max_frames frames in the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    # List to store pHashes of each frame
    frame_hashes = []

    for idx, frame in enumerate(frame_extract(video_path, max_frames=max_frames)):
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert the frame (numpy array) to a PIL image
        pil_image = Image.fromarray(gray_frame)

        # Generate the pHash of the frame
        p_hash = imagehash.phash(pil_image)
        frame_hashes.append(str(p_hash))
        print(f"Frame {idx + 1} hash: {str(p_hash)}")

    cap.release()
    return frame_hashes

def save_video_hashes(video_hashes, out_dir):
    """Save the generated video hashes into a text file."""
    hash_file_path = os.path.join(out_dir, "video_hashes.txt")
    with open(hash_file_path, 'w') as hash_file:
        for hash_value in video_hashes:
            hash_file.write(f"{hash_value}\n")
    return hash_file_path

def extract_first_frame(video_path):
    # Load the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()  # Read the first frame
    if success:
        # Save the first frame as an image
        first_frame_path = 'static/uploads/first_frame.jpg'  # Save in static folder
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
    # Check if the 'video_file' exists in the request files
    if 'video_file' not in request.files:
        return "No file uploaded.", 400

    video_file = request.files['video_file']
    

    # Check if the file has a name (i.e., it is not empty)
    if not video_file.filename:
        return "No file selected.", 400

    # Save uploaded video temporarily using a named temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        video_file.save(temp_video.name)  # Save the uploaded video to the temp file
        video_path = temp_video.name  # Store the temp video file path

    # Ensure processed video directory exists
    processed_dir = os.path.join("static", "Processed_Videos")
    os.makedirs(processed_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Process the video (e.g., cropping face regions or other processing)
    processed_video = create_face_videos(video_path)  # Process the video and get the processed path
    print(f"Processed Video Saved: {processed_video}")  # For logging purposes

    # Extract a hash for analysis from the processed video
    video_hash = generate_video_hash(processed_video)  # Generate hash from the video
    print(f"Extracted Video Hash: {video_hash}")  # For logging purposes

    # Ensure hash storage directory exists
    hash_dir = os.path.join(processed_dir, "hashes")
    os.makedirs(hash_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Save video hash if necessary
    hash_file_path = save_video_hashes(video_hash, hash_dir)  # Ensure this function returns the file path

    # Read hashes from the file
    with open(hash_file_path, 'r') as file:
        hashes_list = [line.strip() for line in file.readlines()]  # Store the hashes in a list

    # Run deepfake detection model
    prediction = detect_fake_video(processed_video)  # Run the model on the processed video
    result_label = "REAL" if prediction[0] == 1 else "FAKE"  # Label the result based on prediction
    confidence_score = prediction[1]  # Get the confidence score from the model
    
    # Extract the first frame
    first_frame_path = extract_first_frame(video_path)


    # Return the result to the front-end with the prediction and the path to the processed video
    return render_template(
        "result.html",
        prediction=result_label,
        confidence=confidence_score,
        processed_video_path=processed_video,
        hash=hashes_list,  # Send list of hashes to the template
        first_frame_path=first_frame_path  # Use the corrected variable name
    )


# @app.route('/upload', methods=['POST'])
# def upload():
#     # Check if the 'video_file' exists in the request files
#     if 'video_file' not in request.files:
#         return "No file uploaded.", 400

#     video_file = request.files['video_file']

#     # Check if the file has a name (i.e., it is not empty)
#     if not video_file.filename:
#         return "No file selected.", 400

#     # Save uploaded video temporarily using a named temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
#         video_file.save(temp_video.name)  # Save the uploaded video to the temp file
#         video_path = temp_video.name  # Store the temp video file path

#     # Ensure processed video directory exists
#     processed_dir = os.path.join("static", "Processed_Videos")
#     os.makedirs(processed_dir, exist_ok=True)  # Create the directory if it doesn't exist

#     # Process the video (e.g., cropping face regions or other processing)
#     processed_video = create_face_videos(video_path)  # Process the video and get the processed path
#     print(f"Processed Video Saved: {processed_video}")  # For logging purposes

#     # Extract a hash for analysis from the processed video
#     video_hash = generate_video_hash(processed_video)  # Generate hash from the video
#     print(f"Extracted Video Hash: {video_hash}")  # For logging purposes

#     # Ensure hash storage directory exists
#     hash_dir = os.path.join(processed_dir, "hashes")
#     os.makedirs(hash_dir, exist_ok=True)  # Create the directory if it doesn't exist

#     # Save video hash if necessary
#     hash_file_path = save_video_hashes(video_hash, hash_dir)  # Ensure this function returns the file path

#     # Read hashes from the file
#     with open(hash_file_path, 'r') as file:
#         hashes_list = [line.strip() for line in file.readlines()]  # Store the hashes in a list

#     # Run deepfake detection model
#     prediction = detect_fake_video(processed_video)  # Run the model on the processed video
#     result_label = "REAL" if prediction[0] == 1 else "FAKE"  # Label the result based on prediction
#     confidence_score = prediction[1]  # Get the confidence score from the model
#     first_frame = extract_first_frame(processed_video)

#     # Return the result to the front-end with the prediction and the path to the processed video
#     return render_template(
#         "result.html",
#         prediction=result_label,
#         confidence=confidence_score,
#         processed_video_path=processed_video,
#         hash=hashes_list,  # Send list of hashes to the template
#         first_frame_path=first_frame_path
#     )

if __name__ == "__main__":
    app.run(debug=True)
