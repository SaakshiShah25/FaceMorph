# Run the following commands in terminal before running the code
# pip install torch torchvision opencv-python dlib matplotlib pillow numpy
# curl -L -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
# install brew on MAC
# brew install cmake

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import dlib
import os
from PIL import Image
import matplotlib.pyplot as plt

class FaceMorpher:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")

        # Initialize face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()

        # Download and load the shape predictor model if it doesn't exist
        # pre-trained model for facial landmark detection
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            print("Please download the shape predictor from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print("Extract and place it in the working directory")
            raise FileNotFoundError("shape_predictor_68_face_landmarks.dat not found")

        self.predictor = dlib.shape_predictor(predictor_path)

        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

       
        self.feature_list = {'jawline','nose','mouth','eyes','eyebrows'}
        # Define facial feature regions (landmarks indices) -> 68 landmarks
        self.features = {
            'jawline': list(range(0, 17)),
            'right_eyebrow': list(range(17, 22)),
            'left_eyebrow': list(range(22, 27)),
            'eyebrows' : list(range(17, 27)),
            'nose': list(range(27, 36)),
            'right_eye': list(range(36, 42)),
            'left_eye': list(range(42, 48)),
            'eyes': list(range(36, 48)),
            'mouth': list(range(48, 68)),
        }

    def get_landmarks(self, image):
        """Extract facial landmarks from an image."""
        # The face detector works better with grayscale images
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect faces
        # It scans the grayscale image and finds rectangular regions containing faces
        # reactangle = 4 coordinates -> left,top,right,bottom
        faces = self.detector(gray)
        if len(faces) == 0:
            print("No faces found in the image!")
            return None

        # Get the first face -> face rectangle
        face = faces[0]

        # Get facial landmarks
        landmarks = self.predictor(gray, face) # returns 68 landmarks

        # Convert landmarks to numpy array
        landmarks_np = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            landmarks_np[i] = (landmarks.part(i).x, landmarks.part(i).y) # x and y co-ordinate of each landmark

        return landmarks_np # return numpy array of 68 landmarks

    def extract_feature_mask(self, image, landmarks, feature_name):
        """Extract a binary mask for a specific facial feature."""
        # desired feature is represented in white(255) while others in black(0)
        if feature_name not in self.feature_list:
            raise ValueError(f"Feature {feature_name} not supported")

        # Create a blank mask -> its size is same as that of the image
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # For eyebrows
        if feature_name == 'eyebrows':
        # separately fill the polygon for left eyebrow and right eyebrow
          left_points = landmarks[self.features['left_eyebrow']]
          cv2.fillConvexPoly(mask, left_points, 255)
          right_points = landmarks[self.features['right_eyebrow']]
          cv2.fillConvexPoly(mask, right_points, 255)

        # For eyes
        elif feature_name == 'eyes':
         # separately fill the polygon for left eye and right eye
          left_points = landmarks[self.features['left_eye']]
          cv2.fillConvexPoly(mask, left_points, 255)
          right_points = landmarks[self.features['right_eye']]
          cv2.fillConvexPoly(mask, right_points, 255)

        else:
          # Get the indices for the specified feature -> eg: For right eye:[36,37,38,39,40,41]
          indices = self.features[feature_name]

          # extract landmarks for the given indices
          # Create a polygon from the landmarks
          # eg : [(100, 120), (110, 130), (115, 140),...]
          points = landmarks[indices]

          # Fill the polygon on the mask
          # points that define the polygon are filled with white(255) rest with black(0)
          cv2.fillConvexPoly(mask, points, 255)

        return mask


    def align_faces(self, source_img, target_img, source_landmarks, target_landmarks):
        """Align the source face (celebrity) to match the target face (person) orientation and size."""
        # Points 36 & 39: Corners of the right eye
        # Points 42 & 45: Corners of the left eye
        # Point 33: Tip of the nose
        # Points 48 & 54: Corners of the mouth

        alignment_points_src = source_landmarks[[36, 39, 42, 45, 33, 48, 54]]
        alignment_points_dst = target_landmarks[[36, 39, 42, 45, 33, 48, 54]]

        # Calculate the transformation matrix (2*3 matrix)
        # How much rotation,scale and translation must be done in source image so the 7 landmarks of the source image align with same landmarks of target image
        transform_matrix = cv2.estimateAffinePartial2D(alignment_points_src, alignment_points_dst)[0]

        # Apply the transformation
        aligned_source = cv2.warpAffine(source_img,
                                        transform_matrix,
                                        (target_img.shape[1], target_img.shape[0]),
                                        borderMode=cv2.BORDER_REPLICATE)

        # aligned version of the source image
        return aligned_source, transform_matrix

    def morph_feature(self, source_img, target_img, feature_name, alpha):
        # Get landmarks for both images
        source_landmarks = self.get_landmarks(source_img)
        target_landmarks = self.get_landmarks(target_img)

        if source_landmarks is None or target_landmarks is None:
            print("Could not detect landmarks in one or both images")
            return None

        # Align source face with target face
        aligned_source, transform_matrix = self.align_faces(source_img, target_img, source_landmarks, target_landmarks)

        # Apply the same transformation that was applied to source image to the source landmarks
        aligned_source_landmarks = np.zeros_like(source_landmarks)
        for i in range(len(source_landmarks)):
            x, y = source_landmarks[i]
            aligned_source_landmarks[i] = [
                transform_matrix[0, 0] * x + transform_matrix[0, 1] * y + transform_matrix[0, 2], # x co-ordinate
                transform_matrix[1, 0] * x + transform_matrix[1, 1] * y + transform_matrix[1, 2] # y co-ordinate
            ]

        # Extract feature mask from aligned source
        source_feature_mask = self.extract_feature_mask(aligned_source, aligned_source_landmarks, feature_name)

        # Extract same feature mask from target (for smooth blending)
        target_feature_mask = self.extract_feature_mask(target_img, target_landmarks, feature_name)

        # bitwise AND operation
        overlap_mask = cv2.bitwise_and(source_feature_mask, target_feature_mask)
        # normalize the overlap_mask from 0-255 to 0-1
        # GaussianBlur is applied to normalized mask to smooth the edges of the mask
        blended_mask = cv2.GaussianBlur(overlap_mask.astype(np.float32) / 255.0, (31, 31), 11) # normalization
        # alpha tells us how much we want to retain from the source image at each pixel
        # eg alpha = 0.7 => retain 70% of the source feature
        blended_mask = (blended_mask * alpha).astype(np.float32) 


        # Start with the target image
        result = target_img.copy()

        for c in range(3):  # For each pixel of each color channel blend the results
            # blended_mask = 1 -> retain only source image pixel value denoted by value of alpha (if alpha=0.8 then 80% source is retained)
            # blended_mask = 0 -> retain only target image pixel value -> it's 0 for areas apart from the outlined feature
            result[:, :, c] = (1 - blended_mask) * target_img[:, :, c] + blended_mask * aligned_source[:, :, c]

        # ensure pixel values range is valid, convert datatype back to uint8
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result, source_feature_mask, target_feature_mask, blended_mask
