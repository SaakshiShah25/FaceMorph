# FaceMorph

# Face Feature Morphing with PyTorch, OpenCV, and Dlib

This project implements a facial **feature morphing** system using deep learning and computer vision libraries. Given two face images (e.g., your photo and a celebrity’s photo), this tool aligns the faces and allows you to selectively **morph facial features** such as eyes, nose, mouth, or eyebrows with smooth blending.

---

## 🔧 Features

- Automatic face and landmark detection using **Dlib**
- Facial alignment using **affine transformations**
- Feature masking and region-specific morphing
- Smooth blending using **Gaussian filtering**
- Supports morphing of multiple facial regions together

---

## 🖥️ Sample Features That Can Be Morphed

- `jawline`
- `nose`
- `mouth`
- `eyes`
- `eyebrows`

---

## 📦 Installation

Run the following commands in your terminal:

```bash
# Step 1: Install dependencies
pip install torch torchvision opencv-python dlib matplotlib pillow numpy

# Step 2: Download Dlib's pretrained shape predictor
curl -L -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# Step 3: [For macOS only] Install CMake using Homebrew
brew install cmake
```
---

## 🧪 How to Run the App

This project includes a **Streamlit frontend** to interactively upload images and visualize the morphing process.

## 🔃 Run with Streamlit

1. Ensure all dependencies are installed (see [Installation](#-installation)).
2. Navigate to the project directory in your terminal.
3. Run the following command:

   ```bash
   streamlit run frontend.py
   ```
---

### 🖼️ What You Can Do

- 📤 **Upload a source image**  
  *(e.g., a celebrity photo)*

- 📤 **Upload a target image**  
  *(e.g., your photo)*

- ✅ **Select facial features to morph:**
  - `eyes`
  - `nose`
  - `mouth`
  - `eyebrows`
  - `jawline`

- 🎚️ **Adjust the blending factor** using a slider  
  *(0 = keep target image, 1 = full source feature)*

- 🔍 **View the results:**
  - Original source and target images
  - The final morphed output
  - Intermediate feature masks:
    - Source feature mask
    - Target feature mask
    - Blended region mask

