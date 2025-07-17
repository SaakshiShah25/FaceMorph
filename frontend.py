import streamlit as st
from face_morph_multiple import FaceMorpher
from PIL import Image
import numpy as np
import io

st.title("Face Morphing Application")

uploaded_source = st.file_uploader("Upload Source Image (Celebrity)", type=['jpg', 'jpeg', 'png'])
uploaded_target = st.file_uploader("Upload Target Image", type=['jpg', 'jpeg', 'png'])
#feature = st.selectbox("Select Feature to Morph", ['mouth', 'nose', 'eyes', 'eyebrows','jawline'])
feature_list = st.multiselect("Select Features to Morph", ['mouth', 'nose', 'eyes', 'eyebrows','jawline'])

alpha = st.slider("Blending Factor", 0.0, 1.0, 0.7) # 1: same as source, 0: same as target

if uploaded_source and uploaded_target:
    # Requires images to be in RGB format
    source_img = np.array(Image.open(uploaded_source).convert("RGB"))
    target_img = np.array(Image.open(uploaded_target).convert("RGB"))

    morpher = FaceMorpher() # create an object 
    results = morpher.morph_feature(source_img, target_img, feature_list, alpha) # call the function

    if results:
        result_img, source_feature_mask, target_feature_mask, blended_mask = results
        # show the source image,target image and morphed image side by side 
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(source_img, caption="Source Image")
        with col2:
            st.image(target_img, caption="Target Image")
        with col3:
            st.image(result_img, caption="Morphed Result")

        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(source_feature_mask, caption="Source Feature Mask")
        with col2:
            st.image(target_feature_mask, caption="Target Feature Mask")
        with col3:
            st.image(blended_mask, caption="Blended Mask")