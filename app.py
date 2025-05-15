import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from PIL import Image
import io
from pre_process import _reshape_img, get_model

# Set page config
st.set_page_config(
    page_title="DeepFracture Detection System",
    page_icon="🦴",
    layout="wide"
)

# Custom CSS for a consistent dark theme
st.markdown("""
<style>
    body {
        background-color: #1a1a2e;
        color: #e0e0e0;
    }
    .title {
        font-size: 48px;
        font-weight: 700;
        color: #00d4ff;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    .subtitle {
        font-size: 24px;
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 30px;
        font-style: italic;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff, #007bff);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 12px 30px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #007bff, #00d4ff);
        box-shadow: 0 0 15px rgba(0, 123, 255, 0.5);
    }
    .stSidebar {
        background-color: #1f1f33;
        border-right: 1px solid #00d4ff;
    }
    .stSidebar .stRadio, .stSidebar .stSelectbox, .stSidebar .stFileUploader {
        background-color: #252537;
        color: #e0e0e0;
        border-radius: 8px;
        padding: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #252537;
        color: #e0e0e0;
        border-radius: 8px;
        margin: 5px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #00d4ff;
        color: #1a1a2e;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #00d4ff;
        color: #1a1a2e;
    }
    .stMetric {
        background-color: #252537;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.2);
        color: #e0e0e0;
    }
    .stMetric label {
        color: #a0a0a0;
    }
    .stExpander {
        background-color: #252537;
        border-radius: 8px;
        border: 1px solid #00d4ff;
        color: #e0e0e0;
    }
    .stExpander [data-baseweb="accordion"] {
        background-color: #252537;
        color: #e0e0e0;
    }
    .stSpinner > div {
        color: #00d4ff;
    }
    .stAlert {
        background-color: #252537;
        color: #e0e0e0;
        border: 1px solid #00d4ff;
        border-radius: 8px;
    }
    .footer {
        text-align: center;
        color: #a0a0a0;
        margin-top: 40px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<p class="title">DeepFracture Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Bone Fracture Analysis</p>', unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        model = get_model("ridge_model")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Function to process image
def process_image(img):
    # Store the original image for display
    original_img = img.copy()
    
    # Get image details
    shape = img.shape
    size = img.size
    dtype = img.dtype
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur
    median = cv2.medianBlur(gray, 5)
    
    # Resize image for model prediction
    img_resized = cv2.resize(img, (int(310/2), int(568/2)))
    
    # Predict threshold using the model
    pred_thresh = model.predict([_reshape_img(img_resized)])
    
    # Apply threshold
    bool_val, threshold_img = cv2.threshold(median, int(pred_thresh), 255, cv2.THRESH_BINARY)
    
    # Find potential fracture points
    initial = []
    final = []
    line = []
    dist_list = []
    
    # Find edges in the threshold image
    for i in range(0, gray.shape[0]):
        tmp_initial = []
        tmp_final = []
        for j in range(0, gray.shape[1]-1):
            if threshold_img[i, j] == 0 and threshold_img[i, j+1] == 255:
                tmp_initial.append((i, j))
            if threshold_img[i, j] == 255 and threshold_img[i, j+1] == 0:
                tmp_final.append((i, j))
        
        x = [each for each in zip(tmp_initial, tmp_final)]
        x.sort(key=lambda each: each[1][1] - each[0][1])
        try:
            line.append(x[len(x)-1])
        except IndexError:
            pass
    
    # Analyze distances for potential fractures
    err = 15
    danger_points = []
    
    for i in range(1, len(line)-1):
        if i < len(line) and len(line[i]) == 2:
            dist_list.append(line[i][1][1] - line[i][0][1])
        try:
            prev_ = line[i-3]
            next_ = line[i+3]
            
            dist_prev = prev_[1][1] - prev_[0][1]
            dist_next = next_[1][1] - next_[0][1]
            diff = abs(dist_next - dist_prev)
            
            if diff > err:
                data = (diff, line[i])
                if len(danger_points):
                    prev_data = danger_points[len(danger_points)-1]
                    if abs(prev_data[0] - data[0]) > 2 or data[1][0][0] - prev_data[1][0][0] != 1:
                        danger_points.append(data)
                else:
                    danger_points.append(data)
        except Exception:
            pass
    
    # Draw rectangles around potential fractures
    result_img = original_img.copy()
    
    for i in range(0, len(danger_points)-1, 2):
        try:
            start_rect = danger_points[i][1][0][::-1]
            start_rect = (start_rect[0]-40, start_rect[1]-40)
            
            end_rect = danger_points[i+1][1][1][::-1]
            end_rect = (end_rect[0]+40, end_rect[1]+40)
            
            cv2.rectangle(result_img, start_rect, end_rect, (0, 255, 0), 2)
        except:
            pass
    
    return {
        "original": original_img,
        "grayscale": gray,
        "threshold": threshold_img,
        "result": result_img,
        "shape": shape,
        "size": size,
        "dtype": dtype,
        "dist_list": dist_list,
        "pred_thresh": pred_thresh[0],
        "danger_points": danger_points
    }

# Main app
st.sidebar.header("System Controls")

# Option to use sample image or upload your own
option = st.sidebar.radio("Select Input Source", ["Upload Image", "Use Sample Image"])

image_file = None
if option == "Upload Image":
    image_file = st.sidebar.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])
else:
    # Use a sample image
    sample_options = ["F1.jpg", "F2.jpg", "F4.jpg", "F6.jpg"]
    selected_sample = st.sidebar.selectbox("Choose Sample Image", sample_options)
    
    # Ensure the sample image path exists
    sample_path = f"images/Fractured Bone/{selected_sample}"
    if os.path.exists(sample_path):
        image_file = sample_path
    else:
        st.sidebar.error(f"Sample image not found: {sample_path}")
        st.sidebar.info("Please ensure the 'images/Fractured Bone' directory exists with sample images.")

# Process the image
if image_file is not None:
    # Display a spinner while processing
    with st.spinner("Analyzing X-ray with Deep Learning Model..."):
        try:
            # Read the image
            if isinstance(image_file, str):
                # Reading from file path (sample image)
                img = cv2.imread(image_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
            else:
                # Reading from uploaded file
                file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
                
            if img is None:
                st.error("Failed to read the image. Please try another file.")
            else:
                # Process the image
                results = process_image(img)
                
                # Display the image details
                col1, col2, col3 = st.columns(3)
                col1.metric("Image Shape", f"{results['shape'][0]}×{results['shape'][1]}×{results['shape'][2]}")
                col2.metric("Image Size", f"{results['size']} pixels")
                col3.metric("Predicted Threshold", f"{int(results['pred_thresh'])}")
                
                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4 = st.tabs(["Original Image", "Grayscale Processing", "Threshold Analysis", "Fracture Detection"])
                
                with tab1:
                    st.image(results["original"], caption="Original X-ray Image", use_container_width=True)
                
                with tab2:
                    st.image(results["grayscale"], caption="Grayscale Processed Image", use_container_width=True)
                
                with tab3:
                    st.image(results["threshold"], caption="Threshold Analysis Output", use_container_width=True)
                
                with tab4:
                    st.image(results["result"], caption="Detected Fracture Regions", use_container_width=True)
                    
                    if len(results["danger_points"]) > 0:
                        st.success(f"Potential fractures detected! Identified {len(results['danger_points'])} critical points.")
                    else:
                        st.info("No potential fractures detected in this X-ray.")
                
                # Show the distance plot
                if len(results["dist_list"]) > 0:
                    st.subheader("Bone Width Analysis")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    x = np.arange(1, len(results["dist_list"]) + 1)
                    ax.plot(x, results["dist_list"], color='#00d4ff')
                    ax.set_title("Bone Width Analysis", color='#e0e0e0')
                    ax.set_xlabel("Scan Line", color='#e0e0e0')
                    ax.set_ylabel("Bone Width (pixels)", color='#e0e0e0')
                    ax.set_facecolor('#252537')
                    fig.patch.set_facecolor('#1a1a2e')
                    ax.spines['top'].set_color('#a0a0a0')
                    ax.spines['right'].set_color('#a0a0a0')
                    ax.spines['left'].set_color('#a0a0a0')
                    ax.spines['bottom'].set_color('#a0a0a0')
                    ax.tick_params(colors='#a0a0a0')
                    ax.grid(True, color='#3a3a4a', linestyle='--', alpha=0.5)
                    st.pyplot(fig)
                    
                    # Create a histogram of pixel intensities
                    st.subheader("Pixel Intensity Distribution")
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    ax2.hist(results["grayscale"].ravel(), 256, [0, 256], color='#00d4ff')
                    ax2.set_title("Pixel Intensity Histogram", color='#e0e0e0')
                    ax2.set_xlabel("Pixel Value", color='#e0e0e0')
                    ax2.set_ylabel("Frequency", color='#e0e0e0')
                    ax2.set_facecolor('#252537')
                    fig2.patch.set_facecolor('#1a1a2e')
                    ax2.spines['top'].set_color('#a0a0a0')
                    ax2.spines['right'].set_color('#a0a0a0')
                    ax2.spines['left'].set_color('#a0a0a0')
                    ax2.spines['bottom'].set_color('#a0a0a0')
                    ax2.tick_params(colors='#a0a0a0')
                    ax2.grid(True, color='#3a3a4a', linestyle='--', alpha=0.5)
                    st.pyplot(fig2)
                
        except Exception as e:
            st.error(f"Error processing image: {e}")
else:
    # Display instructions when no image is uploaded
    st.info("Please upload an X-ray image or select a sample image to initiate analysis.")
    
    # Show a placeholder image
    placeholder_col1, placeholder_col2, placeholder_col3 = st.columns([1, 2, 1])
    with placeholder_col2:
        st.image("https://via.placeholder.com/400x300?text=X-ray+Image+Placeholder", 
                 caption="Upload an X-ray to Begin Analysis", 
                 use_container_width=True)

# Add information about the app
with st.expander("About DeepFracture System"):
    st.markdown("""
    **DeepFracture Detection System** is an AI-powered tool for analyzing X-ray images to detect potential bone fractures using advanced image processing and machine learning techniques.
    
    **How it works:**
    1. The system processes the uploaded X-ray image and converts it to grayscale for enhanced analysis.
    2. A deep learning model predicts the optimal threshold for bone structure segmentation.
    3. The thresholded image is analyzed to identify discontinuities in bone structure.
    4. Potential fracture regions are highlighted with green bounding boxes for clear visualization.
    
    **Note:** This system is intended for demonstration purposes only and is not a substitute for professional medical diagnosis.
    """)

# Add footer
st.markdown("---")
st.markdown('<p class="footer">DeepFracture Detection System | Powered by AI & Streamlit Under Your Work Innovations</p>', unsafe_allow_html=True)