import streamlit as st
import cv2
import os
import random
import numpy as np
import tempfile
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from PIL import Image, ImageEnhance
import base64
from io import BytesIO

# Constants
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
# Enhanced Constants
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
NORMALIZATION_SCHEMES = {
    'None': {'mean': 0.0, 'std': 1.0},
    'ImageNet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
    'Grayscale': {'mean': 0.5, 'std': 0.5},
    'Custom': {'mean': 0.0, 'std': 1.0},
    'PetImages': {'mean': [0.478, 0.443, 0.401], 'std': [0.268, 0.265, 0.275]}
}

# New: Animation presets
ANIMATION_PRESETS = {
    "Gentle Pulse": "animation: pulse 3s ease-in-out infinite;",
    "Slow Float": "animation: float 6s ease-in-out infinite;",
    "Quick Vibrate": "animation: vibrate 0.5s ease infinite;",
    "Rotate": "animation: rotate 8s linear infinite;",
    "Color Shift": "animation: colorshift 10s linear infinite;"
}

# Custom CSS for premium styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Enhanced CSS with more animations and effects
def set_dark_mode():
    """Premium dark theme with working animations and proper rendering"""
def set_dark_mode():
    """Premium dark theme with combined zoom and pan background animation"""
    st.markdown(
        """
        <style>
            :root {
                --primary: #6a11cb;
                --secondary: #2575fc;
                --accent: #00c6fb;
                --surface: rgba(35, 35, 35, 0.9);
                --on-background: #e0e0e0;
                --on-surface: #ffffff;
                --card-bg: rgba(45, 45, 45, 0.9);
                --hover-bg: rgba(58, 58, 58, 0.95);
                --border-color: rgba(255, 255, 255, 0.1);
                --sidebar-bg: rgba(20, 20, 20, 0.85);
                --sidebar-hover: rgba(106, 17, 203, 0.3);
            }
            
            /* Main app container */
            .stApp {
                background: linear-gradient(
                    rgba(26, 26, 26, 0.85), 
                    rgba(26, 26, 26, 0.85)
                );
                color: var(--on-background);
            }
            
            /* Animated background with zoom and pan */
            .stApp::after {
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('https://assets.skyfilabs.com/images/blog/innovative-image-processing-based-final-year-projects-for-students.jpg') center/cover no-repeat;
                z-index: -2;
                animation: 
                    zoom 10s ease infinite,
                    pan 60s linear infinite,
                    gradient 12s ease infinite;
                opacity: 0.7;
                transform-origin: center;
                background-size: 120% 120%; /* Allows for panning movement */
            }
            
            /* Zoom animation */
            @keyframes zoom {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.1); }
            }
            
            /* Horizontal pan animation */
            @keyframes pan {
                0% { background-position: 0% 50%; }
                25% { background-position: 25% 50%; }
                50% { background-position: 50% 50%; }
                75% { background-position: 75% 50%; }
                100% { background-position: 100% 50%; }
            }
            
            /* Gradient overlay animation */
            @keyframes gradient {
                0% { background-position: 0% 0%; }
                50% { background-position: 100% 100%; }
                100% { background-position: 0% 0%; }
            }
            
            /* Sidebar styling */
            [data-testid="stSidebar"] {
                background-color: var(--sidebar-bg) !important;
                backdrop-filter: blur(12px);
                border-right: 1px solid var(--border-color);
            }
            
            /* Content containers */
            .main-container {
                background-color: var(--surface);
                border-radius: 15px;
                padding: 2rem;
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
                margin-bottom: 2rem;
                border: 1px solid var(--border-color);
                backdrop-filter: blur(12px);
            }
            
            /* Particles system */
            .particles {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: -1;
                pointer-events: none;
            }
            
            .particle {
                position: absolute;
                background: rgba(255, 255, 255, 0.15);
                border-radius: 50%;
                animation: float linear infinite;
                filter: blur(1px);
            }
        </style>
        
        <!-- Floating particles element -->
        <div class="particles" id="particles-js"></div>
        
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const container = document.querySelector('.particles');
                const particleCount = 60;
                
                container.innerHTML = '';
                
                for (let i = 0; i < particleCount; i++) {
                    const particle = document.createElement('div');
                    particle.className = 'particle';
                    
                    // Random properties
                    const size = Math.random() * 6 + 2;
                    const posX = Math.random() * 100;
                    const posY = Math.random() * 100;
                    const duration = Math.random() * 20 + 10;
                    const delay = Math.random() * 5;
                    const opacity = Math.random() * 0.4 + 0.2;
                    
                    // Apply styles
                    Object.assign(particle.style, {
                        width: `${size}px`,
                        height: `${size}px`,
                        left: `${posX}%`,
                        top: `${posY}%`,
                        animation: `float ${duration}s ease infinite ${delay}s`,
                        opacity: opacity,
                        backgroundColor: `rgba(${Math.floor(Math.random() * 100)}, 
                                        ${Math.floor(Math.random() * 200)}, 
                                        255, 
                                        ${opacity})`
                    });
                    
                    container.appendChild(particle);
                }
            });
        </script>
        """,
        unsafe_allow_html=True
    )
# Initialize dark mode
set_dark_mode()

def get_base64_of_image(image_path):
    """Convert image to base64 for embedding"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def get_all_image_paths(folder_path):
    """Recursively get all image file paths from a folder and its subfolders"""
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if Path(file).suffix.lower() in SUPPORTED_FORMATS:
                image_paths.append(Path(root) / file)
    return image_paths

def upload_dataset():
    """Enhanced dataset upload component with better visuals"""
    with st.container():
        st.subheader("📁 Dataset Selection", divider='rainbow')
        
        # Upload methods with cards
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("**📦 Upload ZIP Archive**", expanded=True):
                st.caption("Recommended for most users")
                zip_file = st.file_uploader("Drag and drop or click to browse", 
                                          type=['zip'], 
                                          key='zip_upload',
                                          help="Upload your image dataset as a ZIP file")
        
        with col2:
            with st.expander("**💻 Local Path**", expanded=True):
                st.caption("For advanced users with local access")
                manual_path = st.text_input("Enter folder path", 
                                           key='manual_path',
                                           placeholder="C:/path/to/your/images")
        
        # Initialize variables
        all_image_paths = []
        root_folder = None
        
        # Handle ZIP upload
        if zip_file is not None:
            with st.spinner("🔍 Extracting and analyzing ZIP file..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save the zip file
                    zip_path = Path(temp_dir) / "uploaded_folder.zip"
                    with open(zip_path, "wb") as f:
                        f.write(zip_file.getbuffer())
                    
                    # Extract the zip file
                    shutil.unpack_archive(zip_path, temp_dir)
                    
                    # Find the root folder
                    extracted_items = list(Path(temp_dir).glob("*"))
                    if len(extracted_items) == 1 and extracted_items[0].is_dir():
                        root_folder = extracted_items[0]
                    else:
                        root_folder = Path(temp_dir)
                    
                    # Get all images
                    all_image_paths = get_all_image_paths(root_folder)
        
        # Handle manual path input
        elif manual_path and Path(manual_path).exists():
            root_folder = Path(manual_path)
            all_image_paths = get_all_image_paths(root_folder)
        
        return all_image_paths, root_folder

def show_dataset_stats(all_image_paths, root_folder):
    """Enhanced dataset statistics visualization with class distribution"""
    with st.expander("📊 **Dataset Statistics**", expanded=True):
        # Get all unique directories (potential class folders)
        unique_dirs = {img_path.parent for img_path in all_image_paths}
        format_counts = {}
        for img_path in all_image_paths:
            ext = img_path.suffix.lower()
            format_counts[ext] = format_counts.get(ext, 0) + 1
        
        # Create metrics with icons
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📸 Total Images", len(all_image_paths))
        with col2:
            st.metric("📂 Subfolders", len(unique_dirs))
        with col3:
            st.metric("🗂️ File Types", len(format_counts))
        
        # File format distribution visualization
        st.subheader("File Format Distribution", divider='gray')
        if format_counts:
            chart_data = {"Format": list(format_counts.keys()), "Count": list(format_counts.values())}
            st.bar_chart(chart_data, x="Format", y="Count", use_container_width=True)
        else:
            st.warning("No supported image files found")

        # New: Class Distribution Analysis
        st.subheader("📈 Class Distribution Analysis", divider='gray')
        
        # Check if dataset is organized in class folders (common structure)
        # We consider it a class-based dataset if:
        # 1. There are at least 2 folders directly under root
        # 2. Each of these folders contains at least 1 image
        class_folders = []
        try:
            # Get immediate subdirectories of root folder
            class_folders = [f for f in root_folder.iterdir() if f.is_dir()]
            
            # Check if these folders contain images
            valid_class_folders = []
            for folder in class_folders:
                if any(f.suffix.lower() in SUPPORTED_FORMATS for f in folder.iterdir()):
                    valid_class_folders.append(folder)
            
            if len(valid_class_folders) >= 2:
                class_folders = valid_class_folders
        except Exception as e:
            st.warning(f"Could not analyze folder structure: {str(e)}")
        
        if class_folders:
            st.success("🎯 Detected class-based dataset structure")
            
            # Calculate class distribution
            class_dist = {}
            for class_folder in class_folders:
                class_name = class_folder.name
                # Count images in this class folder
                class_images = [img for img in all_image_paths 
                              if str(class_folder) in str(img.parent)]
                class_dist[class_name] = len(class_images)
            
            # Show class distribution metrics
            st.markdown("### Class Distribution")
            
            # Sort classes by count (descending)
            sorted_classes = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)
            
            # Create columns for metrics (up to 6 classes)
            max_classes_to_show = 6
            num_cols = min(3, len(sorted_classes))
            cols = st.columns(num_cols)
            
            for i, (class_name, count) in enumerate(sorted_classes[:max_classes_to_show]):
                with cols[i % num_cols]:
                    st.metric(
                        f"🏷️ {class_name}",
                        count,
                        help=f"{count} images ({count/len(all_image_paths)*100:.1f}% of total)"
                    )
            
            if len(sorted_classes) > max_classes_to_show:
                st.info(f"ℹ️ Showing top {max_classes_to_show} of {len(sorted_classes)} classes")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Class Distribution Chart")
                class_chart_data = {
                    "Class": list(class_dist.keys()),
                    "Count": list(class_dist.values())
                }
                st.bar_chart(class_chart_data, x="Class", y="Count", use_container_width=True)
            
            with col2:
                st.markdown("#### Class Balance Analysis")
                
                # Calculate balance metrics
                max_count = max(class_dist.values())
                min_count = min(class_dist.values())
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                
                st.metric("Largest Class", max_count)
                st.metric("Smallest Class", min_count)
                st.metric("Imbalance Ratio", f"{imbalance_ratio:.1f}:1")
                
                if imbalance_ratio > 3:
                    st.warning("⚠️ Significant class imbalance detected")
                    st.info("Consider using techniques like oversampling, undersampling, or class weights")
                else:
                    st.success("✅ Classes are relatively balanced")
            
            # Show sample images from each class
            st.markdown("#### Sample Images from Each Class")
            sample_cols = st.columns(min(4, len(class_folders)))
            
            for i, class_folder in enumerate(class_folders[:4]):  # Show up to 4 classes
                with sample_cols[i]:
                    class_name = class_folder.name
                    class_images = [img for img in all_image_paths 
                                  if str(class_folder) in str(img.parent)]
                    
                    if class_images:
                        sample_img = random.choice(class_images)
                        img = cv2.imread(str(sample_img))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        st.image(
                            img,
                            caption=f"Class: {class_name}",
                            use_container_width=True
                        )
                        st.caption(f"{len(class_images)} images")
        else:
            st.info("ℹ️ Dataset doesn't appear to be organized in class folders. For class distribution analysis, organize your dataset with separate folders for each class.")

def show_preview_images(image_paths, root_folder, title="🖼️ Preview Images", cols=5, height=200):
    """Enhanced image preview with hover effects and captions"""
    st.subheader(title, divider='rainbow')
    
    # Create a grid of images
    grid_cols = st.columns(min(cols, len(image_paths)))
    for i, img_path in enumerate(image_paths):
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rel_path = img_path.relative_to(root_folder)
                
                # Convert to PIL for better display
                pil_img = Image.fromarray(img_display)
                
                # Create a container with hover effect
                with grid_cols[i % cols]:
                    st.markdown(
                        f'''
                        <div class="image-container" style="height: {height}px;">
                            <img src="data:image/png;base64,{image_to_base64(pil_img)}" 
                                 style="width: 100%; height: 100%; object-fit: cover;">
                            <div class="image-caption">{rel_path}</div>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
        except Exception as e:
            st.error(f"Error loading image {img_path}: {str(e)}")

def image_to_base64(pil_img):
    """Convert PIL image to base64 string"""
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def show_comparison_view(original_img, processed_img, original_title="Original", processed_title="Processed"):
    """Show image comparison with slider"""
    st.markdown(
        f'''
        <div class="img-comp-container">
            <img class="img-comp-img img-comp-before" 
                 src="data:image/png;base64,{image_to_base64(Image.fromarray(original_img))}"
                 alt="{original_title}">
            <img class="img-comp-img img-comp-after" 
                 src="data:image/png;base64,{image_to_base64(Image.fromarray(processed_img))}"
                 alt="{processed_title}">
            <div class="img-comp-slider">
                <div class="img-comp-slider-line"></div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    st.caption("🔍 Drag the slider to compare images")

def apply_blur(img, method, kernel_size, sigma=1.0):
    """Apply blur to image with configurable intensity"""
    if method == "None":
        return img
        
    kernel_size = max(3, kernel_size)  # Ensure odd number >= 3
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    
    if method == "Gaussian":
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=sigma)
    elif method == "Median":
        return cv2.medianBlur(img, kernel_size)
    elif method == "Bilateral":
        return cv2.bilateralFilter(img, kernel_size, sigma, sigma)
    return img

def apply_sharpening(img, method, strength=1.0):
    """Apply sharpening to image with configurable strength"""
    if method == "None":
        return img
        
    if method == "Unsharp Mask":
        blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
        sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
        return sharpened
    elif method == "Laplacian":
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sharpened = img - strength * laplacian
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    elif method == "Custom Kernel":
        kernel = np.array([[-1, -1, -1],
                          [-1,  9*strength, -1],
                          [-1, -1, -1]]) / (9*strength - 8)
        return cv2.filter2D(img, -1, kernel)
    return img

def apply_edge_detection(img, method, threshold1=100, threshold2=200, kernel_size=3, 
                        scale=1, delta=0, blur_radius=0):
    """
    Apply edge detection to an image with configurable parameters.
    
    Args:
        img: Input image (BGR or grayscale)
        method: Edge detection method ("None", "Canny", "Sobel", "Laplacian", "Scharr")
        threshold1: First threshold for Canny
        threshold2: Second threshold for Canny
        kernel_size: Kernel size for Sobel/Laplacian (must be odd)
        scale: Scale factor for Sobel/Laplacian
        delta: Delta value added to results
        blur_radius: Gaussian blur radius (must be positive and odd)
    
    Returns:
        Edge-detected image (same channels as input)
    """
    if method == "None":
        return img
        
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Optional pre-blurring with strict validation
    if blur_radius > 0:
        # Ensure blur radius is odd and at least 1
        blur_radius = max(1, blur_radius)
        blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        try:
            gray = cv2.GaussianBlur(gray, (blur_radius, blur_radius), 0)
        except Exception as e:
            print(f"Error applying Gaussian blur with radius {blur_radius}: {e}")
            # Fall back to no blur if there's an error
            pass
    
    # Validate kernel size (must be odd)
    kernel_size = max(1, kernel_size)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # Apply selected edge detection method
    if method == "Canny":
        edges = cv2.Canny(gray, threshold1, threshold2)
    elif method == "Sobel":
        sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=kernel_size, scale=scale, delta=delta)
        sobely = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=kernel_size, scale=scale, delta=delta)
        abs_gradx = cv2.convertScaleAbs(sobelx)
        abs_grady = cv2.convertScaleAbs(sobely)
        edges = cv2.addWeighted(abs_gradx, 0.5, abs_grady, 0.5, 0)
    elif method == "Laplacian":
        edges = cv2.Laplacian(gray, cv2.CV_16S, ksize=kernel_size)
        edges = cv2.convertScaleAbs(edges)
    elif method == "Scharr":
        scharrx = cv2.Scharr(gray, cv2.CV_16S, 1, 0)
        scharry = cv2.Scharr(gray, cv2.CV_16S, 0, 1)
        abs_gradx = cv2.convertScaleAbs(scharrx)
        abs_grady = cv2.convertScaleAbs(scharry)
        edges = cv2.addWeighted(abs_gradx, 0.5, abs_grady, 0.5, 0)
    else:
        raise ValueError(f"Unknown edge detection method: {method}")
    
    # Convert back to 3 channels if original was color
    if len(img.shape) == 3:
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return edges

def apply_morphological_operation(img, operation, kernel_shape, kernel_size, iterations=1):
    """Apply morphological operations to image"""
    if operation == "None":
        return img
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create kernel
    kernel = None
    if kernel_shape == "Rectangle":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    elif kernel_shape == "Ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    elif kernel_shape == "Cross":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    
    # Apply operation
    if operation == "Erosion":
        return cv2.erode(img, kernel, iterations=iterations)
    elif operation == "Dilation":
        return cv2.dilate(img, kernel, iterations=iterations)
    elif operation == "Opening":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == "Closing":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation == "Gradient":
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    elif operation == "Top Hat":
        return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    elif operation == "Black Hat":
        return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    return img

def apply_fur_enhancement(img, strength=1.0):
    """Enhance fur texture using CLAHE and edge preservation"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0*strength, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        img = cv2.merge((l, a, b))
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0*strength, tileGridSize=(8,8))
        img = clahe.apply(img)
    return img

def apply_pet_eye_enhancement(img, strength=1.0):
    """Enhance eyes in pet images"""
    if len(img.shape) == 3:
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Enhance V channel
        v = cv2.addWeighted(v, 1.0 + (0.2 * strength), 
                          cv2.GaussianBlur(v, (0,0), 5), 
                          -0.2 * strength, 0)
        
        # Merge back and convert
        hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def apply_background_simplification(img, method="edge_preserving"):
    """Simplify background to focus on animal"""
    if method == "edge_preserving":
        return cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
    elif method == "stylization":
        return cv2.stylization(img, sigma_s=60, sigma_r=0.07)
    return img

def apply_pet_specific_augmentation(img, augmentation_type, params):
    """Apply pet-specific augmentations"""
    if augmentation_type == "fur_noise":
        # Add noise that resembles fur patterns
        noise = np.random.normal(0, params, img.shape[:2])
        noise = cv2.GaussianBlur(noise, (5,5), 0)
        noise = np.expand_dims(noise, axis=2)
        if len(img.shape) == 3:
            noise = np.repeat(noise, 3, axis=2)
        noisy = img + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    elif augmentation_type == "ear_occlusion":
        # Simulate ear occlusion (common in pet photos)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create random triangle for ear
        pt1 = (random.randint(w//4, w//2), random.randint(0, h//4))
        pt2 = (random.randint(0, w//4), random.randint(h//4, h//2))
        pt3 = (random.randint(w//4, w//2), random.randint(h//4, h//2))
        
        triangle = np.array([pt1, pt2, pt3], dtype=np.int32)
        cv2.fillConvexPoly(mask, triangle, 255)
        
        # Apply occlusion
        occluded = img.copy()
        occluded[mask > 0] = random.randint(0, 255)
        return occluded
    
    elif augmentation_type == "whisker_enhancement":
        # Enhance whiskers using line detection
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Detect lines (whiskers)
        lsd = cv2.createLineSegmentDetector(0)
        lines = lsd.detect(gray)[0]
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), 
                        (255, 255, 255), 1)
        return img
    
    return img

def normalize_image(img, scheme_name, custom_mean=None, custom_std=None):
    """Normalize image according to scheme"""
    if scheme_name == "None":
        return img
    
    if scheme_name == "Custom":
        mean = custom_mean if custom_mean is not None else 0.0
        std = custom_std if custom_std is not None else 1.0
    else:
        scheme = NORMALIZATION_SCHEMES[scheme_name]
        mean = scheme['mean']
        std = scheme['std']
    
    if len(img.shape) == 2:  # Grayscale
        img = img.astype(np.float32)
        img = (img - mean) / std
    else:  # Color
        img = img.astype(np.float32)
        if isinstance(mean, list):  # Per-channel normalization
            img = (img - np.array(mean)) / np.array(std)
        else:  # Same for all channels
            img = (img - mean) / std
    return img


def process_single_image(img_path, processing_options, output_path=None, is_preview=False):
    """Process a single image with all specified options including pet-specific processing"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return False, None
        
        # Initialize processed_img with the original image
        processed_img = img.copy()
        
        # Handle color mode reverting
        if processing_options['color_mode'] != "Keep original":
            if processing_options['color_mode'] == "Grayscale":
                if len(processed_img.shape) == 3:
                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            elif processing_options['color_mode'] == "RGB":
                if len(processed_img.shape) == 2:
                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
                else:
                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            elif processing_options['color_mode'] == "BGR":
                if len(processed_img.shape) == 2:
                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

        # Apply all processing steps
        if processing_options['thresholding']['apply'] and processing_options['thresholding']['method'] != "None":
            processed_img = apply_thresholding(
                processed_img,
                processing_options['thresholding']['method'],
                processing_options['thresholding']['threshold'],
                processing_options['thresholding']['max_val'],
                processing_options['thresholding']['block_size'],
                processing_options['thresholding']['C']
            )
        
        if processing_options['transformations']['apply'] and processing_options['transformations']['method'] != "None":
            method = processing_options['transformations']['method']
            if method == "Negative":
                processed_img = apply_negative(processed_img)
            elif method == "Gamma":
                processed_img = apply_gamma_correction(processed_img, processing_options['transformations']['gamma'])
            elif method == "Log":
                processed_img = apply_log_transform(processed_img, processing_options['transformations']['c'])
            elif method == "Sliced":
                processed_img = apply_sliced(processed_img, processing_options['transformations']['slices'])
            elif method == "Histogram Equalization":
                processed_img = apply_histogram_equalization(processed_img)
        
        if processing_options['blur']['apply'] and processing_options['blur']['method'] != "None":
            processed_img = apply_blur(
                processed_img,
                processing_options['blur']['method'],
                processing_options['blur']['kernel_size'],
                processing_options['blur']['sigma']
            )
        
        if processing_options['sharpening']['apply'] and processing_options['sharpening']['method'] != "None":
            processed_img = apply_sharpening(
                processed_img,
                processing_options['sharpening']['method'],
                processing_options['sharpening']['strength']
            )
        
        if processing_options['edge_detection']['apply'] and processing_options['edge_detection']['method'] != "None":
            processed_img = apply_edge_detection(
                processed_img,
                processing_options['edge_detection']['method'],
                processing_options['edge_detection']['threshold1'],
                processing_options['edge_detection']['threshold2'],
                processing_options['edge_detection']['kernel_size'],
                processing_options['edge_detection']['scale'],
                processing_options['edge_detection']['delta'],
                processing_options['edge_detection']['blur_radius']
            )
        
        if processing_options['morphological']['apply'] and processing_options['morphological']['operation'] != "None":
            processed_img = apply_morphological_operation(
                processed_img,
                processing_options['morphological']['operation'],
                processing_options['morphological']['kernel_shape'],
                processing_options['morphological']['kernel_size'],
                processing_options['morphological']['iterations']
            )
        
        if processing_options['resize']['apply']:
            interpolation_map = {
                "INTER_NEAREST": cv2.INTER_NEAREST,
                "INTER_LINEAR": cv2.INTER_LINEAR,
                "INTER_CUBIC": cv2.INTER_CUBIC,
                "INTER_AREA": cv2.INTER_AREA
            }
            processed_img = cv2.resize(
                processed_img,
                (processing_options['resize']['width'], processing_options['resize']['height']),
                interpolation=interpolation_map[processing_options['resize']['method']]
            )
        
        if processing_options['normalization']['apply'] and processing_options['normalization']['scheme'] != "None":
            processed_img = normalize_image(
                processed_img,
                processing_options['normalization']['scheme'],
                processing_options['normalization']['custom_mean'],
                processing_options['normalization']['custom_std']
            )
        
        # For preview mode, just return the processed image
        if is_preview:
            return True, processed_img
        
        # For actual processing, save the image
        # Determine save path
        if output_path is None:
            save_path = img_path
        else:
            rel_path = img_path.relative_to(processing_options['root_folder'])
            save_path = output_path / rel_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save image
        ext = save_path.suffix.lower()
        if processing_options['normalization']['apply'] and processing_options['normalization']['scheme'] != "None" and not processing_options['normalization']['save_as_float']:
            # Convert back to 8-bit for standard image formats
            if len(processed_img.shape) == 2:  # Grayscale
                processed_img = cv2.normalize(processed_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            else:  # Color
                processed_img = cv2.normalize(processed_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)
        
        if ext in ['.jpg', '.jpeg']:
            cv2.imwrite(str(save_path), processed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        elif ext == '.png':
            cv2.imwrite(str(save_path), processed_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        elif ext == '.webp':
            cv2.imwrite(str(save_path), processed_img, [int(cv2.IMWRITE_WEBP_QUALITY), 95])
        elif ext == '.npy' or (processing_options['normalization']['apply'] and processing_options['normalization']['save_as_float']):
            np.save(str(save_path.with_suffix('.npy')), processed_img)
            if ext != '.npy':
                try:
                    save_path.unlink()
                except:
                    pass
        else:
            cv2.imwrite(str(save_path), processed_img)
        
        return True, processed_img
            
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return False, None

def home_page():
    """Updated home page with current features and testing section"""
    st.title("✨ VisionPrep")
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
        
        .title-text {
            font-family: 'Montserrat', sans-serif;
            font-weight: 700;
            color: white;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }
        
        .subtitle-text {
            font-family: 'Montserrat', sans-serif;
            font-weight: 400;
            color: var(--on-background);
            margin-bottom: 2rem;
            font-size: 1.2rem;
        }
        
        .hero-section {
            background: linear-gradient(135deg, rgba(106, 17, 203, 0.2), rgba(37, 117, 252, 0.2));
            border-radius: 16px;
            padding: 3rem;
            margin-bottom: 3rem;
            text-align: center;
            border: 1px solid var(--border-color);
            backdrop-filter: blur(10px);
            animation: pulse 6s ease infinite;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .hero-logo {
            width: 180px;
            height: 180px;
            border-radius: 50%;
            object-fit: cover;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 30px rgba(106, 17, 203, 0.5);
            border: 3px solid rgba(255, 255, 255, 0.15);
            transition: all 0.3s ease;
        }
        
        .hero-logo:hover {
            transform: scale(1.05);
            box-shadow: 0 12px 35px rgba(106, 17, 203, 0.7);
        }
        
        .test-image-container {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            justify-content: center;
        }
        
        .test-image-card {
            width: 200px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            background: var(--card-bg);
        }
        
        .test-image-card img {
            width: 100%;
            height: 150px;
            object-fit: cover;
        }
        
        .test-image-card p {
            padding: 0.5rem;
            margin: 0;
            text-align: center;
            font-size: 0.9rem;
        }
    </style>
    
    <div class="hero-section">
        <img class="hero-logo" src="https://sdmntprukwest.oaiusercontent.com/files/00000000-1b5c-6243-9ee4-e1ff8d01b2f5/raw?se=2025-05-04T19%3A39%3A26Z&sp=r&sv=2024-08-04&sr=b&scid=6fe8f62f-9da3-5d9f-86f2-3fc7106dc7fc&skoid=54ae6e2b-352e-4235-bc96-afa2512cc978&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-05-04T09%3A08%3A53Z&ske=2025-05-05T09%3A08%3A53Z&sks=b&skv=2024-08-04&sig=4pQG4K9uexwittZ36nEXVvl8J4lkzAtE3/nePTMyOFk%3D" 
             alt="VisionPrep Logo">
        <h1 class="title-text">Advanced Image Processing Pipeline</h1>
        <p class="subtitle-text">Transform, enhance, and process your images with professional-grade tools</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Rest of your home_page() function remains the same...
    # Features section with updated cards
    st.subheader("🚀 Key Features", divider='rainbow')
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: var(--accent);">🔧 Core Processing</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>🎨 <strong>Color space</strong> conversion (RGB, Grayscale)</li>
                <li>🌀 <strong>Blur</strong> (Gaussian, Median, Bilateral)</li>
                <li>🔪 <strong>Sharpening</strong> (Unsharp Mask, Laplacian)</li>
                <li>🖇️ <strong>Morphological</strong> operations</li>
                <li>📏 <strong>Resizing</strong> with interpolation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: var(--accent);">🔳 Thresholding</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>⚡ <strong>Binary</strong> and adaptive methods</li>
                <li>🎚️ <strong>Otsu's</strong> automatic thresholding</li>
                <li>🌓 <strong>Adaptive Gaussian</strong> thresholding</li>
                <li>🔄 <strong>Inverse</strong> threshold operations</li>
                <li>📶 <strong>ToZero/Trunc</strong> variants</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col2:
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: var(--accent);">🌈 Intensity Transforms</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>🔄 <strong>Negative</strong> image transformation</li>
                <li>☀️ <strong>Gamma</strong> correction</li>
                <li>📊 <strong>Log</strong> transformation</li>
                <li>🔢 <strong>Intensity</strong> slicing</li>
                <li>⚖️ <strong>Histogram</strong> equalization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: var(--accent);">🔄 Batch Processing</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>📂 <strong>Dataset</strong> statistics and analysis</li>
                <li>⚡ <strong>Parallel</strong> processing</li>
                <li>👁️ <strong>Preview</strong> before processing</li>
                <li>💾 <strong>Multiple output</strong> formats</li>
                <li>📈 <strong>Normalization</strong> schemes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # How to use section with animated steps
    st.subheader("📝 How to Use", divider='rainbow')
    
    steps = [
        {"icon": "📤", "title": "Upload Dataset", "desc": "Upload your images as a ZIP file or provide a local path", "color": "#6a11cb"},
        {"icon": "⚙️", "title": "Select Options", "desc": "Choose from various processing and augmentation options", "color": "#2575fc"},
        {"icon": "👁️", "title": "Preview Results", "desc": "See how your images will look before processing", "color": "#00c6fb"},
        {"icon": "🚀", "title": "Batch Processing", "desc": "Apply transformations to your entire dataset efficiently", "color": "#6a11cb"}
    ]
    
    for i, step in enumerate(steps):
        with st.container():
            st.markdown(
                f"""
                <div style="
                    background: rgba({int(step['color'][1:3], 16)}, {int(step['color'][3:5], 16)}, {int(step['color'][5:7], 16)}, 0.1);
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin-bottom: 1rem;
                    border-left: 4px solid {step['color']};
                    transition: all 0.3s ease;
                    animation: float 4s ease infinite;
                    animation-delay: {i * 0.2}s;
                ">
                    <div style="display: flex; align-items: center;">
                        <div style="
                            background: {step['color']};
                            width: 50px;
                            height: 50px;
                            border-radius: 50%;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            margin-right: 1.5rem;
                            font-size: 1.5rem;
                            color: white;
                            flex-shrink: 0;
                        ">
                            {step['icon']}
                        </div>
                        <div>
                            <h3 style="margin: 0; color: white;">{step['title']}</h3>
                            <p style="margin: 0.5rem 0 0; color: var(--on-background);">{step['desc']}</p>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # About section with animated cards
    st.subheader("ℹ️ About This Tool", divider='rainbow')
    
    about_col1, about_col2 = st.columns(2)
    
    with about_col1:
        st.markdown("""
        <div class="feature-card" style="animation: float 6s ease infinite;">
            <h3 style="color: var(--accent);">Purpose</h3>
            <p>This advanced image processing pipeline was designed to help researchers, developers, and data scientists prepare and enhance image datasets for computer vision and deep learning applications.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with about_col2:
        st.markdown("""
        <div class="feature-card" style="animation: float 6s ease infinite; animation-delay: 0.5s;">
            <h3 style="color: var(--accent);">Technology</h3>
            <p>Built with Streamlit, OpenCV, and Python, it combines powerful image processing capabilities with an intuitive interface and professional visualization tools.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Floating action button
    st.markdown(
        """
        <div class="floating-btn" onclick="window.scrollTo({top: 0, behavior: 'smooth'});">
            ↑
        </div>
        """,
        unsafe_allow_html=True
    )
    # Image Testing Section
    st.subheader("🧪 Test Image Processing", divider='rainbow')
    st.markdown("Upload an image to see all processing operations in action:")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and convert image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply all processing operations
        with st.spinner("Processing image..."):
            # Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
            # Brightness/Darkness
            brighter = cv2.convertScaleAbs(img_rgb, alpha=1.5, beta=20)
            darker = cv2.convertScaleAbs(img_rgb, alpha=0.7, beta=-20)
            
            # Thresholding
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
            
            # Adaptive Gaussian Threshold
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
            adaptive_rgb = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB)
            
            # Negative
            negative = cv2.bitwise_not(img_rgb)
            
            # Gamma
            gamma = apply_gamma_correction(img_rgb, 2.2)
            
            # Log
            log = apply_log_transform(img_rgb)
            
            # Sliced
            sliced = apply_sliced(img_rgb)
            
            # Blur
            blur = cv2.GaussianBlur(img_rgb, (15, 15), 0)
            
            # Sharpen
            sharpen = apply_sharpening(img_rgb, "Unsharp Mask", 1.5)
            
            # Histogram Equalization
            equalized = apply_histogram_equalization(img_rgb)
            
            # Prepare all images for display
            operations = [
                ("Original", img_rgb),
                ("Grayscale", gray_rgb),
                ("Brighter", brighter),
                ("Darker", darker),
                ("Threshold", thresh_rgb),
                ("Adaptive Thresh", adaptive_rgb),
                ("Negative", negative),
                ("Gamma", gamma),
                ("Log", log),
                ("Sliced", sliced),
                ("Blur", blur),
                ("Sharpen", sharpen),
                ("Hist Equal", equalized)
            ]
            
            # Display in a grid
            st.markdown("### Processing Results")

            # Calculate how many rows we'll need (4 columns per row)
            num_rows = (len(operations) + 3) // 4  # Round up division

            for row in range(num_rows):
                # Create a new row of columns for each set of 4 images
                cols = st.columns(4)
    
                # Calculate start and end indices for this row
                start_idx = row * 4
                end_idx = min((row + 1) * 4, len(operations))
    
                for col_idx, (name, processed_img) in enumerate(operations[start_idx:end_idx]):
                    with cols[col_idx]:
                        st.image(
                            processed_img,
                            caption=name,
                            use_container_width=True  # Using the new parameter
                        )

def apply_thresholding(img, method, threshold=127, max_val=255, block_size=11, C=2):
    """Apply thresholding operations to image"""
    if method == "None":
        return img
    
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if method == "Binary":
        _, result = cv2.threshold(img, threshold, max_val, cv2.THRESH_BINARY)
    elif method == "Binary Inv":
        _, result = cv2.threshold(img, threshold, max_val, cv2.THRESH_BINARY_INV)
    elif method == "Trunc":
        _, result = cv2.threshold(img, threshold, max_val, cv2.THRESH_TRUNC)
    elif method == "To Zero":
        _, result = cv2.threshold(img, threshold, max_val, cv2.THRESH_TOZERO)
    elif method == "To Zero Inv":
        _, result = cv2.threshold(img, threshold, max_val, cv2.THRESH_TOZERO_INV)
    elif method == "Otsu":
        _, result = cv2.threshold(img, 0, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "Adaptive Gaussian":
        result = cv2.adaptiveThreshold(
            img, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, C
        )
    elif method == "Adaptive Mean":
        result = cv2.adaptiveThreshold(
            img, max_val, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, block_size, C
        )
    return result

def apply_negative(img):
    """Apply negative transformation"""
    return cv2.bitwise_not(img)

def apply_gamma_correction(img, gamma=1.0):
    """Apply gamma correction"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_log_transform(img, c=1):
    """Apply log transformation"""
    img = img.astype(np.float32)
    c = 255 / np.log(1 + np.max(img))
    log_transformed = c * (np.log(img + 1))
    return np.uint8(log_transformed)

import numpy as np
import cv2

def apply_sliced(img, slices=4):
    """Apply intensity slicing with optimized operations
    
    Args:
        img: Input image (grayscale or color)
        slices: Number of intensity slices to apply
        
    Returns:
        Image with intensity slicing applied
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate slice ranges
    max_val = np.max(img)
    slice_size = max_val / slices  # Use float division for more precise boundaries
    
    # Create result array
    result = np.zeros_like(img)
    
    # Vectorized implementation
    for i in range(slices):
        lower = i * slice_size
        upper = (i + 1) * slice_size if i < slices - 1 else max_val + 1  # +1 to include max_val
        
        # Create mask for current slice
        mask = (img >= lower) & (img < upper)
        
        # Apply mask (using numpy indexing is faster than cv2.bitwise_and)
        result[mask] = img[mask]
    
    return result
    
def apply_histogram_equalization(img):
    """Apply histogram equalization"""
    if len(img.shape) == 3:
        # Convert to YCrCb color space
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        
        # Split into channels (this returns a tuple)
        y, cr, cb = cv2.split(img_ycrcb)
        
        # Equalize the Y channel only
        y_eq = cv2.equalizeHist(y)
        
        # Merge back the channels
        img_ycrcb_eq = cv2.merge((y_eq, cr, cb))
        
        # Convert back to BGR
        return cv2.cvtColor(img_ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    else:
        # Grayscale image
        return cv2.equalizeHist(img)
    
def basic_processing_page():
    """Enhanced Basic Processing Page with Pet-Specific Features"""
    st.title("🔧 Basic Image Processing")
    st.markdown("""
    <style>
        .processing-header {
            color: #8a7fff;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .param-section {
            background-color: #2d2d2d;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid #444;
        }
    </style>
    <p class="processing-header">Professional-grade image processing operations</p>
    """, unsafe_allow_html=True)
    
    # Dataset upload and preview
    all_image_paths, root_folder = upload_dataset()
    
    if len(all_image_paths) > 0:
        st.success(f"✅ Found {len(all_image_paths)} images in dataset")
        show_dataset_stats(all_image_paths, root_folder)
        
        if 'preview_paths' not in st.session_state or st.button("🔄 Reselect Preview Images"):
            st.session_state.preview_paths = random.sample(all_image_paths, min(5, len(all_image_paths)))
        
        show_preview_images(st.session_state.preview_paths, root_folder)
        
        # Processing options with tabs - REMOVED "Pet-Specific" tab
        st.subheader("⚙️ Processing Options", divider='rainbow')
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Basic Operations", 
            "Advanced Operations", 
            "Thresholding & Equalization",
            "Output Settings"
        ])
        
        with tab1:
            st.markdown("### 🎨 Color Conversion")
            color_mode = st.selectbox(
                "Color Space",
                ["Keep original", "Grayscale", "RGB", "BGR"],
                index=0,
                key="color_mode"
            )
            
            st.markdown("### 🌀 Blur Options")
            blur_apply = st.checkbox("Apply Blur", value=False, key="blur_apply")
            if blur_apply:
                blur_method = st.selectbox(
                    "Blur Method",
                    ["None", "Gaussian", "Median", "Bilateral"],
                    index=1,
                    key="blur_method"
                )
                blur_kernel = st.slider("Kernel Size", 3, 15, 5, step=2, key="blur_kernel")
                if blur_method == "Gaussian":
                    blur_sigma = st.slider("Sigma (Intensity)", 0.1, 5.0, 1.0, step=0.1, key="blur_sigma")
                else:
                    blur_sigma = 0
                
                # Blur preview button
                if st.button("👁️ Preview Blur", key="preview_blur"):
                    st.subheader("🌀 Blur Preview", divider='rainbow')
                    preview_image_path = random.choice(st.session_state.preview_paths)
                    img = cv2.imread(str(preview_image_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    blurred = apply_blur(img, blur_method, blur_kernel, blur_sigma)
                    
                    # Display comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(img, caption="Original", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(blurred, caption=f"{blur_method} Blur", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### 🔍 Edge Detection")
            edge_apply = st.checkbox("Apply Edge Detection", value=False, key="edge_apply")
            if edge_apply:
                edge_method = st.selectbox(
                    "Edge Detection Method",
                    ["None", "Canny", "Sobel", "Laplacian", "Scharr"],
                    index=1,
                    key="edge_method"
                )
                
                col1, col2 = st.columns(2)
                kernel_size = col1.slider("Kernel Size", 1, 7, 3, step=2, key="edge_kernel")
                blur_radius = col2.slider("Pre-blur Radius", 0, 7, 0, step=2, key="edge_blur")
                
                if edge_method == "Canny":
                    st.markdown("#### Canny Parameters")
                    col1, col2 = st.columns(2)
                    threshold1 = col1.slider("Threshold 1", 0, 255, 100, key="edge_thresh1")
                    threshold2 = col2.slider("Threshold 2", 0, 255, 200, key="edge_thresh2")
                elif edge_method in ["Sobel", "Laplacian", "Scharr"]:
                    st.markdown("#### Advanced Parameters")
                    col1, col2 = st.columns(2)
                    scale = col1.slider("Scale", 0.1, 5.0, 1.0, 0.1, key="edge_scale")
                    delta = col2.slider("Delta", -100, 100, 0, key="edge_delta")
                
                # Edge detection preview
                if st.button("👁️ Preview Edge Detection", key="preview_edge"):
                    st.subheader("🔍 Edge Detection Preview", divider='rainbow')
                    preview_image_path = random.choice(st.session_state.preview_paths)
                    img = cv2.imread(str(preview_image_path))
                    
                    edge_params = {
                        'method': edge_method,
                        'threshold1': threshold1 if edge_method == "Canny" else 100,
                        'threshold2': threshold2 if edge_method == "Canny" else 200,
                        'kernel_size': kernel_size,
                        'scale': scale if edge_method in ["Sobel", "Laplacian", "Scharr"] else 1,
                        'delta': delta if edge_method in ["Sobel", "Laplacian", "Scharr"] else 0,
                        'blur_radius': max(1, blur_radius) if edge_apply and blur_radius > 0 else 0 
                    }
                    
                    edges = apply_edge_detection(img, **edge_params)
                    
                    # Display comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        if len(edges.shape) == 2:
                            st.image(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), caption=f"{edge_method} Edges", use_container_width=True)
                        else:
                            st.image(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB), caption=f"{edge_method} Edges", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("### 🔪 Sharpening Options")
            sharpen_apply = st.checkbox("Apply Sharpening", value=False, key="sharpen_apply")
            if sharpen_apply:
                sharpen_method = st.selectbox(
                    "Sharpening Method",
                    ["None", "Unsharp Mask", "Laplacian", "Custom Kernel"],
                    index=1,
                    key="sharpen_method"
                )
                sharpen_strength = st.slider("Sharpening Strength", 0.1, 3.0, 1.0, step=0.1, key="sharpen_strength")
                
                # Sharpening preview button
                if st.button("👁️ Preview Sharpening", key="preview_sharpening"):
                    st.subheader("🔪 Sharpening Preview", divider='rainbow')
                    preview_image_path = random.choice(st.session_state.preview_paths)
                    img = cv2.imread(str(preview_image_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    sharpened = apply_sharpening(img, sharpen_method, sharpen_strength)
                    
                    # Display comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(img, caption="Original", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(sharpened, caption=f"{sharpen_method} Sharpening", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("### 🖇️ Morphological Operations")
            morph_apply = st.checkbox("Apply Morphological Operations", value=False, key="morph_apply")
            if morph_apply:
                morph_operation = st.selectbox(
                    "Operation",
                    ["None", "Erosion", "Dilation", "Opening", "Closing", "Gradient", "Top Hat", "Black Hat"],
                    index=1,
                    key="morph_operation"
                )
                morph_kernel_shape = st.selectbox(
                    "Kernel Shape",
                    ["Rectangle", "Ellipse", "Cross"],
                    index=0,
                    key="morph_kernel_shape"
                )
                morph_kernel_size = st.slider("Kernel Size", 3, 15, 3, step=2, key="morph_kernel_size")
                morph_iterations = st.slider("Iterations", 1, 10, 1, key="morph_iterations")
                
                # Morphological preview button
                if st.button("👁️ Preview Morphological Operation", key="preview_morph"):
                    st.subheader("🖇️ Morphological Operation Preview", divider='rainbow')
                    preview_image_path = random.choice(st.session_state.preview_paths)
                    img = cv2.imread(str(preview_image_path))
                    
                    morphed = apply_morphological_operation(
                        img,
                        morph_operation,
                        morph_kernel_shape,
                        morph_kernel_size,
                        morph_iterations
                    )
                    
                    # Display comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        if len(morphed.shape) == 2:
                            st.image(cv2.cvtColor(morphed, cv2.COLOR_GRAY2RGB), caption=f"{morph_operation}", use_container_width=True)
                        else:
                            st.image(cv2.cvtColor(morphed, cv2.COLOR_BGR2RGB), caption=f"{morph_operation}", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("### 📏 Resizing Options")
            resize_apply = st.checkbox("Resize Images", value=True, key="resize_apply")
            if resize_apply:
                col1, col2 = st.columns(2)
                width = col1.number_input("Width", min_value=1, value=224, key="resize_width")
                height = col2.number_input("Height", min_value=1, value=224, key="resize_height")
                resize_method = st.selectbox(
                    "Interpolation Method",
                    ["INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_AREA"],
                    index=1,
                    key="resize_method"
                )
                
                # Resizing preview button
                if st.button("👁️ Preview Resizing", key="preview_resize"):
                    st.subheader("📏 Resizing Preview", divider='rainbow')
                    preview_image_path = random.choice(st.session_state.preview_paths)
                    img = cv2.imread(str(preview_image_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    interpolation_map = {
                        "INTER_NEAREST": cv2.INTER_NEAREST,
                        "INTER_LINEAR": cv2.INTER_LINEAR,
                        "INTER_CUBIC": cv2.INTER_CUBIC,
                        "INTER_AREA": cv2.INTER_AREA
                    }
                    
                    resized = cv2.resize(
                        img,
                        (width, height),
                        interpolation=interpolation_map[resize_method]
                    )
                    
                    # Display comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(img, caption=f"Original ({img.shape[1]}x{img.shape[0]})", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(resized, caption=f"Resized ({width}x{height})", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

        with tab3:  # Thresholding & Equalization tab
            st.markdown("### 🔳 Thresholding Operations")
            threshold_apply = st.checkbox("Apply Thresholding", value=False, key="threshold_apply")
            if threshold_apply:
                threshold_method = st.selectbox(
                    "Threshold Method",
                    [
                        "None", "Binary", "Binary Inv", "Trunc", 
                        "To Zero", "To Zero Inv", "Otsu",
                        "Adaptive Gaussian", "Adaptive Mean"
                    ],
                    index=0,
                    key="threshold_method"
                )
                
                if threshold_method not in ["None", "Otsu", "Adaptive Gaussian", "Adaptive Mean"]:
                    threshold_value = st.slider("Threshold Value", 0, 255, 127, key="threshold_value")
                    max_value = st.slider("Max Value", 0, 255, 255, key="max_value")
                
                if threshold_method in ["Adaptive Gaussian", "Adaptive Mean"]:
                    col1, col2 = st.columns(2)
                    block_size = col1.slider("Block Size", 3, 31, 11, step=2, key="block_size")
                    C = col2.slider("C Value", -10, 10, 2, key="C_value")
            
            # Preview for thresholding
            if threshold_apply and st.button("👁️ Preview Thresholding", key="preview_threshold"):
                st.subheader("🔳 Thresholding Preview", divider='rainbow')
                preview_image_path = random.choice(st.session_state.preview_paths)
                img = cv2.imread(str(preview_image_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                threshold_params = {
                    'method': threshold_method,
                    'threshold': threshold_value if 'threshold_value' in st.session_state else 127,
                    'max_val': max_value if 'max_value' in st.session_state else 255,
                    'block_size': block_size if 'block_size' in st.session_state else 11,
                    'C': C if 'C_value' in st.session_state else 2
                }
                
                thresholded = apply_thresholding(img, **threshold_params)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Original", use_container_width=True)
                with col2:
                    if len(thresholded.shape) == 2:
                        st.image(cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB), 
                               caption=f"{threshold_method} Threshold", 
                               use_container_width=True)
                    else:
                        st.image(thresholded, 
                               caption=f"{threshold_method} Threshold", 
                               use_container_width=True)
            
            st.markdown("### 🌈 Intensity Transformations")
            transform_apply = st.checkbox("Apply Intensity Transformation", value=False, key="transform_apply")
            if transform_apply:
                transform_method = st.selectbox(
                    "Transformation Method",
                    ["None", "Negative", "Gamma", "Log", "Sliced", "Histogram Equalization"],
                    index=0,
                    key="transform_method"
                )
                
                if transform_method == "Gamma":
                    gamma_value = st.slider("Gamma Value", 0.1, 5.0, 1.0, 0.1, key="gamma_value")
                elif transform_method == "Log":
                    c_value = st.slider("C Value", 0.1, 10.0, 1.0, 0.1, key="c_value")
                elif transform_method == "Sliced":
                    slices = st.slider("Number of Slices", 2, 10, 4, key="slices")
            
            # Preview for transformations
            if transform_apply and st.button("👁️ Preview Transformation", key="preview_transform"):
                st.subheader("🌈 Intensity Transformation Preview", divider='rainbow')
                preview_image_path = random.choice(st.session_state.preview_paths)
                img = cv2.imread(str(preview_image_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if transform_method == "Negative":
                    transformed = apply_negative(img)
                elif transform_method == "Gamma":
                    transformed = apply_gamma_correction(img, gamma_value)
                elif transform_method == "Log":
                    transformed = apply_log_transform(img, c_value)
                elif transform_method == "Sliced":
                    transformed = apply_sliced(img, slices)
                elif transform_method == "Histogram Equalization":
                    transformed = apply_histogram_equalization(img)
                else:
                    transformed = img.copy()  # Default case if no transformation is selected
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Original", use_container_width=True)
                with col2:
                    st.image(transformed, caption=f"{transform_method} Transformation", use_container_width=True)

        with tab4:  # Output Settings tab
            st.markdown("### 📊 Normalization")
            norm_apply = st.checkbox("Normalize Images", value=False, key="norm_apply")
            if norm_apply:
                norm_scheme = st.selectbox(
                    "Normalization Scheme",
                    list(NORMALIZATION_SCHEMES.keys()),
                    index=0,
                    key="norm_scheme"
                )
                if norm_scheme == 'Custom':
                    col1, col2 = st.columns(2)
                    custom_mean = col1.number_input("Mean Value", value=0.0, key="custom_mean")
                    custom_std = col2.number_input("Std Value", min_value=0.01, value=1.0, key="custom_std")
                else:
                    custom_mean = None
                    custom_std = None
                save_as_float = st.checkbox("Save as float32 (.npy)", value=False, key="save_as_float")
            
            st.markdown("### 💾 Output Settings")
            save_option = st.radio(
                "Output Method:",
                [
                    "Create new folder with processed images (preserve structure)",
                    "Overwrite original images (with backup option)"
                ],
                index=0,
                key="save_option"
            )
            
            if save_option.startswith("Create new folder"):
                output_folder = st.text_input("Output folder name:", "processed_dataset", key="output_folder")
            
            batch_size = st.slider("Batch size (images per process)", 10, 500, 100, key="batch_size")
            num_workers = st.slider("Number of parallel workers", 1, cpu_count(), min(4, cpu_count()), key="num_workers")

        # processing_options dictionary
        processing_options = {
            'root_folder': root_folder,
            'color_mode': color_mode,
            'blur': {
                'apply': blur_apply,
                'method': blur_method if blur_apply else None,
                'kernel_size': blur_kernel if blur_apply else 0,
                'sigma': blur_sigma if blur_apply and blur_method == "Gaussian" else 0
            },
            'sharpening': {
                'apply': sharpen_apply,
                'method': sharpen_method if sharpen_apply else None,
                'strength': sharpen_strength if sharpen_apply else 0
            },
            'edge_detection': {
                'apply': edge_apply,
                'method': edge_method if edge_apply else None,
                'threshold1': threshold1 if edge_apply and edge_method == "Canny" else 0,
                'threshold2': threshold2 if edge_apply and edge_method == "Canny" else 0,
                'kernel_size': kernel_size if edge_apply else 3,
                'scale': scale if edge_apply and edge_method in ["Sobel", "Laplacian", "Scharr"] else 1,
                'delta': delta if edge_apply and edge_method in ["Sobel", "Laplacian", "Scharr"] else 0,
                'blur_radius': blur_radius if edge_apply else 0
            },
            'morphological': {
                'apply': morph_apply,
                'operation': morph_operation if morph_apply else None,
                'kernel_shape': morph_kernel_shape if morph_apply else None,
                'kernel_size': morph_kernel_size if morph_apply else 0,
                'iterations': morph_iterations if morph_apply else 0
            },
            'resize': {
                'apply': resize_apply,
                'width': width if resize_apply else 0,
                'height': height if resize_apply else 0,
                'method': resize_method if resize_apply else None
            },
            'normalization': {
                'apply': norm_apply,
                'scheme': norm_scheme if norm_apply else None,
                'custom_mean': custom_mean if norm_apply and norm_scheme == 'Custom' else None,
                'custom_std': custom_std if norm_apply and norm_scheme == 'Custom' else None,
                'save_as_float': save_as_float if norm_apply else False
            },
            'thresholding': {
                'apply': threshold_apply,
                'method': threshold_method if threshold_apply else None,
                'threshold': threshold_value if threshold_apply and threshold_method not in ["None", "Otsu", "Adaptive Gaussian", "Adaptive Mean"] else 127,
                'max_val': max_value if threshold_apply and threshold_method not in ["None", "Otsu", "Adaptive Gaussian", "Adaptive Mean"] else 255,
                'block_size': block_size if threshold_apply and threshold_method in ["Adaptive Gaussian", "Adaptive Mean"] else 11,
                'C': C if threshold_apply and threshold_method in ["Adaptive Gaussian", "Adaptive Mean"] else 2
            },
            'transformations': {
                'apply': transform_apply,
                'method': transform_method if transform_apply else None,
                'gamma': gamma_value if transform_apply and transform_method == "Gamma" else 1.0,
                'c': c_value if transform_apply and transform_method == "Log" else 1.0,
                'slices': slices if transform_apply and transform_method == "Sliced" else 4
            }
        }
        
        # Preview Processing
        if st.button("👁️ Preview Processing", key="preview_button"):
            st.subheader("🔍 Processing Preview", divider='rainbow')
            
            preview_cols = st.columns(min(5, len(st.session_state.preview_paths)))
            for i, img_path in enumerate(st.session_state.preview_paths):
                success, processed_img = process_single_image(img_path, processing_options, is_preview=True)
                if success and processed_img is not None:
                    if len(processed_img.shape) == 2:  # Grayscale
                        img_display = processed_img
                    else:
                        img_display = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                    
                    rel_path = img_path.relative_to(root_folder)
                    with preview_cols[i]:
                        st.markdown(f'<div class="image-container">', unsafe_allow_html=True)
                        st.image(
                            img_display, 
                            caption=f"Processed: {rel_path}", 
                            use_container_width=True,
                            clamp=True
                        )
                        st.markdown(f'</div>', unsafe_allow_html=True)
        
        # Batch Processing
        st.subheader("⚡ Batch Processing", divider='rainbow')
        
        if st.button("🚀 Start Batch Processing", key="process_button", type="primary"):
            with st.spinner("⚙️ Processing images..."):
                start_time = time.time()
                
                # Create output folder if needed
                if save_option.startswith("Create new folder"):
                    output_path = Path(output_folder)
                    output_path.mkdir(exist_ok=True)
                else:
                    output_path = None
                
                # Process in batches using multiprocessing
                successful = 0
                failed = 0
                
                # Prepare batches
                batches = [all_image_paths[i:i + batch_size] 
                          for i in range(0, len(all_image_paths), batch_size)]
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process function for multiprocessing
                process_func = partial(process_single_image, 
                                    processing_options=processing_options,
                                    output_path=output_path)
                
                with Pool(num_workers) as pool:
                    results = []
                    total_batches = len(batches)
                    
                    for i, batch in enumerate(batches):
                        batch_results = pool.map(process_func, batch)
                        successful += sum(1 for r in batch_results if r[0])
                        failed += sum(1 for r in batch_results if not r[0])
                        
                        # Update progress
                        progress = (i + 1) / total_batches
                        progress_bar.progress(progress)
                        status_text.text(
                            f"📊 Progress: {i * batch_size + len(batch)}/{len(all_image_paths)} images "
                            f"| ✅ Successful: {successful} | ❌ Failed: {failed}"
                        )
                
                # Final report
                elapsed_time = time.time() - start_time
                st.balloons()
                st.success(
                    f"""
                    ## ✅ Processing Complete!
                    
                    - **Total images processed**: {len(all_image_paths)}
                    - **Successful operations**: {successful}
                    - **Failed operations**: {failed}
                    - **Time taken**: {elapsed_time:.2f} seconds
                    - **Processing speed**: {len(all_image_paths)/elapsed_time:.2f} images/second
                    """
                )

def apply_augmentation(img, aug_type, params):
    """Apply a single augmentation to an image"""
    if aug_type == "horizontal_flip":
        return cv2.flip(img, 1)
    elif aug_type == "vertical_flip":
        return cv2.flip(img, 0)
    elif aug_type == "rotate":
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), params, 1)
        return cv2.warpAffine(img, M, (w, h))
    elif aug_type == "rotate_range":
        angle = random.uniform(params[0], params[1])
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(img, M, (w, h))
    elif aug_type == "zoom":
        h, w = img.shape[:2]
        scale = params
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h))
        
        # Crop or pad to maintain original size
        if scale > 1:  # Zoom in - crop center
            start_h, start_w = (new_h - h) // 2, (new_w - w) // 2
            return resized[start_h:start_h+h, start_w:start_w+w]
        else:  # Zoom out - pad with black
            pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
            return cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
    elif aug_type == "zoom_range":
        scale = random.uniform(params[0], params[1])
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h))
        
        if scale > 1:
            start_h, start_w = (new_h - h) // 2, (new_w - w) // 2
            return resized[start_h:start_h+h, start_w:start_w+w]
        else:
            pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
            return cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
    elif aug_type == "brightness":
        img_pil = Image.fromarray(img)
        enhancer = ImageEnhance.Brightness(img_pil)
        return np.array(enhancer.enhance(params))
    elif aug_type == "brightness_range":
        factor = random.uniform(params[0], params[1])
        img_pil = Image.fromarray(img)
        enhancer = ImageEnhance.Brightness(img_pil)
        return np.array(enhancer.enhance(factor))
    elif aug_type == "contrast":
        img_pil = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(img_pil)
        return np.array(enhancer.enhance(params))
    elif aug_type == "contrast_range":
        factor = random.uniform(params[0], params[1])
        img_pil = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(img_pil)
        return np.array(enhancer.enhance(factor))
    elif aug_type == "gaussian_noise":
        noise = np.random.normal(0, params, img.shape)
        noisy = img + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    elif aug_type == "salt_pepper":
        noisy = img.copy()
        amount = params
        # Salt mode
        num_salt = np.ceil(amount * img.size * 0.5)
        coords = [np.random.randint(0, i-1, int(num_salt)) for i in img.shape]
        noisy[coords[0], coords[1]] = 255
        # Pepper mode
        num_pepper = np.ceil(amount * img.size * 0.5)
        coords = [np.random.randint(0, i-1, int(num_pepper)) for i in img.shape]
        noisy[coords[0], coords[1]] = 0
        return noisy
    elif aug_type in ["fur_noise", "ear_occlusion", "whisker_enhancement"]:
        return apply_pet_specific_augmentation(img, aug_type, params)
    return img

def data_augmentation_page():
    """Enhanced data augmentation page with blur and sharpening options"""
    st.title("🔄 Data Augmentation")
    st.markdown("""
    <style>
        .augmentation-header {
            color: #2575fc;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
    </style>
    <p class="augmentation-header">Generate diverse training data by creating augmented versions of your images</p>
    """, unsafe_allow_html=True)
    
    all_image_paths, root_folder = upload_dataset()
    
    if len(all_image_paths) > 0:
        st.success(f"✅ Found {len(all_image_paths)} images in dataset")
        show_dataset_stats(all_image_paths, root_folder)
        
        # Select preview images
        if 'preview_paths' not in st.session_state or st.button("🔄 Reselect Preview Images"):
            st.session_state.preview_paths = random.sample(all_image_paths, min(5, len(all_image_paths)))
        
        show_preview_images(st.session_state.preview_paths, root_folder, "Original Images")
        
        # Augmentation options with tabs
        st.subheader("⚙️ Augmentation Options", divider='rainbow')
        
        tab1, tab2 = st.tabs(["Basic Augmentations", "Advanced Settings"])
        
        with tab1:
            st.markdown("### 🔄 Spatial Transformations")
            col1, col2 = st.columns(2)
            with col1:
                flip_h = st.checkbox("Horizontal Flip", value=False, key="flip_h")
                flip_v = st.checkbox("Vertical Flip", value=False, key="flip_v")
                rotate = st.checkbox("Rotation", value=False, key="rotate")
            with col2:
                zoom = st.checkbox("Zoom", value=False, key="zoom")
                brightness = st.checkbox("Brightness Adjustment", value=False, key="brightness")
                contrast = st.checkbox("Contrast Adjustment", value=False, key="contrast")
            
            st.markdown("### 🌀 Blur & Sharpening")
            col1, col2 = st.columns(2)
            with col1:
                blur = st.checkbox("Gaussian Blur", value=False, key="blur")
                if blur:
                    blur_kernel = st.slider("Blur Kernel Size", 3, 15, 5, step=2, key="blur_kernel")
                    blur_sigma = st.slider("Blur Sigma", 0.1, 5.0, 1.0, step=0.1, key="blur_sigma")
            with col2:
                sharpen = st.checkbox("Sharpening", value=False, key="sharpen")
                if sharpen:
                    sharpen_method = st.selectbox(
                        "Sharpening Method",
                        ["Unsharp Mask", "Laplacian", "Custom Kernel"],
                        index=0,
                        key="sharpen_method"
                    )
                    sharpen_strength = st.slider("Sharpening Strength", 0.1, 3.0, 1.0, step=0.1, key="sharpen_strength")
        
        with tab2:
            st.markdown("### 🎲 Noise & Variations")
            noise = st.checkbox("Add Noise", value=False, key="noise")
            
            # Rotation options
            if rotate:
                st.markdown("#### Rotation Settings")
                rotation_mode = st.radio(
                    "Rotation Mode", 
                    ["Fixed Angle", "Random Range"], 
                    index=0,
                    key="rotation_mode"
                )
                if rotation_mode == "Fixed Angle":
                    rotation_angle = st.slider(
                        "Rotation Angle (degrees)", 
                        -180, 180, 30,
                        key="rotation_angle"
                    )
                else:
                    col1, col2 = st.columns(2)
                    rotation_min = col1.slider("Min Angle", -180, 180, -30, key="rotation_min")
                    rotation_max = col2.slider("Max Angle", -180, 180, 30, key="rotation_max")
            
            # Zoom options
            if zoom:
                st.markdown("#### Zoom Settings")
                zoom_mode = st.radio(
                    "Zoom Mode", 
                    ["Fixed", "Random Range"], 
                    index=0,
                    key="zoom_mode"
                )
                if zoom_mode == "Fixed":
                    zoom_factor = st.slider(
                        "Zoom Factor", 
                        0.5, 2.0, 1.2, 0.1,
                        key="zoom_factor"
                    )
                else:
                    col1, col2 = st.columns(2)
                    zoom_min = col1.slider("Min Zoom", 0.5, 1.5, 0.9, 0.1, key="zoom_min")
                    zoom_max = col2.slider("Max Zoom", 0.5, 2.0, 1.3, 0.1, key="zoom_max")
            
            # Brightness options
            if brightness:
                st.markdown("#### Brightness Settings")
                brightness_mode = st.radio(
                    "Brightness Mode", 
                    ["Fixed", "Random Range"], 
                    index=0,
                    key="brightness_mode"
                )
                if brightness_mode == "Fixed":
                    brightness_factor = st.slider(
                        "Brightness Factor", 
                        0.1, 3.0, 1.5, 0.1,
                        key="brightness_factor"
                    )
                else:
                    col1, col2 = st.columns(2)
                    brightness_min = col1.slider("Min Brightness", 0.1, 1.5, 0.8, 0.1, key="brightness_min")
                    brightness_max = col2.slider("Max Brightness", 0.5, 3.0, 1.7, 0.1, key="brightness_max")
            
            # Contrast options
            if contrast:
                st.markdown("#### Contrast Settings")
                contrast_mode = st.radio(
                    "Contrast Mode", 
                    ["Fixed", "Random Range"], 
                    index=0,
                    key="contrast_mode"
                )
                if contrast_mode == "Fixed":
                    contrast_factor = st.slider(
                        "Contrast Factor", 
                        0.1, 3.0, 1.5, 0.1,
                        key="contrast_factor"
                    )
                else:
                    col1, col2 = st.columns(2)
                    contrast_min = col1.slider("Min Contrast", 0.1, 1.5, 0.8, 0.1, key="contrast_min")
                    contrast_max = col2.slider("Max Contrast", 0.5, 3.0, 1.7, 0.1, key="contrast_max")
            
            # Noise options
            if noise:
                st.markdown("#### Noise Settings")
                noise_mode = st.radio(
                    "Noise Type", 
                    ["Gaussian", "Salt & Pepper"], 
                    index=0,
                    key="noise_mode"
                )
                if noise_mode == "Gaussian":
                    noise_scale = st.slider(
                        "Noise Scale", 
                        1, 50, 10,
                        key="noise_scale"
                    )
                else:
                    noise_amount = st.slider(
                        "Noise Amount", 
                        0.001, 0.1, 0.01, 0.001,
                        key="noise_amount"
                    )
        
        # Generate all possible augmentation options
        augmentations = []
        if flip_h:
            augmentations.append(("horizontal_flip", None))
        if flip_v:
            augmentations.append(("vertical_flip", None))
        if rotate:
            if rotation_mode == "Fixed Angle":
                augmentations.append(("rotate", rotation_angle))
            else:
                augmentations.append(("rotate_range", (rotation_min, rotation_max)))
        if zoom:
            if zoom_mode == "Fixed":
                augmentations.append(("zoom", zoom_factor))
            else:
                augmentations.append(("zoom_range", (zoom_min, zoom_max)))
        if brightness:
            if brightness_mode == "Fixed":
                augmentations.append(("brightness", brightness_factor))
            else:
                augmentations.append(("brightness_range", (brightness_min, brightness_max)))
        if contrast:
            if contrast_mode == "Fixed":
                augmentations.append(("contrast", contrast_factor))
            else:
                augmentations.append(("contrast_range", (contrast_min, contrast_max)))
        if blur:
            augmentations.append(("gaussian_blur", (blur_kernel, blur_sigma)))
        if sharpen:
            augmentations.append(("sharpening", (sharpen_method, sharpen_strength)))
        if noise:
            if noise_mode == "Gaussian":
                augmentations.append(("gaussian_noise", noise_scale))
            else:
                augmentations.append(("salt_pepper", noise_amount))
        
        # Preview augmentations
        if st.button("👁️ Preview Augmentations", key="preview_augmentations") and st.session_state.preview_paths:
            st.subheader("🔍 Augmentation Preview", divider='rainbow')
            
            # Select one image for preview
            preview_image_path = random.choice(st.session_state.preview_paths)
            img = cv2.imread(str(preview_image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply each augmentation to show examples
            for aug_type, params in augmentations:
                st.markdown(f"#### {aug_type.replace('_', ' ').title()}")
                cols = st.columns(3)
                
                for i in range(3):  # Show 3 examples per augmentation
                    if "range" in aug_type:
                        # For range augmentations, use random value within range
                        if aug_type == "rotate_range":
                            angle = random.uniform(params[0], params[1])
                            augmented = apply_augmentation(img.copy(), "rotate", angle)
                        elif aug_type == "zoom_range":
                            scale = random.uniform(params[0], params[1])
                            augmented = apply_augmentation(img.copy(), "zoom", scale)
                        elif aug_type == "brightness_range":
                            factor = random.uniform(params[0], params[1])
                            augmented = apply_augmentation(img.copy(), "brightness", factor)
                        elif aug_type == "contrast_range":
                            factor = random.uniform(params[0], params[1])
                            augmented = apply_augmentation(img.copy(), "contrast", factor)
                    elif aug_type == "gaussian_blur":
                        augmented = cv2.GaussianBlur(img.copy(), (params[0], params[0]), params[1])
                    elif aug_type == "sharpening":
                        augmented = apply_sharpening(img.copy(), params[0], params[1])
                    else:
                        augmented = apply_augmentation(img.copy(), aug_type, params)
                    
                    with cols[i]:
                        st.markdown(f'<div class="image-container">', unsafe_allow_html=True)
                        st.image(augmented, use_container_width=True)
                        st.markdown(f'</div>', unsafe_allow_html=True)
        
        # Batch processing options
        st.subheader("⚡ Batch Augmentation", divider='rainbow')
        
        output_folder = st.text_input("Output folder name:", "augmented_dataset", key="aug_output_folder")
        copies_per_image = st.number_input(
            "Copies per image", 
            min_value=1, 
            max_value=100, 
            value=1,
            key="copies_per_image"
        )
        
        if st.button("🚀 Generate Augmented Dataset", key="augment_button", type="primary"):
            with st.spinner("✨ Generating augmented images..."):
                start_time = time.time()
                
                # Create output folder
                output_path = Path(output_folder)
                output_path.mkdir(exist_ok=True)
                
                # Process all images
                total_images = 0
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, img_path in enumerate(all_image_paths):
                    # Load original image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Save original (always)
                    rel_path = img_path.relative_to(root_folder)
                    original_save_path = output_path / rel_path
                    original_save_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(img).save(original_save_path)
                    total_images += 1
                    
                    # Determine how many augmentations we need to apply
                    num_augmentations = len(augmentations)
                    if num_augmentations == 0:
                        continue  # No augmentations selected
                    
                    # Generate augmented copies
                    for copy_num in range(copies_per_image):
                        augmented = img.copy()
                        
                        if copy_num < num_augmentations:
                            # Apply single augmentation for first N copies
                            aug_type, params = augmentations[copy_num]
                            
                            # Handle range parameters
                            if "range" in aug_type:
                                if aug_type == "rotate_range":
                                    angle = random.uniform(params[0], params[1])
                                    augmented = apply_augmentation(augmented, "rotate", angle)
                                elif aug_type == "zoom_range":
                                    scale = random.uniform(params[0], params[1])
                                    augmented = apply_augmentation(augmented, "zoom", scale)
                                elif aug_type == "brightness_range":
                                    factor = random.uniform(params[0], params[1])
                                    augmented = apply_augmentation(augmented, "brightness", factor)
                                elif aug_type == "contrast_range":
                                    factor = random.uniform(params[0], params[1])
                                    augmented = apply_augmentation(augmented, "contrast", factor)
                            elif aug_type == "gaussian_blur":
                                augmented = cv2.GaussianBlur(augmented, (params[0], params[0]), params[1])
                            elif aug_type == "sharpening":
                                augmented = apply_sharpening(augmented, params[0], params[1])
                            else:
                                augmented = apply_augmentation(augmented, aug_type, params)
                        else:
                            # If we need more copies than augmentations, combine them randomly
                            num_to_apply = random.randint(1, num_augmentations)
                            selected_augmentations = random.sample(augmentations, num_to_apply)
                            
                            for aug_type, params in selected_augmentations:
                                if "range" in aug_type:
                                    if aug_type == "rotate_range":
                                        angle = random.uniform(params[0], params[1])
                                        augmented = apply_augmentation(augmented, "rotate", angle)
                                    elif aug_type == "zoom_range":
                                        scale = random.uniform(params[0], params[1])
                                        augmented = apply_augmentation(augmented, "zoom", scale)
                                    elif aug_type == "brightness_range":
                                        factor = random.uniform(params[0], params[1])
                                        augmented = apply_augmentation(augmented, "brightness", factor)
                                    elif aug_type == "contrast_range":
                                        factor = random.uniform(params[0], params[1])
                                        augmented = apply_augmentation(augmented, "contrast", factor)
                                elif aug_type == "gaussian_blur":
                                    augmented = cv2.GaussianBlur(augmented, (params[0], params[0]), params[1])
                                elif aug_type == "sharpening":
                                    augmented = apply_sharpening(augmented, params[0], params[1])
                                else:
                                    augmented = apply_augmentation(augmented, aug_type, params)
                        
                        # Save augmented image
                        if copy_num < num_augmentations:
                            # For single augmentations, name them with the augmentation type
                            aug_type = augmentations[copy_num][0]
                            aug_save_path = output_path / rel_path.parent / f"{img_path.stem}_{aug_type}{img_path.suffix}"
                        else:
                            # For combined augmentations, just use a sequential number
                            aug_save_path = output_path / rel_path.parent / f"{img_path.stem}_aug{copy_num}{img_path.suffix}"
                        
                        aug_save_path.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(augmented).save(aug_save_path)
                        total_images += 1
                    
                    # Update progress
                    progress = (i + 1) / len(all_image_paths)
                    progress_bar.progress(progress)
                    status_text.text(f"📊 Progress: {i+1}/{len(all_image_paths)} images | Total generated: {total_images}")
                
                # Final report with animation
                elapsed_time = time.time() - start_time
                st.balloons()
                st.success(
                    f"""
                    ## ✅ Augmentation Complete!
                    
                    - **Original images**: {len(all_image_paths)}
                    - **Augmented images**: {total_images - len(all_image_paths)}
                    - **Total generated**: {total_images}
                    - **Time taken**: {elapsed_time:.2f} seconds
                    """
                )

def main():
    """Main application with dark mode sidebar"""
    # Custom sidebar styling
    st.markdown("""
    <style>
        .sidebar-title {
            color: white !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            margin-bottom: 1.5rem !important;
            text-align: center !important;
        }
        
        .sidebar-logo {
            display: block;
            margin: 0 auto 1.5rem auto;
            width: 80%;
            max-width: 200px;
            border-radius: 50%;
            box-shadow: 0 4px 20px rgba(106, 17, 203, 0.5);
            border: 2px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .sidebar-logo:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 25px rgba(106, 17, 203, 0.7);
        }
        
        .sidebar-section {
            margin-bottom: 1.5rem !important;
        }
        
        .sidebar-link {
            display: flex !important;
            align-items: center !important;
            padding: 0.5rem 1rem !important;
            border-radius: 8px !important;
            margin-bottom: 0.5rem !important;
            transition: all 0.3s ease !important;
            color: var(--on-surface) !important;
        }
        
        .sidebar-link:hover {
            background: rgba(255, 255, 255, 0.1) !important;
            transform: translateX(5px) !important;
        }
        
        .sidebar-link.active {
            background: linear-gradient(45deg, var(--primary), var(--secondary)) !important;
            color: white !important;
        }
        
        .sidebar-icon {
            margin-right: 0.75rem !important;
            font-size: 1.2rem !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar content
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center;">
                <img class="sidebar-logo" src="https://sdmntprukwest.oaiusercontent.com/files/00000000-1b5c-6243-9ee4-e1ff8d01b2f5/raw?se=2025-05-04T19%3A39%3A26Z&sp=r&sv=2024-08-04&sr=b&scid=6fe8f62f-9da3-5d9f-86f2-3fc7106dc7fc&skoid=54ae6e2b-352e-4235-bc96-afa2512cc978&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-05-04T09%3A08%3A53Z&ske=2025-05-05T09%3A08%3A53Z&sks=b&skv=2024-08-04&sig=4pQG4K9uexwittZ36nEXVvl8J4lkzAtE3/nePTMyOFk%3D" 
                     alt="VisionPrep Logo">
                <h1 class="sidebar-title">VisionPrep</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        # Navigation
        page_options = {
            "Home": "🏠",
            "Basic Processing": "🔧",
            "Data Augmentation": "🔄"
        }
        
        page = st.radio(
            "Navigation",
            list(page_options.keys()),
            index=0,
            label_visibility="collapsed",
            format_func=lambda x: f"{page_options[x]} {x}"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # About section
        st.markdown("""
        <div class="sidebar-section">
            <h3>ℹ️ About</h3>
            <p>Professional image processing and augmentation tool for computer vision workflows.</p>
            <p>---</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Page routing
    if page == "Home":
        home_page()
    elif page == "Basic Processing":
        basic_processing_page()
    else:
        data_augmentation_page()

if __name__ == "__main__":
    main()