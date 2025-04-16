import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Image Conversion Helper ---

def to_uint8(image):
    """Normalize and convert image to uint8 for display."""
    if image.dtype == np.float64 or image.dtype == np.float32:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = np.uint8(image)
    elif image.dtype != np.uint8:
        image = np.uint8(image)
    return image

# --- Filter Functions ---

# Edge Detection
def sobel(image):
    return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)

def scharr(image):
    return cv2.Scharr(image, cv2.CV_64F, 1, 0)

def laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F)

def canny(image):
    return cv2.Canny(image, 100, 200)

# Blur Filters
def gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def median_blur(image):
    return cv2.medianBlur(image, 5)

def bilateral_filter(image):
    return cv2.bilateralFilter(image, 9, 75, 75)

def box_filter(image):
    return cv2.boxFilter(image, -1, (5, 5))

def motion_blur(image):
    kernel = np.zeros((15, 15))
    kernel[int((15 - 1)/2), :] = np.ones(15)
    kernel = kernel / 15
    return cv2.filter2D(image, -1, kernel)

# Sharpening Filters
def unsharp_mask(image):
    blurred = cv2.GaussianBlur(image, (9, 9), 10.0)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

def high_boost(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    mask = cv2.subtract(image, blurred)
    return cv2.add(image, mask)

# Thresholding
def otsu(image):
    _, result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result

def adaptive_mean(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

def adaptive_gaussian(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

# Color Manipulation
def equalize_hist(image):
    return cv2.equalizeHist(image)

def clahe(image):
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe_obj.apply(image)

# --- Filter Registry ---

FILTER_CATEGORIES = {
    "edges": {
        "Sobel": {"func": sobel, "requires_gray": True},
        "Scharr": {"func": scharr, "requires_gray": True},
        "Laplacian": {"func": laplacian, "requires_gray": True},
        "Canny": {"func": canny, "requires_gray": True}
    },
    "blur": {
        "GaussianBlur": {"func": gaussian_blur, "requires_gray": False},
        "MedianBlur": {"func": median_blur, "requires_gray": False},
        "BilateralFilter": {"func": bilateral_filter, "requires_gray": False},
        "BoxFilter": {"func": box_filter, "requires_gray": False},
        "MotionBlur": {"func": motion_blur, "requires_gray": False}
    },
    "sharpen": {
        "UnsharpMask": {"func": unsharp_mask, "requires_gray": False},
        "HighBoost": {"func": high_boost, "requires_gray": False}
    },
    "threshold": {
        "Otsu": {"func": otsu, "requires_gray": True},
        "AdaptiveMean": {"func": adaptive_mean, "requires_gray": True},
        "AdaptiveGaussian": {"func": adaptive_gaussian, "requires_gray": True}
    },
    "color": {
        "HistogramEqualization": {"func": equalize_hist, "requires_gray": True},
        "CLAHE": {"func": clahe, "requires_gray": True}
    }
}

# --- Visualization Function ---

def visualize_filters(image_path, category='edges', save=False, show=True, gray_scale=True,
                      subplot_size=3, save_each=False, grid=True):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_color = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    if category not in FILTER_CATEGORIES:
        raise ValueError(f"Unsupported category: {category}. "
                         f"Available categories: {list(FILTER_CATEGORIES.keys())}")

    filters = FILTER_CATEGORIES[category]

    # Clamp subplot size
    subplot_size = max(0.5, min(subplot_size, 5))
    total = len(filters) + 1  # original + filters

    # Calculate layout
    cols = math.ceil(math.sqrt(total)) if grid else total
    rows = math.ceil(total / cols)

    fig_width = cols * subplot_size
    fig_height = rows * subplot_size
    plt.figure(figsize=(fig_width, fig_height))

    # Show original
    display_original = image_gray if gray_scale else image_color
    if len(display_original.shape) == 2:
        display_original = cv2.cvtColor(display_original, cv2.COLOR_GRAY2RGB)
    else:
        display_original = cv2.cvtColor(display_original, cv2.COLOR_BGR2RGB)

    plt.subplot(rows, cols, 1)
    plt.imshow(display_original)
    plt.title('Original')
    plt.axis('off')

    # Output save root
    if save_each:
        base_output_dir = os.path.join("output_filters", category)
        os.makedirs(base_output_dir, exist_ok=True)

    for idx, (name, meta) in enumerate(filters.items(), start=2):
        try:
            func = meta["func"]
            input_img = image_gray if gray_scale else image_color
            result = to_uint8(func(input_img))

            if len(result.shape) == 2:
                rgb_result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            else:
                rgb_result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

            # Show in subplot
            plt.subplot(rows, cols, idx)
            plt.imshow(rgb_result)
            plt.title(name)
            plt.axis('off')

            # Save each filter image separately
            if save_each:
                output_dir = os.path.join(base_output_dir, name)
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, f"{name}.png")
                cv2.imwrite(out_path, result)  # Save raw (not RGB converted)

        except Exception as e:
            print(f"[!] Failed to apply filter {name}: {e}")

    if save:
        plt.savefig("filtered_output.png")

    if show:
        plt.show()


# --- Utility Functions ---

def list_categories():
    return list(FILTER_CATEGORIES.keys())

def list_filters(category=None):
    if category:
        return list(FILTER_CATEGORIES.get(category, {}).keys())
    return {cat: list(filters.keys()) for cat, filters in FILTER_CATEGORIES.items()}
