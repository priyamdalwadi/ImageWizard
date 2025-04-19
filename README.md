# ImageWizard
 imagewiz is a beginner-friendly Python library for quickly visualizing and comparing multiple image filters with just one line of code.


Perfect for:

Exploring image pre-processing techniques

Teaching image filtering concepts

Rapid anomaly detection prototyping


✨ Features
🧠 One-line visualization of multiple filters

🎛️ Pre-built categories: edges, blur, sharpen, threshold, color

🖼️ Auto-subplot with grid layout

📂 Option to save each filtered image separately (organized by filter)

⚙️ Control grayscale mode and subplot sizing

✅ Built with OpenCV + Matplotlib


🚀 Quick Start
pip install -e .

from imagewiz import visualize_filters

visualize_filters("your_image.jpg", category="edges", gray_scale=True, subplot_size=3)

