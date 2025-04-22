from imagewiz.filters import visualize_filters

visualize_filters.edges(
    image="images/img.jpg",
    gray=True,
    save_each=True,
    save_fig=False,
    show=True,
    figsize=(12, 8),
    # Inline filter params:
    Canny__threshold1=50,
    Canny__threshold2=150,
    Sobel__ksize=7
)

visualize_filters.blur(
    image="images/img.jpg",
    gray=False,
    save_each=True,
    save_fig=False,
    show=True,
    figsize=(12, 8),
    Gaussian__ksize=7,
    Median__ksize=3
)