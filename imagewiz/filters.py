import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union, Any, Tuple
import logging
from tqdm import tqdm

# Configure logging
tqdm.pandas()
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Type definitions
ImageInput = Union[str, Path, np.ndarray]
FilterFunc = Callable[..., np.ndarray]

# --- Core Implementation ---
def _visualize_filters_impl(
    image: ImageInput,
    category: str,
    gray: bool = True,
    save_fig: bool = False,
    save_each: bool = False,
    show: bool = True,
    cmap: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    filter_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    from pathlib import Path

    # Registry of filters
    FILTER_REGISTRY: Dict[str, Dict[str, Dict[str, Any]]] = {
        'edges': {
            'Sobel': {'func': FilterVisualizer.sobel, 'gray': True},
            'Scharr': {'func': FilterVisualizer.scharr, 'gray': True},
            'Laplacian': {'func': FilterVisualizer.laplacian, 'gray': True},
            'Canny': {'func': FilterVisualizer.canny, 'gray': True},
        },
        'blur': {
            'Gaussian': {'func': FilterVisualizer.gaussian_blur, 'gray': False},
            'Median': {'func': FilterVisualizer.median_blur, 'gray': False},
            'Bilateral': {'func': FilterVisualizer.bilateral_filter, 'gray': False},
            'Motion': {'func': FilterVisualizer.motion_blur, 'gray': False},
        },
        'sharpen': {
            'Unsharp': {'func': FilterVisualizer.unsharp_mask, 'gray': False},
            'HighBoost': {'func': FilterVisualizer.high_boost, 'gray': False},
        },
        'threshold': {
            'Otsu': {'func': FilterVisualizer.otsu, 'gray': True},
            'Adaptive': {'func': FilterVisualizer.adaptive_threshold, 'gray': True},
        },
        'color': {
            'HistogramEqualization': {'func': FilterVisualizer.equalize_hist, 'gray': True},
            'CLAHE': {'func': FilterVisualizer.clahe, 'gray': True},
        }
    }

    if category not in FILTER_REGISTRY:
        raise ValueError(f"Unsupported category '{category}'. Available: {list(FILTER_REGISTRY.keys())}")

    viz = FilterVisualizer(image, gray=gray)
    filters = FILTER_REGISTRY[category]
    total = len(filters) + 1
    cols = math.ceil(math.sqrt(total))
    rows = math.ceil(total / cols)
    figsize = figsize or (cols * 3, rows * 3)
    plt.figure(figsize=figsize)

    imgs = [('Original', viz.original_gray if gray else viz.original_color)]
    for name, meta in filters.items():
        func = meta['func']
        inp = viz.original_gray if meta['gray'] else viz.original_color
        params = filter_params.get(name, {}) if filter_params else {}
        try:
            res = FilterVisualizer.to_uint8(func(inp, **params))
            imgs.append((name, res))
        except Exception as e:
            logging.warning(f"Failed '{name}' with params {params}: {e}")

    for idx, (title, img) in enumerate(imgs, start=1):
        ax = plt.subplot(rows, cols, idx)
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap or 'gray')
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')

        if save_each and viz.image_path:
            out_dir = viz.output_dir / category / title
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{title}.png"
            cv2.imwrite(str(out_path), img)
            logging.info(f"Saved {out_path}")

    if save_fig and viz.image_path:
        fig_path = (viz.image_path.parent if viz.image_path else Path('.')) / f"{category}_filters.png"
        plt.savefig(str(fig_path), bbox_inches='tight')
        logging.info(f"Saved figure {fig_path}")

    if show:
        plt.tight_layout()
        plt.show()

class FilterVisualizer:
    """
    Image loader and filter implementations.
    
    Also provides category-specific shortcuts via the VisualizeFiltersAPI.
    """
    def __init__(self, image: ImageInput, gray: bool = True, output_dir: Union[str, Path] = "output_filters") -> None:
        self.image_path = None
        self.gray = gray
        self.output_dir = Path(output_dir)
        self._load(image)

    def _load(self, image: ImageInput) -> None:
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            self.image_path = img_path
            color = cv2.imread(str(img_path))
            if color is None:
                raise ValueError(f"Failed to read image: {img_path}")
        else:
            color = image.copy()
        self.original_color = color
        self.original_gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        logging.info("Image loaded successfully.")

    @staticmethod
    def to_uint8(image: np.ndarray) -> np.ndarray:
        if image.dtype in (np.float32, np.float64):
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = image.astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
        return image

    # Filter implementations
    @staticmethod
    def sobel(image: np.ndarray, dx: int = 1, dy: int = 0, ksize: int = 5) -> np.ndarray:
        return cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize=ksize)

    @staticmethod
    def scharr(image: np.ndarray, dx: int = 1, dy: int = 0) -> np.ndarray:
        return cv2.Scharr(image, cv2.CV_64F, dx, dy)

    @staticmethod
    def laplacian(image: np.ndarray, ksize: int = 1) -> np.ndarray:
        return cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)

    @staticmethod
    def canny(image: np.ndarray, threshold1: int = 100, threshold2: int = 200) -> np.ndarray:
        return cv2.Canny(image, threshold1, threshold2)

    @staticmethod
    def gaussian_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    @staticmethod
    def median_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
        return cv2.medianBlur(image, ksize)

    @staticmethod
    def bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    @staticmethod
    def motion_blur(image: np.ndarray, size: int = 15) -> np.ndarray:
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2, :] = 1
        kernel /= size
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def unsharp_mask(image: np.ndarray, ksize: int = 9, amount: float = 1.5) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
        return cv2.addWeighted(image, amount, blurred, -0.5 * (amount - 1), 0)

    @staticmethod
    def high_boost(image: np.ndarray, ksize: int = 5, amount: float = 1.0) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
        mask = cv2.subtract(image, blurred)
        return cv2.addWeighted(image, 1 + amount, mask, amount, 0)

    @staticmethod
    def otsu(image: np.ndarray) -> np.ndarray:
        _, result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result

    @staticmethod
    def adaptive_threshold(
        image: np.ndarray,
        method: str = 'mean',
        block_size: int = 11,
        C: int = 2
    ) -> np.ndarray:
        flags = cv2.ADAPTIVE_THRESH_MEAN_C if method == 'mean' else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        return cv2.adaptiveThreshold(image, 255, flags, cv2.THRESH_BINARY, block_size, C)

    @staticmethod
    def equalize_hist(image: np.ndarray) -> np.ndarray:
        return cv2.equalizeHist(image)

    @staticmethod
    def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe_obj.apply(image)

# --- Public API with static typing for IDE ---
class VisualizeFiltersAPI:
    """
    Main entrypoint. Call directly or via category shortcuts for inline params.
    """
    def __call__(
        self,
        image: ImageInput,
        category: str = 'edges',
        gray: bool = True,
        save_fig: bool = False,
        save_each: bool = False,
        show: bool = True,
        cmap: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        filter_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        _visualize_filters_impl(
            image=image,
            category=category,
            gray=gray,
            save_fig=save_fig,
            save_each=save_each,
            show=show,
            cmap=cmap,
            figsize=figsize,
            filter_params=filter_params,
        )

    def edges(
        self,
        image: ImageInput,
        gray: bool = True,
        save_fig: bool = False,
        save_each: bool = False,
        show: bool = True,
        cmap: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        Sobel__dx: int = 1,
        Sobel__dy: int = 0,
        Sobel__ksize: int = 5,
        Scharr__dx: int = 1,
        Scharr__dy: int = 0,
        Laplacian__ksize: int = 1,
        Canny__threshold1: int = 100,
        Canny__threshold2: int = 200,
    ) -> None:
        """Shortcut for 'edges' filters with inline params."""
        params = {
            'Sobel': {'dx': Sobel__dx, 'dy': Sobel__dy, 'ksize': Sobel__ksize},
            'Scharr': {'dx': Scharr__dx, 'dy': Scharr__dy},
            'Laplacian': {'ksize': Laplacian__ksize},
            'Canny': {'threshold1': Canny__threshold1, 'threshold2': Canny__threshold2},
        }
        _visualize_filters_impl(
            image=image, category='edges', gray=gray,
            save_fig=save_fig, save_each=save_each, show=show,
            cmap=cmap, figsize=figsize, filter_params=params
        )

    def blur(
        self,
        image: ImageInput,
        gray: bool = True,
        save_fig: bool = False,
        save_each: bool = False,
        show: bool = True,
        cmap: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        Gaussian__ksize: int = 5,
        Median__ksize: int = 5,
        Bilateral__d: int = 9,
        Bilateral__sigma_color: float = 75,
        Bilateral__sigma_space: float = 75,
        Motion__size: int = 15,
    ) -> None:
        """Shortcut for 'blur' filters with inline params."""
        params = {
            'Gaussian': {'ksize': Gaussian__ksize},
            'Median': {'ksize': Median__ksize},
            'Bilateral': {'d': Bilateral__d, 'sigma_color': Bilateral__sigma_color, 'sigma_space': Bilateral__sigma_space},
            'Motion': {'size': Motion__size},
        }
        _visualize_filters_impl(
            image=image, category='blur', gray=gray,
            save_fig=save_fig, save_each=save_each, show=show,
            cmap=cmap, figsize=figsize, filter_params=params
        )

    def sharpen(
        self,
        image: ImageInput,
        gray: bool = True,
        save_fig: bool = False,
        save_each: bool = False,
        show: bool = True,
        cmap: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        Unsharp__ksize: int = 9,
        Unsharp__amount: float = 1.5,
        HighBoost__ksize: int = 5,
        HighBoost__amount: float = 1.0,
    ) -> None:
        """Shortcut for 'sharpen' filters with inline params."""
        params = {
            'Unsharp': {'ksize': Unsharp__ksize, 'amount': Unsharp__amount},
            'HighBoost': {'ksize': HighBoost__ksize, 'amount': HighBoost__amount},
        }
        _visualize_filters_impl(
            image=image, category='sharpen', gray=gray,
            save_fig=save_fig, save_each=save_each, show=show,
            cmap=cmap, figsize=figsize, filter_params=params
        )

    def threshold(
        self,
        image: ImageInput,
        gray: bool = True,
        save_fig: bool = False,
        save_each: bool = False,
        show: bool = True,
        cmap: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        Adaptive__method: str = 'mean',
        Adaptive__block_size: int = 11,
        Adaptive__C: int = 2,
    ) -> None:
        """Shortcut for 'threshold' filters with inline params."""
        params = {
            'Adaptive': {'method': Adaptive__method, 'block_size': Adaptive__block_size, 'C': Adaptive__C},
        }
        _visualize_filters_impl(
            image=image, category='threshold', gray=gray,
            save_fig=save_fig, save_each=save_each, show=show,
            cmap=cmap, figsize=figsize, filter_params=params
        )

    def color(
        self,
        image: ImageInput,
        gray: bool = True,
        save_fig: bool = False,
        save_each: bool = False,
        show: bool = True,
        cmap: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        CLAHE__clip_limit: float = 2.0,
        CLAHE__tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> None:
        """Shortcut for 'color' filters with inline params."""
        params = {
            'CLAHE': {'clip_limit': CLAHE__clip_limit, 'tile_grid_size': CLAHE__tile_grid_size},
        }
        _visualize_filters_impl(
            image=image, category='color', gray=gray,
            save_fig=save_fig, save_each=save_each, show=show,
            cmap=cmap, figsize=figsize, filter_params=params
        )

# module-level instance for API
visualize_filters = VisualizeFiltersAPI()
