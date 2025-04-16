from imagewiz import visualize_filters

from imagewiz.filters import list_categories, list_filters

#images/topView.png
from imagewiz import visualize_filters

visualize_filters(
    "images/topView.png",
    category="blur",
    gray_scale=False,
    save_each=True,         # 🔥 Saves each filter output
    grid=True,              # ✅ Displays in a grid
    subplot_size=3
)

