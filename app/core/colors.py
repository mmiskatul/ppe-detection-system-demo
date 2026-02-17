from typing import Dict, List

from app.core.model import get_model_class_names

# Standard, high-contrast palette for detection overlays.
STANDARD_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
]


def get_class_colors() -> Dict[str, str]:
    classes: List[str] = get_model_class_names()
    return {
        class_name: STANDARD_COLORS[idx % len(STANDARD_COLORS)]
        for idx, class_name in enumerate(classes)
    }
