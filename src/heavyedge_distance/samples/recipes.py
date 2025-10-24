"""Recipies to build sample data."""

from heavyedge import get_sample_path as he_sample

RECIPES = {
    "MeanProfiles-AreaScaled.h5": lambda path: [
        "heavyedge",
        "scale",
        "--type=area",
        he_sample("MeanProfiles.h5"),
        "-o",
        path,
    ],
    "MeanProfiles-PlateauScaled.h5": lambda path: [
        "heavyedge",
        "scale",
        "--type=plateau",
        he_sample("MeanProfiles.h5"),
        "-o",
        path,
    ],
}
