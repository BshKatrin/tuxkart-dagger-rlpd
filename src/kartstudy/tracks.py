"""Fixed learner split. Difficulty is reporting metadata, never a policy input."""

TRAIN_GROUPS = {
    "simple": (
        "lighthouse",
        "xr591",
        "scotland",
        "sandtrack",
        "gran_paradiso_island",
        "snowtuxpeak",
        "cornfield_crossing",
    ),
    "medium": ("black_forest", "olivermath"),
    "hard": ("zengarden", "stk_enterprise", "minigolf"),
}
HELD_OUT_GROUPS = {
    "simple": ("abyss", "candela_city", "cocoa_temple", "hacienda", "volcano_island"),
    "medium": ("mines", "snowmountain"),
    "hard": ("ravenbridge_mansion", "fortmagma"),
}
TRAIN_TRACKS = tuple(track for tracks in TRAIN_GROUPS.values() for track in tracks)
HELD_OUT_TRACKS = tuple(
    track for tracks in HELD_OUT_GROUPS.values() for track in tracks
)
ALL_TRACKS = TRAIN_TRACKS + HELD_OUT_TRACKS


def track_group(track):
    for split, groups in (("train", TRAIN_GROUPS), ("held_out", HELD_OUT_GROUPS)):
        for difficulty, tracks in groups.items():
            if track in tracks:
                return split, difficulty
    raise ValueError(f"Unspecified study track: {track}")
