"""Auto-generated marine taxonomy stub. Refine in Phase 5 with WoRMS lookups.
   GROUPS values are 0-indexed class_idx (NOT COCO category_id).
"""

from typing import Iterable

GROUPS: dict[str, list[int]] = {'other': [0, 2, 4, 8, 9, 14, 15, 17, 18], 'cnidarian': [1, 5, 12, 20, 21, 27, 30], 'worm': [3], 'fish': [6], 'echinoderm': [7, 11, 19, 25, 31], 'crustacean': [10, 13, 26, 29], 'cephalopod': [16], 'mollusk_other': [22, 23, 24], 'sponge': [28]}

_idx_to_group = {idx: g for g, members in GROUPS.items() for idx in members}

def group_of(class_idx: int) -> str:
    return _idx_to_group.get(class_idx, "other")

def members_of(group: str) -> Iterable[int]:
    return GROUPS.get(group, [])
