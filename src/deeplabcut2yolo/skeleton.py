# project name is licensed under GNU General Public License v3.0, see LICENSE.
# Copyright 2024 Sira Pornsiriprasert <code@psira.me>


def get_flip_idx(n_classes: int, symmetric_pairs: list[tuple]) -> list[int]:
    flip_idx = list(range(n_classes))
    for a, b in symmetric_pairs:
        flip_idx[a], flip_idx[b] = flip_idx[b], flip_idx[a]
    return flip_idx


# TODO get_flip_idx_from_skeleton()
