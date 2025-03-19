"""Problem thermal condition enumeration for thermoelastic2d problem."""

from thermoelastic2d.utils import get_res_bounds



import numpy as np
import random
from itertools import product, combinations
from typing import Any, Dict, List, Tuple, FrozenSet


class ThermalEnumeration:
    """
    A class to enumerate thermal boundary conditions for a 2D domain.

    Each thermal boundary condition is represented as a frozenset of key-value tuples:
      - "heatsink_elements": tuple of indices where the heatsinks are applied
      - "volfrac": volume fraction value

    The parameters are:
      - num_heatsinks: Number of heatsinks to place.
      - heatsink_size: The size (number of contiguous elements) of each heatsink.
      - heatsink_locations: Allowed boundaries (e.g. "L", "T", "LT", "LTB", "LB").
      - volfrac: Volume fraction.

    The default (training/validation) grid is:
      - num_heatsinks: {1, 2, 3}
      - heatsink_size: {5, 9, 13, 17}
      - heatsink_locations: {"L", "T", "LT", "LTB"}
      - volfrac: {0.25, 0.26, ..., 0.4}

    Test dataset overrides:
      - Test 1: num_heatsinks -> {4, 5}
      - Test 2: heatsink_size -> {21, 25}
      - Test 3: heatsink_locations -> {"LB"}
      - Test 4: volfrac -> {0.2, 0.21, ..., 0.24}
    """

    def __init__(self, nelx: int, nely: int):

        # The placement of effects (e.g. fixed, loaded, heatink) happens on the edges of the elements, so we need to add 1 to the number of elements
        lci, tri, rci, bri = get_res_bounds(nelx + 1, nely + 1)
        self.lci = lci
        self.rci = rci
        self.bri = bri
        self.tri = tri

        # Default training/validation parameters.
        self.default_params: Dict[str, List[Any]] = {
            "num_heatsinks": [1, 2, 3],
            "heatsink_size": [5, 9, 13, 17],
            "heatsink_locations": ["L", "T", "LT", "LTB"],
            "volfrac": [0.3]
            # "volfrac": [round(x, 2) for x in np.arange(0.25, 0.41, 0.01)]
        }

        # Test dataset overrides.
        self.test_overrides: Dict[str, Dict[str, List[Any]]] = {
            "test1": {"num_heatsinks": [4, 5]},
            "test2": {"heatsink_size": [21, 25]},
            "test3": {"heatsink_locations": ["LB"]},
            "test4": {"volfrac": [round(x, 2) for x in np.arange(0.2, 0.25, 0.01)]}
        }

    def _enumerate_placements(self, allowed_edges: Dict[str, List[int]], num: int, size: int) -> List[Tuple[int, ...]]:
        """
        Enumerate all valid placements of contiguous segments of a given size on allowed edges.
        For multiple heatsinks (num > 1), only combinations with non-overlapping segments (on the same edge)
        are retained.

        Returns:
          A list of placements; each placement is a tuple of indices.
        """
        segments = []
        for edge, candidates in allowed_edges.items():
            if len(candidates) < size:
                continue
            for i in range(len(candidates) - size + 1):
                segments.append((edge, tuple(candidates[i:i + size]), i, i + size))

        if num == 1:
            return [seg[1] for seg in segments]

        valid_placements = []
        for combo in combinations(segments, num):
            valid = True
            groups: Dict[str, List[Tuple[int, int]]] = {}
            for seg in combo:
                edge_label, seg_tuple, start, end = seg
                groups.setdefault(edge_label, []).append((start, end))
            for segs in groups.values():
                segs_sorted = sorted(segs, key=lambda x: x[0])
                for i in range(len(segs_sorted) - 1):
                    if segs_sorted[i][1] > segs_sorted[i + 1][0]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                combined = []
                for seg in combo:
                    combined.extend(seg[1])
                combined = tuple(sorted(combined))
                valid_placements.append(combined)
        return valid_placements

    def _sample_placements(self, allowed_edges: Dict[str, List[int]], num: int, size: int) -> List[Tuple[int, ...]]:
        """
        Instead of enumerating all placements, this method randomly samples a single valid placement.
        For multiple heatsinks, it repeatedly samples until a valid, non-overlapping combination is found.

        Returns:
          A list containing one placement (tuple of indices), or an empty list if none found.
        """
        segments = []
        for edge, candidates in allowed_edges.items():
            if len(candidates) < size:
                continue
            for i in range(len(candidates) - size + 1):
                segments.append((edge, tuple(candidates[i:i + size]), i, i + size))

        if not segments:
            return []

        if num == 1:
            return [random.choice(segments)[1]]

        max_attempts = 1000
        for _ in range(max_attempts):
            candidate_combo = random.sample(segments, num)
            valid = True
            groups: Dict[str, List[Tuple[int, int]]] = {}
            for seg in candidate_combo:
                edge, seg_tuple, start, end = seg
                groups.setdefault(edge, []).append((start, end))
            for segs in groups.values():
                segs_sorted = sorted(segs, key=lambda x: x[0])
                for i in range(len(segs_sorted) - 1):
                    if segs_sorted[i][1] > segs_sorted[i + 1][0]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                combined = []
                for seg in candidate_combo:
                    combined.extend(seg[1])
                return [tuple(sorted(combined))]
        return []

    def _construct_boundary_condition(self, params: Dict[str, Any], sample: bool = False) -> List[
        FrozenSet[Tuple[str, Any]]]:
        """
        Constructs thermal boundary conditions based on the provided parameter dictionary.
        Depending on the 'sample' flag, the heatsink placements are either fully enumerated or
        a single valid placement is sampled.

        Returns:
          A list of frozensets, each representing a thermal boundary condition.
        """
        # Mapping for heatsink locations.
        heatsink_location_map: Dict[str, Dict[str, List[int]]] = {
            "L": {"L": self.lci},
            "R": {"R": self.rci},
            "T": {"T": self.tri},
            "B": {"B": self.bri},
            "LT": {"L": self.lci, "T": self.tri},
            "LTB": {"L": self.lci, "T": self.tri, "B": self.bri},
            "LB": {"L": self.lci, "B": self.bri}
        }

        allowed_heatsinks = heatsink_location_map[params["heatsink_locations"]]

        # Choose enumeration or sampling based on the flag.
        if sample:
            heatsink_placements = self._sample_placements(allowed_heatsinks, params["num_heatsinks"],
                                                          params["heatsink_size"])
        else:
            heatsink_placements = self._enumerate_placements(allowed_heatsinks, params["num_heatsinks"],
                                                             params["heatsink_size"])

        conditions = []
        for placement in heatsink_placements:
            condition = frozenset({
                ("heatsink_elements", placement),
                ("volfrac", params["volfrac"])
            })
            conditions.append(condition)
        return conditions

    def _get_param_grid(self, dataset: str) -> Dict[str, List[Any]]:
        """
        Returns the parameter grid for the given dataset.
        For training/validation, the full default grid is returned.
        For test datasets, parameters not overridden are fixed to the first default value.
        """
        dataset_key = dataset.lower()
        if dataset_key in ["training", "validation"]:
            return self.default_params
        elif dataset_key in self.test_overrides:
            override = self.test_overrides[dataset_key]
            grid: Dict[str, List[Any]] = {}
            for key, default_vals in self.default_params.items():
                if key in override:
                    grid[key] = override[key]
                else:
                    grid[key] = default_vals
            return grid
        else:
            raise ValueError(f"Dataset '{dataset}' not recognized. Valid options are 'training', 'validation', "
                             f"or one of {list(self.test_overrides.keys())}.")

    def enumerate_conditions(self, dataset: str) -> List[FrozenSet[Tuple[str, Any]]]:
        """
        Enumerates and returns the set of thermal boundary conditions (as frozensets)
        for the specified dataset. If sample is True, then for each parameter combination only
        one (randomly sampled) heatsink placement is generated.
        """
        grid = self._get_param_grid(dataset)
        keys = list(grid.keys())
        all_conditions: List[FrozenSet[Tuple[str, Any]]] = []

        for values in product(*(grid[key] for key in keys)):
            param_combo = dict(zip(keys, values))
            conditions = self._construct_boundary_condition(param_combo, sample=False)
            all_conditions.extend(conditions)
        return all_conditions


    def sample_conditions(self, dataset: str, sample_size: int = 1000) -> List[FrozenSet[Tuple[str, Any]]]:
        """
        Enumerates and returns the set of thermal boundary conditions (as frozensets)
        for the specified dataset. If sample is True, then for each parameter combination only
        one (randomly sampled) heatsink placement is generated.
        """
        grid = self._get_param_grid(dataset)
        keys = list(grid.keys())
        all_conditions: List[FrozenSet[Tuple[str, Any]]] = []

        while len(all_conditions) < sample_size:
            for values in product(*(grid[key] for key in keys)):
                param_combo = dict(zip(keys, values))
                conditions = self._construct_boundary_condition(param_combo, sample=True)
                all_conditions.extend(conditions)
                if len(all_conditions) >= sample_size:
                    break
        return all_conditions



if __name__ == "__main__":

    enumerator = ThermalEnumeration(64, 64)

    # For training/validation: full enumeration (2304 conditions in this example)
    # training_conditions = enumerator.enumerate_conditions("training")
    training_conditions = enumerator.sample_conditions("training", sample_size=1000)
    print(f"Training conditions: {len(training_conditions)} boundary conditions generated.")

    # For Test 3 (only support_locations is overridden):
    # test3_conditions = enumerator.enumerate_conditions("test3")
    test3_conditions = enumerator.sample_conditions("test3", sample_size=100)
    print(f"Test 3 conditions: {len(test3_conditions)} boundary condition(s) generated.")

    # Print one example condition:
    if training_conditions:
        print("Example training condition:")
        for key, value in sorted(training_conditions[0]):
            print(f"  {key}: {value}")






