"""Problem elastic condition enumeration for thermoelastic2d problem."""

from thermoelastic2d.utils import get_res_bounds

import numpy as np
import random
from itertools import product, combinations
from typing import Any, Dict, List, Tuple, FrozenSet


class ElasticEnumeration:
    """
    A class to enumerate elastic boundary conditions for a 2D topology optimization problem.

    For each parameter combination, it can either enumerate all possible contiguous placements
    of supports (fixed elements) and loads (force elements) along specified boundaries or
    intelligently sample a single valid combination.

    The boundaries are kept separate so that, for example, if supports may be placed on both
    the left and right edges, a contiguous support segment will never span across them.

    The final boundary condition is a frozenset of key-value tuples:
      - "fixed_elements": tuple of indices from supports
      - "force_elements_x": tuple of indices (if forces in x-direction are applied)
      - "force_elements_y": tuple of indices (if forces in y-direction are applied)
      - "volfrac": volume fraction value
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
            "num_supports": [2, 3, 4],
            "support_size": [1, 3, 5],
            "support_locations": ["L", "T", "LT", "LTB"],
            "num_loads": [1, 2],
            "load_size": [1],
            "load_directions": ["x", "y"],
            "load_placements": ["R"],
            "volfrac": [0.3]
            # "volfrac": [round(x, 2) for x in np.arange(0.25, 0.41, 0.01)]
        }

        # Overrides for each test dataset.
        self.test_overrides: Dict[str, Dict[str, List[Any]]] = {
            "test1": {"num_supports": [5, 6]},
            "test2": {"support_size": [1, 7]},
            "test3": {"support_locations": ["LB"]},
            "test4": {"num_loads": [3, 4]},
            "test5": {"load_directions": ["xy"]},
            "test6": {"load_placements": ["B"]},
            "test7": {"volfrac": [round(x, 2) for x in np.arange(0.2, 0.25, 0.01)]}
        }

    def _enumerate_placements(self, allowed_edges: Dict[str, List[int]], num: int, size: int) -> List[Tuple[int, ...]]:
        """
        Enumerate all valid placements (as tuples of indices) of a contiguous segment of a given size
        on the allowed edges. If more than one placement is requested (num > 1), then the returned combination
        will consist of segments that do not overlap if they come from the same edge.

        Returns:
          A list of placements, where each placement is a tuple of indices.
        """
        # First, enumerate all contiguous segments on each allowed edge.
        segments = []
        for edge, candidates in allowed_edges.items():
            if len(candidates) < size:
                continue  # Not enough indices on this edge.
            for i in range(len(candidates) - size + 1):
                seg = candidates[i:i + size]
                segments.append((edge, tuple(seg), i, i + size))

        # If only one segment is requested, return all segments (just the segment indices).
        if num == 1:
            return [seg[1] for seg in segments]

        # For num > 1, enumerate all combinations and filter out those with overlapping segments on the same edge.
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
        Instead of enumerating all possible contiguous segments, this method intelligently samples a single valid placement.
        For a single segment (num == 1), it randomly chooses one of the possible segments.
        For multiple segments, it repeatedly samples a combination of segments until a valid non-overlapping set is found.

        Returns:
          A list containing one placement (as a tuple of indices) if a valid combination is found,
          or an empty list if not.
        """
        # Enumerate segments as in _enumerate_placements.
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
        # Fallback: if no valid combination was found in max_attempts, return empty list.
        return []

    def _construct_boundary_condition(self, params: Dict[str, Any], sample: bool = False) -> List[
        FrozenSet[Tuple[str, Any]]]:
        """
        Given a parameter dictionary, constructs boundary conditions.
        Depending on the 'sample' flag, it either enumerates all valid placements or
        intelligently samples a single valid placement for supports and loads.
        """
        # Mapping for supports.
        support_location_map: Dict[str, Dict[str, List[int]]] = {
            "L": {"L": self.lci},
            "R": {"R": self.rci},
            "T": {"T": self.tri},
            "B": {"B": self.bri},
            "LT": {"L": self.lci, "T": self.tri},
            "LTB": {"L": self.lci, "T": self.tri, "B": self.bri},
            "LB": {"L": self.lci, "B": self.bri}
        }
        # Mapping for loads.
        load_location_map: Dict[str, Dict[str, List[int]]] = {
            "L": {"L": self.lci},
            "R": {"R": self.rci},
            "T": {"T": self.tri},
            "B": {"B": self.bri},
        }

        # Choose enumeration or sampling based on the flag.
        if sample:
            support_func = self._sample_placements
            load_func = self._sample_placements
        else:
            support_func = self._enumerate_placements
            load_func = self._enumerate_placements

        allowed_supports = support_location_map[params["support_locations"]]
        support_placements = support_func(
            allowed_edges=allowed_supports,
            num=params["num_supports"],
            size=params["support_size"]
        )

        allowed_loads = load_location_map[params["load_placements"]]
        load_placements = load_func(
            allowed_edges=allowed_loads,
            num=params["num_loads"],
            size=params["load_size"]
        )

        conditions = []
        for sp in support_placements:
            for lp in load_placements:
                if params["load_directions"] == "x":
                    force_elements_x = lp
                    force_elements_y = tuple()
                elif params["load_directions"] == "y":
                    force_elements_x = tuple()
                    force_elements_y = lp
                elif params["load_directions"] == "xy":
                    force_elements_x = lp
                    force_elements_y = lp
                else:
                    force_elements_x = lp
                    force_elements_y = tuple()
                condition = frozenset({
                    ("fixed_elements", sp),
                    ("force_elements_x", force_elements_x),
                    ("force_elements_y", force_elements_y),
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
        Enumerates and returns the full set of boundary conditions (as frozensets)
        for the specified dataset. If sample is True, then for each parameter combination only
        one (randomly sampled) support and load placement is generated.
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
        Samples a set of boundary conditions (as frozensets) for the specified dataset.
        """
        grid = self._get_param_grid(dataset)
        keys = list(grid.keys())
        all_conditions: List[FrozenSet[Tuple[str, Any]]] = []


        num_samples = np.prod([len(grid[key]) for key in keys])
        print('Total number of conditions:', num_samples)

        while len(all_conditions) < sample_size:
            for values in product(*(grid[key] for key in keys)):
                param_combo = dict(zip(keys, values))
                conditions = self._construct_boundary_condition(param_combo, sample=True)
                all_conditions.extend(conditions)
                if len(all_conditions) >= sample_size:
                    break

        return all_conditions



if __name__ == "__main__":

    enumerator = ElasticEnumeration(64, 64)

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
















