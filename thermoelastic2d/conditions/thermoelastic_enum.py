from typing import Any, Dict, List
import random
import numpy as np

from thermoelastic2d.conditions.elastic_enum import ElasticEnumeration
from thermoelastic2d.conditions.thermal_enum import ThermalEnumeration

class ThermoelasticEnumeration:
    """
    A class for sampling thermoelastic boundary conditions by merging
    thermal and elastic enumerations. This wrapper allows you to specify separate datasets
    for the elastic and thermal domains.

    Each thermoelastic condition is produced by merging a sampled elastic condition and a
    sampled thermal condition (converted to dictionaries). If both domains use the same key
    (e.g., 'volfrac'), the thermal domain's value will override the elastic one.

    Example usage:
        nelx, nely = 64, 64
        thermo_enum = ThermoelasticEnumeration(nelx, nely)
        conditions = thermo_enum.sample_conditions(
            elastic_dataset="test1",
            thermal_dataset="test1",
            sample_size=1000
        )
    """

    def __init__(self, nelx: int, nely: int):

        self.elastic_enumerator = ElasticEnumeration(nelx, nely)
        self.thermal_enumerator = ThermalEnumeration(nelx, nely)

        self.volfrac_set = [round(x, 2) for x in np.arange(0.25, 0.41, 0.01)]

    def sample_conditions(self,
                          elastic_dataset: str = "training",
                          thermal_dataset: str = "training",
                          sample_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Samples a list of thermoelastic boundary conditions by merging a sampled elastic condition
        with a sampled thermal condition for each entry.

        Parameters:
            elastic_dataset: The dataset type for elastic conditions (e.g. "training", "test1", etc.)
            thermal_dataset: The dataset type for thermal conditions.
            sample_size: The number of combined conditions to sample.

        Returns:
            A list of dictionaries, each representing a combined thermoelastic condition.
            (Note: If keys overlap, values from the thermal condition will override those from elastic.)
        """
        # Sample from each enumerator
        elastic_samples = self.elastic_enumerator.sample_conditions(elastic_dataset, sample_size=sample_size)
        thermal_samples = self.thermal_enumerator.sample_conditions(thermal_dataset, sample_size=sample_size)


        # Combine conditions by pairing a random elastic and a random thermal condition.
        combined_conditions = []
        for idx in range(sample_size):
            elastic_condition = dict(random.choice(elastic_samples))
            thermal_condition = dict(random.choice(thermal_samples))
            combined_condition = {**elastic_condition, **thermal_condition}
            combined_condition['volfrac'] = random.choice(self.volfrac_set)
            combined_conditions.append(combined_condition)
        return combined_conditions

    def enumerate_conditions(self,
                             elastic_dataset: str = "training",
                             thermal_dataset: str = "training") -> List[Dict[str, Any]]:
        """
        Enumerates all thermoelastic boundary conditions by taking the Cartesian product
        of the elastic and thermal conditions. Note that this can yield a very large number of combinations.

        Parameters:
            elastic_dataset: The dataset type for elastic conditions.
            thermal_dataset: The dataset type for thermal conditions.

        Returns:
            A list of dictionaries, each representing a merged thermoelastic condition.
        """
        elastic_enumeration = self.elastic_enumerator.enumerate_conditions(elastic_dataset)
        thermal_enumeration = self.thermal_enumerator.enumerate_conditions(thermal_dataset)

        combined_conditions = []
        for e_cond in elastic_enumeration:
            for t_cond in thermal_enumeration:
                # Convert frozensets to dictionaries and merge them.
                combined_condition = {**dict(e_cond), **dict(t_cond)}
                combined_condition['volfrac'] = random.choice(self.volfrac_set)
                combined_conditions.append(combined_condition)
        return combined_conditions


# Example usage:
if __name__ == "__main__":
    # Example: obtaining resolution bounds and instantiating the enumerators.
    nelx, nely = 64, 64
    thermoelastic_enumerator = ThermoelasticEnumeration(nelx, nely)

    # Sampling combined conditions: specify dataset for each physics domain.
    combined_conditions = thermoelastic_enumerator.sample_conditions(
        elastic_dataset="test1", thermal_dataset="test1", sample_size=1000
    )
    print(f"Combined thermoelastic conditions: {len(combined_conditions)} conditions sampled.")

    # Display one example condition.
    if combined_conditions:
        print("Example thermoelastic condition:")
        for key, value in sorted(combined_conditions[0].items()):
            print(f"  {key}: {value}")