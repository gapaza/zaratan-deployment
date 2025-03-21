# zaratan-deployment

This holds the code for enumerating and evaluating a set of boundary conditions for a 2D thermoelastic toplogy optimization problem.




## Installation


The python library requirements can be found in the `requirements.txt` file. To install the required libraries, run the following command from the root directory of the repository:

```bash 
pip install -r requirements.txt
```



## Usage


The main script for running the optimization is `main.py`, <b>which should be ran from the root directory of this project</b>. The script has a set of command line arguments that can be passed. To see the list of arguments, run the following command from the root directory of the repository:

```bash
python main.py --help
```


This will generate the following output:

```bash
usage: main.py [-h] [--num-procs NUM_PROCS] [--samples SAMPLES] [--save-dir SAVE_DIR]

Generate a dataset using multiple processes with configurable parameters.

options:
  -h, --help            show this help message and exit
  --num-procs NUM_PROCS
                        Number of processes to use (default: 5)
  --samples SAMPLES     Number of samples to generate (default: 5)
  --save-dir SAVE_DIR   Directory to save the dataset (default: /Users/gapaza/repos/datasets/thermoelastic2dv1)
```

Basically, you can specify the number of concurrent optimization processes to run (`--num-procs`), the number of samples to generate (`--samples`), and the directory to save the dataset (`--save-dir`).



An example module-based call to the program with 5 samples and 5 processes might look like:

```bash
python3 -m main --num-procs 5 --samples 5 --save-dir /path/to/save/directory
```

When the program finishes, there will be a set of pickle files in the specified directory.
Here, each file contains a triple of optimized designs:
- An optimized elastic design
- An optimized thermal design
- An optimized thermoelastic design

Here, the boundary conditions for the thermoelastic design are the elastic and thermal boundary conditions combined.



## Dataset

This code is capable of generating training, validation, and test datasets.
The purpose of validation dataset is to assess how well the model generalizes to new combinations of seen boundary conditions.
The purpose of the testing datasets is to assess how well the model can extrapolate to unseen boundary conditions.
The parameter ranges covered in these datasets are show in the enumeration tables below:


### Structural Enumeration Table
| Parameters              | Training / Validation       | Test 1    | Test 2    | Test 3    | Test 4    | Test 5    | Test 6    | Test 7                   |
|-------------------------|---------------------------|-----------|-----------|-----------|-----------|-----------|-----------|--------------------------|
| Number of Supports     | {2, 3, 4}                 | {5, 6}    | ------    | ------    | ------    | ------    | ------    | ------                   |
| Support Size (n-elements) | {1, 3, 5}             | ------    | {1, 7}    | ------    | ------    | ------    | ------    | ------                   |
| Support Locations      | {L, T, LT, LTB}           | ------    | ------    | {LB}      | ------    | ------    | ------    | ------                   |
| Number of Loads       | {1, 2}                     | ------    | ------    | ------    | {3, 4}    | ------    | ------    | ------                   |
| Load Size (n-elements) | {1}                       | ------    | ------    | ------    | ------    | ------    | ------    | ------                   |
| Load Directions       | {x, y}                     | ------    | ------    | ------    | ------    | {xy}      | ------    | ------                   |
| Load Placements      | {R}                         | ------    | ------    | ------    | ------    | ------    | {B}       | ------                   |
| Volume Fraction      | {0.25, 0.26, ..., 0.4}      | ------    | ------    | ------    | ------    | ------    | ------    | {0.2, 0.21, ..., 0.24}  |



### Thermal Enumeration Table
| Parameters            | Training / Validation       | Test 1    | Test 2    | Test 3    | Test 4                  |
|-----------------------|---------------------------|-----------|-----------|-----------|--------------------------|
| Number of Heatsinks  | {1, 2, 3}                 | {4, 5}    | ------    | ------    | ------                   |
| Heatsink Size (n-elements) | {5, 9, 13, 17}       | ------    | {21, 25}  | ------    | ------                   |
| Heatsink Locations  | {L, T, LT, LTB}           | ------    | ------    | {LB}      | ------                   |
| Volume Fraction     | {0.25, 0.26, ..., 0.4}    | ------    | ------    | ------    | {0.2, ..., 0.24}        |














