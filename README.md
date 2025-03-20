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





















