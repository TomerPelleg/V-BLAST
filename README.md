# V-BLAST Implementations and Experiments

The following repository is the code for the paper <TODO - CHANGE AFTER PUBLICATION>. The main feature is taking a
matrix, and computing a
permutation of the columns. The permutations are either V-BLAST, or some other permutations which we claim have similar
properties. This project uses `uv`, and assumes you have it installed.

## V-BLAST Implementations

This project has various implementations of V-BLAST both in Python and in C. The implementations are found in the
`columns_ordering_algorithms/exact_vblast` directory. The implementations are:

| File Name              | Explanation                                                                                                                                | Source (Paper)                                                                 |
|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| `chang_paige.py`       | Based on Chang and Paige algorithms                                                                                                        | [doi:10.1016/j.apnum.2007.01.008](https://doi.org/10.1016/j.apnum.2007.01.008) | 
| `chang_paige_paper.py` | Same as the above, but with the notations we had in our paper (used to check we didn't have errors)                                        | [doi:10.1016/j.apnum.2007.01.008](https://doi.org/10.1016/j.apnum.2007.01.008) |
| `computing_inverse.py` | Loosely Based on Benesty, Chen, and Huang Paper                                                                                            | [doi:10.1109/tsp.2003.812897](https://doi.org/10.1109/tsp.2003.812897)        | 
| `householder.py`       | Our contribution. The same idea as Chang and Paige, but we use Householder reflections and not Givens rotations                            | CHANGE AFTER PUBLICATION                                                       |
| `naive.py`             | Naive implementation that in any step takes the column maximizing the current diagonal element. Has worse complexity O(n^4) and not O(n^3) | -                                                                              |

We also have approximate V-BLAST algorithms: algorithms that compute permutations that are claimed to have similar
properties, and are more efficient to compute. They are in the `columns_ordering_algorithms/approximate_vblast`
directory.

| File Name                     | Explanation      | Source (Paper)                                                 |
|-------------------------------|------------------|----------------------------------------------------------------|
| `sqrd.py`                     | Based on SQRD    | [doi:10.1049/el:20010899](https://doi.org/10.1049/el:20010899) | 
| `repeated_random_sampling.py` | Our contribution | CHANGE AFTER PUBLICATION                                       |

### C code

We also have C code in the respective directories under a directory named `c`. We have a CMake file for compiling in
`CMakeLists.txt`. We built our code on Linux with gcc. The C implementations require MKL, and we used the version
supplied with OneAPI version 2026.1. You may have to change the CMake if you want to use a different version, and almost
surely need to change it if you either don't use Linux or have a non-Intel processor.

#### Compiling the C code

To compile the C code, run

```bash
./build.sh 
```

From the `columns_ordering_algorithms` directory. If the compilation succeeded, you will see `.so` files prefixed by `_`
in the relevant directories (e.g. `columns_ordering_algorithms/exact_vblast/_householder.so`).

### Testing Implementations

The entrypoint of the `columns_ordering_algorithms/exact_vblast` directory tests whether different V-BLAST
implementations return the same permutation across many matrices. It generates random matrices (with random sizes, not
necessarily square), runs the requested algorithms and reports if they are all the same (and also the average runtime).
You can run it with

```
uv run columns_ordering_algorithms/exact_vblast/main.py
```

Add new implementations if you want! (directly to the file, sorry)

# Experiments

The other main feature of this repository is the ability to replicate all the experiments from the thesis. It includes
an implementation of Schnorr Euchner algorithm (in the `schnorr_euchner` directory), and also all the experiments
framework in the `experiments` directory. The experiments allow you to supply a config for the experiments, and will run
the requested experiments on a single core (if you give the relevant env variables, supplied in the .env for my Linux
machine). An example run is

```bash
PYTHONPATH=.. uv run --env-file .env main.py --config configs/iterations_ecdf_size_50.json
```

from the `experiments` subdirectory. You can find in the `configs` directory all the configurations used for the graphs
in the paper, and of course create your own. Note that some experiments may take a long time to complete (~100 hours),
depending on your machine.

# Tests

During the development process, we maintained some tests for the algorithms and classes (mainly for the C
implementations). They use pytest, and you can run them from the `tests` subdirectory:

```bash
PYTHONPATH=.. uv run pytest
```

Consider adding the tests to a git hook on push if you do heavy changes and want to maintain them.