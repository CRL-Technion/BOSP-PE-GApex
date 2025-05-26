# Efficient Approximate Bi-Objective Shortest-Path Computation in the Presence of Correlated Objectives
A Python and C++ combined implementation of the algorithms described in the paper [ref]. The C++ code was originally forked from [Han Zhang's A\*pex repo](https://github.com/HanZhang39/A-pex).
The repository is comprised of two main components:
1. **Preprocesing**: identifying correlated clusters within a DIMACS graph and genearting a new, generalized graph representation enriched with super-edges.
2. **Query**: implementation of the PE-GA\*pex algorithm, an extension to the A\*pex algorithm endowed with Partial Expansion (PE) for performance.

## Preprocessing
The preprocessing code is mainly implemented in Python. The entry `main` is located in file `preprocessing.py`. Please update all the variables under the `Definitions` section. 
Note that the preprocessing algorithm calls some C++ code under the `/CPP` folder. At the end of this phase, a new graph representation (in the standard DIMACS format) will be generated, 
ready to be used in the query phase.

## Query
The query phase code can be executed by running `multiobj.exe`. The executable implements several bi-objective search algorithm, specifically A\*pex and PE-GA\*pex.

An example of executing **A\*pex** on a given query file:
```
multiobj.exe -m {DIMACS_distance_filename} {DIMACS_time_filename} -e {epsilon approximation factor} -q {query_filename} -a Apex -o {outputfile} -l {log_file} -t {timeout}'
```

An example of executing **PE-GA\*pex** on a given query file:
```
multiobj.exe -m {DIMACS_distance_filename} {DIMACS_time_filename} -e {epsilon approximation factor} -q {query_filename} -a GApex_PE -o {outputfile} -l {log_file} -t {timeout}'
```
