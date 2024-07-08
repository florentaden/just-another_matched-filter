# ***Just Another Matched-Filter***

This Matched-Filter (MF) code is like the others, it scans continuous data with
template signals. I have written this code during my PhD and used it on seismic data. Might not be the fastest (does not run on GPUs) but works like a charm. Fully in `Python`, its particularity is to use MPI through `mpi4py`, which is
also the only package you need.

To install it with Anaconda:
```bash
conda install -c anaconda mpi4py
```

with pip:
```bash
pip install mpi4py
```

***JAM*** has been written with HPC in mind but is also efficient with the CPUs of your
home/office computer. It runs a template per CPU, the data to scan are loaded by only one CPU in each node and are accessible by all of the others CPUs of the node.
Thus, the maximum number of required CPU should be less or equal to the number of template.

There is still a little bit of work to finalize the optimization of the memory usage (call C functions?) in each node but I could scan a day of data (24h), 12 stations, 3 components at 100Hz with 156 templates in ~7min on 48 CPUs distributed on 6 nodes.


## ***What you need***

***JAM*** requires that the templates are stored in a `numpy` archive (e.g. `.npz`, can be also hdf5..) containing 3 keys: ***header***, ***data***, ***moveout***. You can also re-write this part to suit your needs or contact me if you run into any problems.

The ***header*** key should give access to a list of headers that tells you which station and component you are looking at, for example: ***STA1.HHZ***.
Can be anything as long as it matches your data.

The ***data*** gives a `numpy` array containing the data to be scanned. If there is *N* headers and *T* samples, then the array size is *NxT*.  

The ***moveout*** gives a `numpy` array containing the number of samples that have to be taken into account during the stacking of the cross-correlation functions. The moveouts correspond to the delays between the first and the other seismic wave arrival times: the station which has the first arrival has a moveout of 0.

To run ***JAMF***, you need a config file with the path to the data, to the templates and also other things such as the year or the julian day to scan.. ***(The way the config file is read might change in a close future)***.

## ***How to run it***

In a terminal:

```bash
mpirun -np the_number_of_CPU_to_use python just-another_matched-filter.py config
```

Using sbatch on a HPC platform using SLURM you can run a job such as:

```bash
#SBATCH --job-name=?
#SBATCH --output=jamf.log
#SBATCH --nodes=?
#SBATCH --ntasks-per-node=?
#SBATCH --partition=?
#SBATCH --mem=?
#SBATCH --exclusive

mpirun -np $SLURM_NTASKS python just-another_matched-filter.py config
```
