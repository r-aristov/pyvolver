pyvolver
=============

pyvolver is a distributed python implementation of genetic algorithm, using mpi4py library to communicate between nodes.

Implementation information
-------

Initial generation, mutations, breeding and evaluation are executed on slave nodes (estimators) concurrently. 
Evaluation results (list of float values of fitness-function) are then sent to master node, deciding which individuals should be 
bred, killed (resulting in free slot on estimator node) or mutated. After that master node estimates potential free slots on estimator nodes
and issues one of several possible commands: 
 - breed and keep  	- if both parents are on the same node and there is slot available
 - breed and reloc 	- if both parents are on the same node and there is NO slot available
 - reloc 			- if individual should be sent to some other node for breeding
 
Combined with few other commands this allows pyvolver to significantly reduce data transfer overhead as individuals are sent via network
only if they can not be bred without network interaction. As individuals are transferred peer-to-peer between nodes, there is no network bottleneck 
on master node - that feature improves performance on computational clusters. 

Dependencies:
 - mpi4py	(http://mpi4py.scipy.org/docs/)
 - numpy	(http://www.numpy.org/)

Usage
-------

Whole mechanism is completely transparent. Evolving any parameters requires to derrive new class from base Individual to contain genome and
to define few callback functions, implementing task-specific breeding and mutation.
After that the only thing you need to do is to create Evolver instance like this:

evo = Evolver(epochs, population_size, decimation_rate, mutation_rate, generator, fitness, breeder, mutator, process_alpha)

and call evo.master() on master node and evo.slave() on estimators.

Example
-------

This repository contains sample code in bitmap_evolver.py (skimage is required to run). 
Evolution can be started by following command: 

mpiexec -n NUMBER_OF_NODES python bitmap_evolver.py -p POPULATION_SIZE -e EPOCHS_TO_EVOLVE -i test.png

Bitmap evolver takes sample image specified by "-i" key and tries to evolve grayscale version of it using only 4 colors in such way, that
block_reduced (with kernel 2x2) evolved bitmap looks like block_reduced sample. It's fun to watch picture emerging from pure chaos!

