from __future__ import print_function

import argparse
import sys

import numpy as np
from PIL import Image
from mpi4py import MPI

from skimage.measure import block_reduce

from evolver import Evolver, Individual


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def quantize(image, levels):
    bins = np.linspace(0.0, 1.0, num=levels)
    bins_cns = (len(bins) - 1) * 1.0
    return (np.digitize(image, bins) - 1) / (bins_cns - 1)


kernel_size = (2, 2)
quanta = 4
sample = None
reduced_sample = None

class EvoBitmap(Individual):
    def __init__(self, genome=None):
        Individual.__init__(self)
        self.genome = np.random.rand(*sample.shape) if genome is None else genome

    def __repr__(self):
        return "%s - %1.5f" % (self.iid, self.fitness)


def generator():
    return EvoBitmap()


def mutator(individual):
    individual.genome = matrix_mutate(individual.genome, 0.10)


def breeder(p1, p2):
    child = EvoBitmap(matrix_breed(p1.genome, p2.genome))
    return child


def fitness(individual):
    image = quantize(individual.genome, quanta)
    individual.fitness = matrix_fitness(image)
    return individual.fitness


def process_alpha(epoch, alpha_individual):
    if epoch % 20 == 0:
        quantized = quantize(alpha_individual.genome, quanta)
        alpha = quantized * 255
        img = alpha.astype(np.uint8)
        Image.fromarray(img).save("./out/%d_out_%1.5f.png" % (epoch, alpha_individual.fitness))


def matrix_mutate(genome, p):
    s = genome.shape
    pattern = np.random.binomial(1, p, s)
    inv_pattern = 1 - pattern
    old_genome = np.multiply(genome, inv_pattern)
    mutation = np.multiply(np.random.random(s), pattern)
    return old_genome + mutation


def matrix_breed(genome1, genome2):
    s = genome1.shape
    pattern = np.random.binomial(1, 0.5, s)
    inv_pattern = 1 - pattern
    p1_candidate = np.multiply(genome1, pattern)
    p2_candidate = np.multiply(genome2, inv_pattern)
    return p1_candidate + p2_candidate


def matrix_fitness(genome):
    reduced = block_reduce(genome, kernel_size, func=np.mean)
    fit = -((reduced_sample - reduced) ** 2).sum()
    return fit


def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if size == 1:
        print("Can not evolve on single node, 2 nodes required at least!")
        sys.exit(-1)

    parser = argparse.ArgumentParser(description='Distributed evolver')
    parser.add_argument('-p',"--popsize", default=(size - 1), type=int,
                        help='population size (default: equals to node count-1 [%d])' % (size - 1))
    parser.add_argument('-e',"--epochs", default=1, type=int,
                        help='epochs count (default: 1)')
    parser.add_argument("-i", "--input", required=True, help='input bitmap')

    args = parser.parse_args()

    per_node = int(np.rint(args.popsize / (size - 1)))
    pop_size = per_node * (size - 1)


    global sample, reduced_sample

    im = Image.open(args.input)
    sample = np.array(im)[:, :, :3]
    sample = rgb2gray(sample) / 255.0

    reduced_sample = block_reduce(sample, kernel_size, func=np.mean)
    smpl = reduced_sample * 255
    img = smpl.astype(np.uint8)

    Image.fromarray(img).save("sample.png")

    evo = Evolver(args.epochs, pop_size, 0.5, 0.15, generator, fitness, breeder, mutator, process_alpha)

    rank = comm.Get_rank()
    if rank == 0:
        if pop_size != args.popsize:
            print("Population size adjusted to %d [%d per node x %d nodes + master]" % (pop_size, per_node, size - 1))
        evo.master()
    else:
        evo.slave()


if __name__ == '__main__':
    main()
