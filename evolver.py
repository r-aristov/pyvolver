from __future__ import print_function

import datetime
import platform
import random
import sys
import time

import numpy as np
from mpi4py import MPI


class Individual:
    cnt = 0

    def __init__(self):
        self.fitness = 0.0
        self.iid = Individual.cnt
        Individual.cnt += 1

    def __repr__(self):
        return "#%d [%1.3f]" % (self.iid, self.fitness)


def test_fitness(dude):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    fit = random.random()
    dude.fitness = fit

    host = platform.node()
    # print("host %s - rank %d - input %d - fit %1.5f - time %s" % (host, rank, dude.iid, fit, datetime.datetime.now()))
    delay = 0  # random.randint(5, 10)
    time.sleep(delay)
    return dude


MAX_TAG = 2 ** 16 - 1


class EvoMaster:
    def __init__(self, population_size, per_node, decimation_ratio=0.4, mutation_ratio=0.15):
        self.population_size = population_size
        self.per_node = per_node
        self.nodes = MPI.COMM_WORLD.Get_size() - 1

        self.slots = [0] * (self.nodes + 1)
        self.population = []
        self.fit_weights = []

        self.epoch = 0
        self.decimation_ratio = decimation_ratio
        self.mutation_ratio = mutation_ratio

        self.survivors_idx = int((1.0 - self.decimation_ratio) * self.population_size)

        # for i in range(0, self.nodes):
        #     self.slots.append([0]*per_node)

        break_point = 1.2
        border = 3
        lin = np.linspace(border * break_point, -border / break_point, self.survivors_idx)
        self.breed_probs = (1.0 + np.tanh(lin))
        self.breed_probs /= self.breed_probs.sum()
        self.breed_probs = list(self.breed_probs)

    def evolve(self, iterations):
        comm = MPI.COMM_WORLD
        print("Evolution started at %s, %d iterations follows" % (datetime.datetime.now(), iterations))
        start_total = time.time()

        print("Generating %d individuals " % self.population_size, end='')
        sys.stdout.flush()
        s_time = time.time()
        comm.barrier()
        print("done in %1.3fs" % (time.time() - s_time))
        sys.stdout.flush()
        print("Decimation ratio: %1.2f [%d individuals survive], mutation ratio: %1.2f [~%d individuals mutate]\n" %
              (self.decimation_ratio, self.survivors_idx,
               self.mutation_ratio, int(self.mutation_ratio*self.population_size)))
        sys.stdout.flush()
        for i in range(1, iterations + 1):
            self.epoch = i

            print("Epoch #%d: " % self.epoch, end='')
            sys.stdout.flush()
            epoch_start = time.time()

            print("eval", end='')
            sys.stdout.flush()
            s_time = time.time()
            fit_result = comm.gather([], root=0)
            fit_result = [item for sublist in fit_result for item in sublist]
            self.population = sorted(fit_result, key=lambda result: result[2], reverse=True)
            alpha_fit = self.population[0][2]
            omega_fit = self.population[self.population_size-1][2]
            print("  %1.3fs " % (time.time() - s_time), end='')
            sys.stdout.flush()

            decimate_commands = {}
            for individual in self.population[self.survivors_idx:]:
                node = individual[0]
                iid = individual[1]
                self.slots[node] += 1
                decimate_commands.setdefault(node, []).append(iid)
            for n in decimate_commands.keys():
                comm.send({'cmd': 'decimate', 'iids': decimate_commands[n]}, dest=n)

            fit_gen = [i[2] for i in self.population]
            fit_arr = np.fromiter(fit_gen, dtype=np.double)
            self.fit_weights = fit_arr / fit_arr.sum()

            breed_count = self.population_size - self.survivors_idx

            s_time = time.time()
            self.breed(breed_count)
            print("\tbreed %1.3fs " % (time.time() - s_time), end='')
            sys.stdout.flush()

            s_time = time.time()
            comm.barrier()
            print("\tmutate %1.3fs " % (time.time() - s_time), end='')
            sys.stdout.flush()

            comm.barrier()
            print("\t| done in %1.3f" % (time.time() - epoch_start), end='')
            print("\t alpha: %1.4f" % alpha_fit, end='')
            print("\t omega: %1.4f" % omega_fit)
        print("\nEvolution ended at %s in %1.3f" % (datetime.datetime.now(), time.time() - start_total))


    def get_free_node(self):
        return next((i for i, node in enumerate(self.slots) if node != 0), None)

    def check_relocs(self, relocated, src_node, src_iid, dst_node, dst_iid):
        reloc = relocated.get((dst_node, dst_iid), [])
        if len(reloc) != 0:
            print("!! %s-%s already relocated to %s" % (dst_node, dst_iid, reloc))
        reloc = relocated.get((src_node, src_iid), [])
        if len(reloc) != 0:
            print("!! %s-%s already relocated to %s" % (dst_node, dst_iid, reloc))

    def breed(self, breed_count):
        comm = MPI.COMM_WORLD
        rng = np.arange(self.population_size)

        p1_breed = np.random.choice(rng, breed_count, p=self.fit_weights)
        p2_breed = np.random.choice(rng, breed_count, p=self.fit_weights)

        # relocated = {}
        for k in range(0, breed_count):
            p1_node, p1_iid, _ = self.population[p1_breed[k]]
            p2_node, p2_iid, _ = self.population[p2_breed[k]]

            if p1_node == p2_node:
                src_node = p1_node
                if self.slots[src_node] > 0:
                    dst_node = src_node
                    comm.send({'cmd': 'breed_keep', 'iids': [p1_iid, p2_iid]}, dest=dst_node)
                else:
                    dst_node = self.get_free_node()
                    t1 = random.randint(1, MAX_TAG)
                    comm.ssend({'cmd': 'get_reloc', 'src_nodes': [p1_node], 'tags': [t1]}, dest=dst_node)
                    comm.send({'cmd': 'breed_reloc', 'iids': [p1_iid, p2_iid], 'dst_node': dst_node, 'tags': [t1]},
                              dest=src_node)
            else:
                src_iid, src_node, dst_node, dst_iid = (p2_iid, p2_node, p1_node, p1_iid) if self.slots[p1_node] > \
                                                                                             self.slots[p2_node] else (
                    p1_iid, p1_node, p2_node, p2_iid)
                if self.slots[dst_node] == 0:
                    # self.check_relocs(relocated, src_node, src_iid, dst_node, dst_iid)
                    dst_node = self.get_free_node()
                    t1 = random.randint(1, MAX_TAG)
                    t2 = random.randint(1, MAX_TAG)
                    comm.ssend({'cmd': 'get_reloc_breed', 'src_nodes': [p1_node, p2_node], 'tags': [t1, t2]},
                               dest=dst_node)
                    comm.send({'cmd': 'reloc', 'iids': [p1_iid], 'dst_node': dst_node, 'tags': [t1]}, dest=p1_node)
                    comm.send({'cmd': 'reloc', 'iids': [p2_iid], 'dst_node': dst_node, 'tags': [t2]}, dest=p2_node)
                else:
                    t1 = random.randint(1, MAX_TAG)
                    comm.ssend({'cmd': 'get_reloc_breed', 'src_nodes': [src_node], 'iids': [dst_iid], 'tags': [t1]},
                               dest=dst_node)
                    comm.send({'cmd': 'reloc', 'iids': [src_iid], 'dst_node': dst_node, 'tags': [t1]}, dest=src_node)

                    # relocated.setdefault((src_node, src_iid),[]).append(dst_node)
            self.slots[dst_node] -= 1

        for k in range(1, comm.Get_size()):
            comm.send({'cmd': 'breed_done', 'dst_node': self.population[0][0], 'alpha_iid': self.population[0][1]}, dest=k)


class EvoSlave:
    def __init__(self, slots, generator_func, fitness_func, breed_func, mutate_func, process_alpha_func):
        self.slots = slots

        self.population = []
        self.fit_weights = []

        self.relocs = []
        self.breeds = []
        self.to_decimate = []

        self.alpha = -1

        self.epoch = 0
        self.decimation_ratio = 0.4
        self.mutation_ratio = 0.15

        self.generator_func = generator_func
        self.fitness_func = fitness_func
        self.breed_func = breed_func
        self.mutate_func = mutate_func
        self.process_alpha_func = process_alpha_func
        # self.survivors_idx = int((1.0 - self.decimation_ratio) * self.population_size)

    def generate_population(self):
        self.population = []
        for i in range(self.slots):
            individual = self.generator_func()
            self.population.append(individual)

    def evolve(self, iterations):
        comm = MPI.COMM_WORLD
        self.generate_population()
        comm.barrier()

        for i in range(1, iterations + 1):
            self.relocs = []
            self.breeds = []
            self.to_decimate = []

            self.evaluate()
            while self.breed(): pass
            if self.alpha != -1:
                self.process_alpha_func(i, self.population[self.alpha])
            self.decimate()
            comm.barrier()
            self.mutate()
            comm.barrier()
            # print("#%d breeds: %d, relocs: %d, decimate: %d" % (rank, len(self.breeds), len(self.relocs), len(self.to_decimate)))
            # sys.stdout.flush()

    def evaluate(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        evaluated = []
        for idx, individual in enumerate(self.population):
            eval_result = (rank, idx, self.fitness_func(individual))
            # print(eval_result)
            evaluated.append(eval_result)
        comm.gather(evaluated, root=0)

    def breed(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        data = comm.recv(source=0, tag=0)
        iids = data.get('iids', [])
        src_nodes = data.get('src_nodes', [])
        dst_node = data.get('dst_node', -1)
        tags = data.get('tags', [])

        # print("Node %d, got cmd: %s" % (rank, data))

        if data['cmd'] == 'decimate':
            # print("Decimating %s-%s" % (rank, iids))
            # sys.stdout.flush()
            self.to_decimate = list(iids)

        elif data['cmd'] == 'reloc':
            # print("#%d: Relocating %s-%s to %s, tag %s" % (rank, rank, iids[0], dst_node, tags))
            # sys.stdout.flush()
            p1 = self.population[iids[0]]
            comm.send(p1, dest=dst_node, tag=tags[0])
        elif data['cmd'] == 'get_reloc':
            # print("#%d: Receiving individual from %s, tag %s" % (rank, src_nodes, tags))
            sys.stdout.flush()

            p1 = comm.recv(source=src_nodes[0], tag=tags[0])
            self.relocs.append(p1)

        elif data['cmd'] == 'get_reloc_breed':
            if len(src_nodes) == 1:
                # print("#%d: Receiving individual from %s and breeding with local %s, tag %s" % (rank, src_nodes, iids, tags))
                # sys.stdout.flush()

                p1 = comm.recv(source=src_nodes[0], tag=tags[0])
                p2 = self.population[iids[0]]
                breed = self.breed_func(p1, p2)
                self.breeds.append(breed)
            else:
                # print("#%d: Receiving individuals from %s and breeding, tag %s" % (rank, src_nodes, tags))
                # sys.stdout.flush()

                p1 = comm.recv(source=src_nodes[0], tag=tags[0])
                p2 = comm.recv(source=src_nodes[1], tag=tags[1])
                breed = self.breed_func(p1, p2)
                self.breeds.append(breed)

        elif data['cmd'] == 'breed_keep':
            # print("#%d: Breeding %s-%s and %s-%s" % (rank, rank, iids[0], rank, iids[1]))
            # sys.stdout.flush()
            p1 = self.population[iids[0]]
            p2 = self.population[iids[1]]
            breed = self.breed_func(p1, p2)
            self.breeds.append(breed)

        elif data['cmd'] == 'breed_reloc':
            # print("#%d: Breeding %s-%s and %s-%s and relocating to %s, tag %s" % (rank, rank, iids[0], rank, iids[1], dst_node, tags))
            # sys.stdout.flush()
            p1 = self.population[iids[0]]
            p2 = self.population[iids[1]]
            breed = self.breed_func(p1, p2)
            comm.send(breed, dest=dst_node, tag=tags[0])

        elif data['cmd'] == 'breed_done':
            if dst_node == rank:
                alpha = data.get('alpha_iid', -1)
                self.alpha = alpha
            else:
                self.alpha = -1
            return False

        else:
            print("Unknown cmd: %s" % data)
            sys.stdout.flush()
            raise Exception()
        return True

    def decimate(self):
        for index in sorted(self.to_decimate, reverse=True):
            del self.population[index]

    def mutate(self):
        population = [self.population, self.breeds, self.relocs]
        self.population = [item for sublist in population for item in sublist]
        to_mutate = np.random.binomial(1,self.mutation_ratio, len(self.population))
        # print(to_mutate)
        for idx, val in enumerate(self.population):
            if to_mutate[idx] == 1: self.mutate_func(val)


class Evolver:
    def __init__(self, epochs, population_size, decimation_rate, mutation_rate, generator, fitness, breeder, mutator, process_alpha):
        size = MPI.COMM_WORLD.Get_size()
        self.per_node = int(np.rint(population_size / (size - 1)))
        self.population_size = self.per_node * (size - 1)

        self.generator = generator
        self.fitness = fitness
        self.breeder = breeder
        self.mutator = mutator
        self.process_alpha = process_alpha
        self.epochs = epochs
        self.decimation_rate = decimation_rate
        self.mutation_rate = mutation_rate

    def master(self):
        mevo = EvoMaster(self.population_size, self.per_node, self.decimation_rate, self.mutation_rate)
        mevo.evolve(self.epochs)

    def slave(self):
        sevo = EvoSlave(self.per_node, self.generator, self.fitness, self.breeder, self.mutator, self.process_alpha)
        sevo.evolve(self.epochs)
