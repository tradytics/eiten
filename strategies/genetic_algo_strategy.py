# Basic libraries
import warnings
import numpy as np
from utils import dotdict, normalize_weights
warnings.filterwarnings("ignore")


class GeneticAlgoStrategy:
    """
    My own custom implementation of genetic algorithms for portfolio
    """

    def __init__(self):
        self.name = "Genetic Algo"
        self.initial_genes = 100
        self.selection_top = 10
        self.mutation_iterations = 50
        self.weight_update_factor = 0.01
        self.gene_length = None
        self.genes_in_each_iteration = 250
        self.iterations = 100
        self.crossover_probability = 0.1

    def generate_portfolio(self, **kwargs):
        kwargs = dotdict(kwargs)
        symbols = list(kwargs.cov_matrix.columns)
        self.gene_length = len(symbols)

        # Create initial genes
        initial_genes = self.generate_initial_genes(symbols)

        for i in range(self.iterations):
            # Select
            top_genes = self.select(kwargs.sample_returns, initial_genes)
            # print("Iteration %d Best Sharpe Ratio: %.3f" % (i, top_genes[0][0]))
            top_genes = [item[1] for item in top_genes]

            # Mutate
            mutated_genes = self.mutate(top_genes)
            initial_genes = mutated_genes

        top_genes = self.select(kwargs.sample_returns, initial_genes)
        best_gene = top_genes[0][1]
        # Gene is a distribution of weights for different stocks
        # transposed_gene = np.array(best_gene).transpose()
        # returns = np.dot(return_matrix, transposed_gene)
        # returns_cumsum = np.cumsum(returns)
        n_best = normalize_weights(best_gene)
        weights = {symbols[x]: n_best[x] for x in range(0, len(best_gene))}
        return weights

    def generate_initial_genes(self, symbols):
        return np.array(
            [self.generate_gene() for _ in range(self.gene_length)])

    def mutate(self, genes):
        new_genes = []

        for gene in genes:
            for x in range(0, self.mutation_iterations):
                mutation = gene + (self.weight_update_factor *
                                   np.random.uniform(-1, 1, self.gene_length))
                new_genes.append(mutation)

        new_genes = genes + new_genes
        np.random.shuffle(new_genes)
        genes_to_keep = new_genes[:self.genes_in_each_iteration]

        # Add crossovers
        crossovers = self.crossover(new_genes)
        genes_to_keep = genes_to_keep + crossovers

        return genes_to_keep

    def select(self, return_matrix, genes):
        genes_with_scores = []
        for gene in genes:
            # Gene is a distribution of weights for different stocks
            transposed_gene = gene.transpose()
            returns = np.dot(return_matrix, transposed_gene)
            # returns_cumsum = np.cumsum(returns)

            # Get fitness score
            fitness = self.fitness_score(returns)
            genes_with_scores.append([fitness, gene])

        # Sort
        random_genes = [self.generate_gene() for _ in range(5)]
        genes_with_scores = sorted(
            genes_with_scores, reverse=True, key=lambda x: x[0])
        genes_with_scores = (genes_with_scores[:self.selection_top] +
                             random_genes)
        return genes_with_scores

    def fitness_score(self, returns):
        sharpe_returns = np.mean(returns) / np.std(returns)
        return sharpe_returns

    def generate_gene(self):
        return np.random.uniform(-1, 1, self.gene_length)

    def crossover(self, population):
        rng = np.random.default_rng()
        crossover_population = []

        population = np.array(
            list(filter(lambda x: type(x) == np.ndarray, population)))
        for z in range(0, len(population)):
            if np.random.uniform(0, 1) < self.crossover_probability:
                a, b = rng.choice(population, 2)
                random_split = np.random.randint(1, len(a) - 1)
                ab = np.concatenate(
                    (a[:random_split], b[random_split:]), axis=0)
                crossover_population.append(ab)

        return crossover_population
