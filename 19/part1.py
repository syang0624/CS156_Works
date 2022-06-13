import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# This code is from Finn M.
# I am including this in my folder to further investigation

class BinaryHopfield:
    def __init__(self, patterns=[]):
        self.patterns = patterns

    def store_memory(self, data):
        print("Storing Memory 🧠")
        self.patterns.append(data)
        self.neurons = data[0].shape[0]

        # weight initialisation
        weights = np.zeros((self.neurons, self.neurons))
        rho = np.sum([np.sum(d) for d in data]) / (len(data)*self.neurons)

        # hebbian learning using train data
        for i in range(len(data)):
            transformed_data = data[i] - rho
            weights += np.outer(transformed_data, transformed_data)

        # removing diagonal weights (nodes connecting to themselves)
        diag = np.diag(np.diag(weights))
        weights = weights - diag

        # Normalisation thingy
        weights /= len(data)

        self.weights = weights
        print("Memory Stored ⚡🧠⚡ (Probably, idk, pretty buggy)")

    def recall(self, data, iterations=20, threshold=0, sync=True):
        print("PATTERNS:")
        for i in self.patterns:
            visualise_square(i)
        print('\n')

        print("Trying to recall corrupted memory")
        print('\n')
        print("CORRUPTED MEMORY:")
        visualise_square(data)
        self.iterations = iterations
        self.threshold = threshold

        data_copy = np.copy(data)

        # run updates for each row
        predicted = []
        for i in range(len(data)):
            predicted.append(self._update(data_copy[i], sync))

        stacked_predicted = self._reshape(predicted)

        print("RECALLED MEMORY:")
        visualise_square(stacked_predicted)
        return stacked_predicted

    def _reshape(self, data):
        data = tuple(data)
        stacked_data = np.vstack(data)
        return stacked_data

    def _update(self, data, sync):
        if sync:
            s = self._synch_update(data)
        else:
            s = self._async_update(data)
        return s


    def _synch_update(self, data):
        s = data
        energy = self.energy(data)

        for i in range(self.iterations):
            # np.sign() represents the activation function
            # threshold is the bias here
            # Note: the synchronicity is embedded in this step
            s = np.sign(self.weights @ s - self.threshold)
            new_energy =self.energy(s)

            # terminate if convergence happens before maxing iterations
            if energy == new_energy:
                return s
            energy = new_energy
        return s

    def _async_update(self, data):
        s = data
        energy = self.energy(data)

        for i in range(self.iterations):
            for j in range(100): # this might have to be changed for large networks? idk..
                # select random neuron and update it
                neuron_index = np.random.randint(0, self.neurons)
                s[neuron_index] = np.sign(self.weights[neuron_index].T @ s - self.threshold)
            new_energy =self.energy(s)

            # terminate if convergence happens before maxing iterations
            if energy == new_energy:
                return s
            energy = new_energy
        return s


    def energy(self, s):
        # just copied the formula, this is basically gibberish to me
        return -0.5 * s @ self.weights @ s + np.sum(s * self.threshold)

def generate_random_square(size):
    square = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            square[i,j] = np.random.choice([-1, 1])
    return square

def corrupt_square(square):
    error_square = np.copy(square) # you need deep copy here or they both change
    i_val = np.random.choice([i for i in range(square.shape[0])])
    j_val = np.random.choice([i for i in range(square.shape[1])])
    square_val = square[i_val, j_val]
    print("SQUARE_VAL:", square_val)

    # This is really ugly but 2d numpy arrays aren't working properly
    if square_val > 0:
        error_square[i_val, j_val] = -1
    else:
        error_square[i_val, j_val] = 1
    return (error_square, i_val, j_val)

def visualise_square(square):
    img = Image.fromarray(square*255)
    plt.imshow(img)
    plt.show()

# generating some squares to act as memories
pattern_square = generate_random_square(10)
error_square, i_val, j_val = corrupt_square(pattern_square)

hoppedy_hop = BinaryHopfield()
hoppedy_hop.store_memory(pattern_square)
hoppedy_hop.recall(error_square, iterations=100, sync=False)
print(f"The altered pixel occurs at [{i_val}, {j_val}]")

# Why doesn't it recognise patterns i want to cry
