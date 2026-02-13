import numpy as np

from ml_genn import Connection, Network, Population
from ml_genn.compilers import InferenceCompiler
from ml_genn.connectivity import FixedProbability
from ml_genn.initializers import Normal, Uniform
from ml_genn.neurons import LeakyIntegrate, SpikeInput, UserNeuron
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential

from utils import TopoGraphic, SpatialDelay
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes)

from ml_genn.utils.data import preprocess_spikes
NUM_NEURONS = 5000#1_012_500
EXCITARTORY_RATIO = 0.8
NUM_EXCITATORY = int(NUM_NEURONS * EXCITARTORY_RATIO)
NUM_INHIBITORY = int(NUM_NEURONS * (1 - EXCITARTORY_RATIO))
BATCH_SIZE = 2
NUM_EPOCHS = 100
DT = 1.0

K = 3000

SPATIAL_DELAY = True

PROBABILITY_CONNECTION = K/NUM_NEURONS#0.1

EXCITATORY_INHIBITORY_RATIO = 4.0

NUM_EXCITATORY = int(round((NUM_NEURONS * EXCITATORY_INHIBITORY_RATIO) / (1.0 + EXCITATORY_INHIBITORY_RATIO)))

NUM_INHIBITORY = NUM_NEURONS - NUM_EXCITATORY
SCALE = (4000.0 / NUM_NEURONS) * (0.02 / PROBABILITY_CONNECTION)
EXCITATORY_WEIGHT = 4.0E-3 * SCALE / 10
INHIBITORY_WEIGHT = -51.0E-3 * SCALE / 10

L = NUM_NEURONS/1012500*6000#6000.0
sigma = NUM_NEURONS/1012500*400.0
n_side_E = int(np.ceil(np.sqrt(NUM_EXCITATORY)))
n_side_I = int(np.ceil(np.sqrt(NUM_INHIBITORY)))

vel = 0.2

E_vel= L/n_side_E * vel
I_vel = L/n_side_I * vel


# Preprocess
T = []
ind = []
for t in range(100):
    if np.random.rand() >= np.exp(-20 * 1 / 1000.0):
        T.append(t)
        ind.append(0)
spikes = [(preprocess_spikes(np.array(T), np.array(ind), 1))]
labels = [0]
T = []
ind = []
for t in range(100):
    if np.random.rand() >= np.exp(-20 * 1 / 1000.0):
        T.append(t)
        ind.append(0)
spikes.append((preprocess_spikes(np.array(T), np.array(ind), 1)))
labels.append(0)

# Determine max spikes and latest spike time
max_spikes = calc_max_spikes(spikes)
latest_spike_time = calc_latest_spike_time(spikes)
print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")

# Get number of input and output neurons from dataset 
# and round up outputs to power-of-two
num_input = 1
num_output = 2
max_delay = int((2*(sigma**2))**0.5 * (vel / DT))
print(max_delay)

serialiser = Numpy("shd_checkpoints")
network = Network()
LeakyIntegrateFireWithOffset = UserNeuron(vars={"V": ("(Isyn + Ioffset - V)/TauM", "Vrst")},
                  threshold="V - Vthresh",
                  output_var_name="V",
                  param_vals={"Ioffset": 0.051, "TauM": 20.0, "Vreset": 0, "Vthresh": 1, "Vrst": 0},
                  var_vals={"V": 0})

with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * max_spikes),
                       num_input)
    E_hidden = Population(LeakyIntegrateFireWithOffset,
                        NUM_EXCITATORY)
    I_hidden = Population(LeakyIntegrateFireWithOffset,
                        NUM_INHIBITORY)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                        num_output)
    # Connections
    Connection(input, E_hidden, FixedProbability(100/NUM_EXCITATORY, Normal(mean=0.03, sd=0.01)),
               Exponential(5.0))
    Connection(input, I_hidden, FixedProbability(100/NUM_INHIBITORY, Normal(mean=0.03, sd=0.01)),
               Exponential(5.0))
    Connection(E_hidden, E_hidden, TopoGraphic(Uniform(min=EXCITATORY_WEIGHT, max=EXCITATORY_WEIGHT), num=K*EXCITARTORY_RATIO, sigma_space=sigma/L*n_side_E, grid_num_x=int(n_side_E), delay=SpatialDelay(E_vel, grid_num_x=int(n_side_E), grid_num_x2=int(n_side_E))),
               Exponential(5.0), max_delay_steps=max_delay)
    Connection(E_hidden, I_hidden, TopoGraphic(Uniform(min=EXCITATORY_WEIGHT, max=EXCITATORY_WEIGHT), num=K*(1.0-EXCITARTORY_RATIO), sigma_space=sigma/L*n_side_I, grid_num_x=int(n_side_E), grid_num_x2=int(n_side_I), delay=SpatialDelay(I_vel, grid_num_x=int(n_side_E), grid_num_x2=int(n_side_I))),
               Exponential(5.0), max_delay_steps=max_delay)
    Connection(I_hidden, I_hidden, TopoGraphic(Uniform(min=INHIBITORY_WEIGHT, max=INHIBITORY_WEIGHT), num=K*(1.0-EXCITARTORY_RATIO), sigma_space=sigma/L*n_side_I, grid_num_x=int(n_side_I), delay=SpatialDelay(I_vel, grid_num_x=int(n_side_I), grid_num_x2=int(n_side_I))),
               Exponential(5.0), max_delay_steps=max_delay)
    Connection(I_hidden, E_hidden, TopoGraphic(Uniform(min=INHIBITORY_WEIGHT, max=INHIBITORY_WEIGHT), num=K*EXCITARTORY_RATIO, sigma_space=sigma/L*n_side_E, grid_num_x=int(n_side_I), grid_num_x2=int(n_side_E), delay=SpatialDelay(E_vel, grid_num_x=int(n_side_I), grid_num_x2=int(n_side_E))),
               Exponential(5.0), max_delay_steps=max_delay)
    '''Connection(I_hidden, output, Dense(Normal(mean=0.03, sd=0.01)),
               Exponential(5.0))'''

max_example_timesteps = int(np.ceil(latest_spike_time / DT))


compiler = InferenceCompiler(evaluate_timesteps=max_example_timesteps,
                                reset_in_syn_between_batches=True,
                                batch_size=BATCH_SIZE)
compiled_net = compiler.compile(network)

with compiled_net:
    # Evaluate model on numpy dataset
    metrics, _  = compiled_net.evaluate({input: spikes},
                                        {output: labels})
