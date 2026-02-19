import numpy as np

from ml_genn import Connection, Network, Population
from ml_genn.compilers import InferenceCompiler
from ml_genn.connectivity import FixedProbability
from ml_genn.initializers import Normal, Uniform
from ml_genn.neurons import LeakyIntegrate, SpikeInput, UserNeuron
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
import mnist
from ml_genn.utils.data import linear_latency_encode_data

from pygenn import init_var
from ml_genn.callbacks import SpikeRecorder
from utils import TopoGraphic, SpatialDelay
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes)

from ml_genn.utils.data import preprocess_spikes
import matplotlib.pyplot as plt
NUM_NEURONS = 450000#1_012_500

BATCH_SIZE = 1
NUM_EPOCHS = 100
DT = 1.0



K = 3000

SPATIAL_DELAY = True

PROBABILITY_CONNECTION = K/NUM_NEURONS

EXCITATORY_INHIBITORY_RATIO = 4.0

NUM_EXCITATORY = int(round((NUM_NEURONS * EXCITATORY_INHIBITORY_RATIO) / (1.0 + EXCITATORY_INHIBITORY_RATIO)))

NUM_INHIBITORY = NUM_NEURONS - NUM_EXCITATORY
SCALE = (4000.0 / NUM_NEURONS) * (0.02 / PROBABILITY_CONNECTION)
EXCITATORY_WEIGHT = 4.0E-3 * SCALE / 10
INHIBITORY_WEIGHT = -51.0E-3 * SCALE / 10
L = NUM_NEURONS/1012500*6000#6000.0
sigma = NUM_NEURONS/1012500*400.0 #400.0
n_side_E = int(np.ceil(np.sqrt(NUM_EXCITATORY)))
n_side_I = int(np.ceil(np.sqrt(NUM_INHIBITORY)))

vel = 0.2

E_vel= L/n_side_E * vel
I_vel = L/n_side_I * vel

EXAMPLE_TIME = 20.0
mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
labels = mnist.test_labels()
spikes = linear_latency_encode_data(
    mnist.test_images(),
    EXAMPLE_TIME - (2.0 * DT), 2.0 * DT)

# Determine max spikes and latest spike time
max_spikes = calc_max_spikes(spikes)
latest_spike_time = calc_latest_spike_time(spikes)
print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")

# Get number of input and output neurons from dataset 
# and round up outputs to power-of-two
num_input = 784
num_output = 2
max_delay = int((2*(sigma**2))**0.5 * (vel / DT))
print(max_delay)

serialiser = Numpy("shd_checkpoints")
network = Network()

LeakyIntegrateFireWithOffset = UserNeuron(vars={"V": ("(Isyn-V+(Ioffset))/TauM", "Vreset")},
                  threshold="V - Vthresh",
                  output_var_name="V",
                  param_vals={"Ioffset": 0.051, "TauM": 20.0, "Vthresh": 1, "Vreset": 0},
                  var_vals={"V": 0})





with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * num_input),
                       num_input)
    E_hidden = Population(LeakyIntegrateFireWithOffset,
                        NUM_EXCITATORY, record_spikes=True)
    I_hidden = Population(LeakyIntegrateFireWithOffset,
                        NUM_INHIBITORY)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                        num_output)
    # Connections
    inputE = Connection(input, E_hidden, FixedProbability(num_input/NUM_EXCITATORY, Normal(mean=10.0, sd=1e-10)),
               Exponential(5.0))
    EE = Connection(E_hidden, E_hidden, TopoGraphic(Normal(mean=EXCITATORY_WEIGHT, sd=1e-10), num=int(K*EXCITATORY_INHIBITORY_RATIO/(1+EXCITATORY_INHIBITORY_RATIO)), sigma_space=sigma/L*n_side_E, grid_num_x=int(n_side_E), delay=SpatialDelay(E_vel, grid_num_x=int(n_side_E), grid_num_x2=int(n_side_E))),
               Exponential(5.0), max_delay_steps=max_delay)
    EI = Connection(E_hidden, I_hidden, TopoGraphic(Normal(mean=EXCITATORY_WEIGHT, sd=1e-10), num=int(K*(1-(EXCITATORY_INHIBITORY_RATIO/(1+EXCITATORY_INHIBITORY_RATIO)))), sigma_space=sigma/L*n_side_I, grid_num_x=int(n_side_E), grid_num_x2=int(n_side_I), delay=SpatialDelay(I_vel, grid_num_x=int(n_side_E), grid_num_x2=int(n_side_I))),
               Exponential(5.0), max_delay_steps=max_delay)
    II = Connection(I_hidden, I_hidden, TopoGraphic(Normal(mean=INHIBITORY_WEIGHT, sd=1e-10), num=int(K*(1-(EXCITATORY_INHIBITORY_RATIO/(1+EXCITATORY_INHIBITORY_RATIO)))), sigma_space=sigma/L*n_side_I, grid_num_x=int(n_side_I), delay=SpatialDelay(I_vel, grid_num_x=int(n_side_I), grid_num_x2=int(n_side_I))),
               Exponential(10.0), max_delay_steps=max_delay)
    IE = Connection(I_hidden, E_hidden, TopoGraphic(Normal(mean=INHIBITORY_WEIGHT, sd=1e-10), num=int(K*EXCITATORY_INHIBITORY_RATIO/(1+EXCITATORY_INHIBITORY_RATIO)), sigma_space=sigma/L*n_side_E, grid_num_x=int(n_side_I), grid_num_x2=int(n_side_E), delay=SpatialDelay(E_vel, grid_num_x=int(n_side_I), grid_num_x2=int(n_side_E))),
               Exponential(10.0), max_delay_steps=max_delay)
    '''Connection(I_hidden, output, Dense(Normal(mean=0.03, sd=0.01)),
               Exponential(5.0))'''



max_example_timesteps = int(np.ceil(latest_spike_time / DT))

callbacks = ["batch_progress_bar", SpikeRecorder(E_hidden, key="E_hidden")]
compiler = InferenceCompiler(evaluate_timesteps=100,
                                reset_in_syn_between_batches=True,
                                batch_size=BATCH_SIZE)
compiled_net = compiler.compile(network)

# Create spatial connectivity: each input neuron connects to multiple hidden neurons
# Input layer organized as 28x28 grid, mapped to hidden layer
input_side = 28
connections_per_input = 1  # Number of connections per input neuron

pre_ind = []
post_ind = []
start_x_loc = np.random.randint(0, n_side_E-28)
start_y_loc = np.random.randint(0, n_side_E-28)
for input_id in range(num_input):
    input_row = input_id // input_side
    input_col = input_id % input_side
    
    for i in range(connections_per_input):
        
        hidden_row = start_x_loc + input_row #+ np.random.randint(-10, 10)
        hidden_col = start_y_loc + input_col #+ np.random.randint(-10, 10)
        
        hidden_id = hidden_row * n_side_E + hidden_col
        
        pre_ind.append(input_id)
        post_ind.append(hidden_id)

inputE.connectivity.pre_ind = np.asarray(pre_ind)
inputE.connectivity.post_ind = np.asarray(post_ind)


with compiled_net:
    # Evaluate model on numpy dataset
    metrics, cb  = compiled_net.evaluate({input: spikes[:1]},
                                        {output: labels[:1]}, callbacks=callbacks)
    exc_spike_ids = cb["E_hidden"][1][0]
    exc_spike_times = cb["E_hidden"][0][0]
