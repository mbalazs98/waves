import numpy as np

from ml_genn import Connection, Network, Population
from ml_genn.compilers import InferenceCompiler,EventPropCompiler
from ml_genn.connectivity import FixedProbability, AvgPoolDense2D
from ml_genn.initializers import Normal, Uniform
from ml_genn.neurons import LeakyIntegrate, SpikeInput, UserNeuron, LeakyIntegrateFire
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from ml_genn.callbacks import Checkpoint

from pygenn import init_var
from ml_genn.callbacks import SpikeRecorder, VarRecorder
from utils import TopoGraphic, SpatialDelay
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes)

from ml_genn.utils.data import preprocess_tonic_spikes, preprocess_tonic_spikes_separate
import matplotlib.pyplot as plt

from ml_genn.optimisers import Adam
from tonic.datasets import DVSLip
from tonic.transforms import MergePolarities

from argparse import ArgumentParser

from ml_genn.communicators import MPI
import json

parser = ArgumentParser()
parser.add_argument("--num_hidden", type=int, default=128, help="Conduction velocity")
parser.add_argument("--k", type=int, default=3000, help="Conduction velocity")
parser.add_argument("--velocity", type=float, default=200, help="Conduction velocity")
parser.add_argument("--sigma", type=int, default=400, help="Connection probability")
parser.add_argument("--offset_multiplier", type=float, default=20.0, help="Is spatial")
parser.add_argument("--is_spatial", type=int, default=1, help="Is spatial")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

unique_suffix = "_".join(("_".join(str(i) for i in val) if isinstance(val, list) 
                         else str(val))
                         for arg, val in vars(args).items())
np.random.seed(args.seed)

NUM_NEURONS =args.num_hidden**2 + (args.num_hidden//2)**2

BATCH_SIZE = 64
NUM_EPOCHS = 300
DT = 1.0/10



K = args.k

SPATIAL = bool(args.is_spatial)

PROBABILITY_CONNECTION = K/NUM_NEURONS

EXCITATORY_INHIBITORY_RATIO = 4.0

NUM_EXCITATORY = int(round((NUM_NEURONS * EXCITATORY_INHIBITORY_RATIO) / (1.0 + EXCITATORY_INHIBITORY_RATIO)))

NUM_INHIBITORY = NUM_NEURONS - NUM_EXCITATORY
SCALE = (4000.0 / NUM_NEURONS) * (0.02 / PROBABILITY_CONNECTION)
EXCITATORY_WEIGHT = 4.0E-3 * SCALE / 2
INHIBITORY_WEIGHT = -51.0E-3 * SCALE / 2
L = 6000*(np.sqrt(NUM_EXCITATORY)/900) # µm
sigma = args.sigma # µm
n_side_E = int(np.ceil(np.sqrt(NUM_EXCITATORY)))
n_side_I = int(np.ceil(np.sqrt(NUM_INHIBITORY)))
vel = args.velocity # µm/ms 

E_vel= L/n_side_E / vel
I_vel = L/n_side_I / vel

offset = 50000 # microsecond
dataset = DVSLip(save_to="data/", train=True)
ordering = dataset.ordering
sensor_size = dataset.sensor_size

merge = MergePolarities()
max_spikes = 0
latest_spike_time = 0
spikes, labels = [], []

communicator = MPI()
print(f"Training on rank {communicator.rank} / {communicator.num_ranks}")

data_range = range(communicator.rank, len(dataset), communicator.num_ranks)

for i in data_range:
    events, label = dataset[i]
    events["t"] += offset
    spikes.append(preprocess_tonic_spikes(merge(events), dataset.ordering,
                                                        (dataset.sensor_size[0], dataset.sensor_size[1], 1), histogram_thresh=None, dt=DT))
    labels.append(label)

for i in data_range:
    events, label = dataset[i]
    events["t"] += offset
    spikes.append(preprocess_tonic_spikes(merge(events), dataset.ordering,
                                                        (dataset.sensor_size[0], dataset.sensor_size[1], 1), histogram_thresh=None, dt=DT))
    labels.append(label)
                   
# Determine max spikes and latest spike time
max_spikes = calc_max_spikes(spikes)
latest_spike_time = calc_latest_spike_time(spikes)

print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")





max_delay = int((2*(L**2))**0.5 / (vel * DT))
print(f"Max delay {max_delay}")


network = Network()

LeakyIntegrateFireWithOffset_E = UserNeuron(vars={"V": ("(Isyn-V+(Ioffset))/TauM", "Vreset")},
                  threshold="V - Vthresh",
                  output_var_name="V",
                  param_vals={"Ioffset": 0.051*args.offset_multiplier, "TauM": 20.0, "Vthresh": 1, "Vreset": 0},
                  var_vals={"V": np.random.rand(NUM_EXCITATORY)})


LeakyIntegrateFireWithOffset_I = UserNeuron(vars={"V": ("(Isyn-V+(Ioffset))/TauM", "Vreset")},
                  threshold="V - Vthresh",
                  output_var_name="V",
                  param_vals={"Ioffset": 0.051*args.offset_multiplier, "TauM": 20.0, "Vthresh": 1, "Vreset": 0},
                  var_vals={"V": np.random.rand(NUM_INHIBITORY)})



input_side = 128

with network:
    # Populations
    input = Population(SpikeInput(max_spikes=max_spikes * BATCH_SIZE),
                       input_side**2)
    E_hidden = Population(LeakyIntegrateFireWithOffset_E,
                        (int(np.sqrt(NUM_EXCITATORY)), int(np.sqrt(NUM_EXCITATORY)), 1), record_spikes=True)
    I_hidden = Population(LeakyIntegrateFireWithOffset_I,
                        NUM_INHIBITORY)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="sum_var"),
                        100)
    # Connections 
    # Input to excitatory hidden layer with spatial pooling
    inputE = Connection(input, E_hidden, TopoGraphic(Normal(mean=1, sd=0.01), num=2, sigma_space=1.0, grid_num_x=int(input_side), grid_num_x2=int(n_side_E)),
               Exponential(5.0))
    # Input to inhibitory hidden layer with spatial pooling
    '''inputI = Connection(input, I_hidden, TopoGraphic(Normal(mean=np.sqrt(2/NUM_INHIBITORY), sd=np.sqrt(2/NUM_INHIBITORY)), num=1, sigma_space=1.0, grid_num_x=int(input_side), grid_num_x2=int(n_side_I)),
               Exponential(5.0))'''
    if SPATIAL:
        EE = Connection(E_hidden, E_hidden, TopoGraphic(Normal(mean=EXCITATORY_WEIGHT, sd=1e-10), num=int(K*EXCITATORY_INHIBITORY_RATIO/(1+EXCITATORY_INHIBITORY_RATIO)), sigma_space=sigma/L*n_side_E, grid_num_x=int(n_side_E), delay=SpatialDelay(E_vel, grid_num_x=int(n_side_E), grid_num_x2=int(n_side_E))),
                Exponential(5.0), max_delay_steps=max_delay)
        EI = Connection(E_hidden, I_hidden, TopoGraphic(Normal(mean=EXCITATORY_WEIGHT, sd=1e-10), num=int(K*(1-(EXCITATORY_INHIBITORY_RATIO/(1+EXCITATORY_INHIBITORY_RATIO)))), sigma_space=sigma/L*n_side_I, grid_num_x=int(n_side_E), grid_num_x2=int(n_side_I), delay=SpatialDelay(I_vel, grid_num_x=int(n_side_E), grid_num_x2=int(n_side_I))),
                Exponential(5.0), max_delay_steps=max_delay)
        II = Connection(I_hidden, I_hidden, TopoGraphic(Normal(mean=INHIBITORY_WEIGHT, sd=1e-10), num=int(K*(1-(EXCITATORY_INHIBITORY_RATIO/(1+EXCITATORY_INHIBITORY_RATIO)))), sigma_space=sigma/L*n_side_I, grid_num_x=int(n_side_I), delay=SpatialDelay(I_vel, grid_num_x=int(n_side_I), grid_num_x2=int(n_side_I))),
                Exponential(10.0), max_delay_steps=max_delay)
        IE = Connection(I_hidden, E_hidden, TopoGraphic(Normal(mean=INHIBITORY_WEIGHT, sd=1e-10), num=int(K*EXCITATORY_INHIBITORY_RATIO/(1+EXCITATORY_INHIBITORY_RATIO)), sigma_space=sigma/L*n_side_E, grid_num_x=int(n_side_I), grid_num_x2=int(n_side_E), delay=SpatialDelay(E_vel, grid_num_x=int(n_side_I), grid_num_x2=int(n_side_E))),
                Exponential(10.0), max_delay_steps=max_delay)
    else:
        EE = Connection(E_hidden, E_hidden, FixedProbability(weight=Normal(mean=EXCITATORY_WEIGHT, sd=1e-10), p=PROBABILITY_CONNECTION),
                Exponential(5.0))
        EI = Connection(E_hidden, I_hidden, FixedProbability(weight=Normal(mean=EXCITATORY_WEIGHT, sd=1e-10), p=PROBABILITY_CONNECTION),
                Exponential(5.0))
        II = Connection(I_hidden, I_hidden, FixedProbability(weight=Normal(mean=INHIBITORY_WEIGHT, sd=1e-10), p=PROBABILITY_CONNECTION),
                Exponential(5.0))
        IE = Connection(I_hidden, E_hidden, FixedProbability(weight=Normal(mean=INHIBITORY_WEIGHT, sd=1e-10), p=PROBABILITY_CONNECTION),
                Exponential(5.0))
    EO = Connection(E_hidden,output , AvgPoolDense2D(weight = np.random.normal(0, 0.02, (int(NUM_EXCITATORY/(8**2)), 100)), pool_size=8, pool_strides=8),
               Exponential(5.0))
    
    



max_example_timesteps = int(np.ceil((latest_spike_time + 100)))

callbacks = []
if communicator.rank == 0:
        serialiser = Numpy("lip_checkpoints_" + unique_suffix)
        #callbacks.append(Checkpoint(serialiser))

compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                max_spikes=1000,
                                losses="sparse_categorical_crossentropy",
                                batch_size=BATCH_SIZE, dt=DT,
                                kernel_profiling=False,
                                communicator=communicator,
                                strict_buffer_checking=True,
                                rng_seed=args.seed)
optimisers = {inputE: {"weight": Adam(0.001)},
             EO: {"weight": Adam(0.001)}}
compiled_net = compiler.compile(network, optimisers=optimisers)


acc = 0
early_stop = 10
with compiled_net:
    if communicator.rank == 0:
        compiled_net.save_connectivity(("best"), serialiser)
    for i in range(NUM_EPOCHS):
        metrics, _  = compiled_net.train({input: spikes},
                                        {output: labels}, callbacks=callbacks, num_epochs=1, start_epoch=start_epoch-1)

        if metrics[output].result > acc:
                acc = metrics[output].result
                results_dic["train_acc"] = str(metrics[output].result)
                results_dic["epoch"] = str(i)
                early_stop = 10
                with open(f"results/lip_{communicator.rank}_{unique_suffix}.json", 'w') as f:
                        json.dump(results_dic, f, indent=4)
                compiled_net.save(("best",), serialiser)
                
        else:
                early_stop -= 1
                if early_stop < 0:
                        break

