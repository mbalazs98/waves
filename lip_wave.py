import numpy as np

from ml_genn import Connection, Network, Population
from ml_genn.compilers import InferenceCompiler,EventPropCompiler
from ml_genn.connectivity import FixedProbability, AvgPool2D, Conv2DTranspose, Dense
from ml_genn.initializers import Normal, Uniform
from ml_genn.neurons import LeakyIntegrate, SpikeInput, UserNeuron, LeakyIntegrateFire
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from ml_genn.callbacks import Checkpoint

from pygenn import init_var
from ml_genn.callbacks import SpikeRecorder, VarRecorder
from utils import TopoGraphic, SpatialDelay
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes)

from ml_genn.utils.data import preprocess_tonic_spikes
import matplotlib.pyplot as plt

from ml_genn.optimisers import Adam
from tonic.datasets import DVSLip
NUM_NEURONS =256**2 + 128**2#450000//2#1_012_500

BATCH_SIZE = 32
NUM_EPOCHS = 1
DT = 1.0/10



K = 3000

SPATIAL = True

PROBABILITY_CONNECTION = K/NUM_NEURONS#0.1

EXCITATORY_INHIBITORY_RATIO = 4.0

NUM_EXCITATORY = int(round((NUM_NEURONS * EXCITATORY_INHIBITORY_RATIO) / (1.0 + EXCITATORY_INHIBITORY_RATIO)))

NUM_INHIBITORY = NUM_NEURONS - NUM_EXCITATORY
SCALE = (4000.0 / NUM_NEURONS) * (0.02 / PROBABILITY_CONNECTION)
EXCITATORY_WEIGHT = 4.0E-3 * SCALE / 2
INHIBITORY_WEIGHT = -51.0E-3 * SCALE / 2
L = 6000*(128/900)#NUM_NEURONS/1012500*6000#6000.0 # µm
sigma = 400#NUM_NEURONS/1012500*400.0 #400.0 # µm
n_side_E = int(np.ceil(np.sqrt(NUM_EXCITATORY)))
n_side_I = int(np.ceil(np.sqrt(NUM_INHIBITORY)))

vel = 200 # µm/ms 

E_vel= L/n_side_E / vel
I_vel = L/n_side_I / vel

'''EXAMPLE_TIME = 20.0
mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
labels = mnist.test_labels()
spikes = linear_latency_encode_data(
    mnist.test_images(),
    EXAMPLE_TIME - (2.0 * DT), 2.0 * DT + 10)'''
offset = 100000 # microsecond
dataset = DVSLip(save_to="data/", train=True)
ordering = dataset.ordering
sensor_size = dataset.sensor_size


max_spikes = 0
latest_spike_time = 0
spikes, labels = [], []
for i, data in enumerate(dataset):
    print(i)
    events, label = data
    events["t"] += offset
    spikes.append(preprocess_tonic_spikes(events, dataset.ordering,
                                                        dataset.sensor_size, histogram_thresh=None, dt=DT, polarity="merge"))
    labels.append(label)
                   
# Determine max spikes and latest spike time
max_spikes = calc_max_spikes(spikes)
latest_spike_time = calc_latest_spike_time(spikes)
print("merge")
print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")



# Get number of input and output neurons from dataset 
# and round up outputs to power-of-two
max_delay = int((2*(L**2))**0.5 / (vel / DT))
print(max_delay)

serialiser = Numpy("shd_checkpoints")
network = Network()

LeakyIntegrateFireWithOffset_E = UserNeuron(vars={"V": ("(Isyn-V+(Ioffset))/TauM", "Vreset")},
                  threshold="V - Vthresh",
                  output_var_name="V",
                  param_vals={"Ioffset": 0.051*20, "TauM": 20.0, "Vthresh": 1, "Vreset": 0},
                  var_vals={"V": np.random.rand(NUM_EXCITATORY)})


LeakyIntegrateFireWithOffset_I = UserNeuron(vars={"V": ("(Isyn-V+(Ioffset))/TauM", "Vreset")},
                  threshold="V - Vthresh",
                  output_var_name="V",
                  param_vals={"Ioffset": 0.051*20, "TauM": 20.0, "Vthresh": 1, "Vreset": 0},
                  var_vals={"V": np.random.rand(NUM_INHIBITORY)})



# Input layer organized as 28x28 grid, mapped to hidden layer with spatial pooling
input_side = 128

with network:
    # Populations
    input = Population(SpikeInput(max_spikes=max_spikes),
                       input_side**2)
    E_hidden = Population(LeakyIntegrateFireWithOffset_E,
                        NUM_EXCITATORY, record_spikes=True)
    I_hidden = Population(LeakyIntegrateFireWithOffset_I,
                        NUM_INHIBITORY)
    pool = Population(LeakyIntegrateFire(tau_mem=20.0),
                        input_side**2)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                        input_side**2)
    # Connections 
    # Input to excitatory hidden layer with spatial pooling
    inputE = Connection(input, E_hidden, TopoGraphic(Normal(mean=0.0, sd=np.sqrt(2/NUM_EXCITATORY)), num=2, sigma_space=1.0, grid_num_x=int(input_side), grid_num_x2=int(n_side_E)),
               Exponential(5.0))
    # Input to inhibitory hidden layer with spatial pooling
    inputI = Connection(input, I_hidden, TopoGraphic(Normal(mean=0.0, sd=np.sqrt(2/NUM_INHIBITORY)), num=1, sigma_space=1.0, grid_num_x=int(input_side), grid_num_x2=int(n_side_I)),
               Exponential(5.0))
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
    Epool = Connection(E_hidden,pool , TopoGraphic(Normal(mean=1.0, sd=1e-10), num=1, sigma_space=1.0, grid_num_x=int(n_side_E), grid_num_x2=int(input_side)),
               Exponential(5.0))
    Ipool = Connection(I_hidden, pool, TopoGraphic(Normal(mean=1.0, sd=1e-10), num=1, sigma_space=1.0, grid_num_x=int(n_side_I), grid_num_x2=int(input_side)),
               Exponential(5.0))
    pool_output = Connection(pool,output , Dense(Normal(mean=0.0, sd=0.03)),
               Exponential(5.0))
    
    



max_example_timesteps = int(np.ceil((latest_spike_time + 100) / DT))

callbacks = ["batch_progress_bar", SpikeRecorder(E_hidden, key="E_hidden"),  Checkpoint(serialiser)]
#callbacks = ["batch_progress_bar", VarRecorder(output, "v", key="E_hidden"),  Checkpoint(serialiser)]
'''compiler = InferenceCompiler(evaluate_timesteps=100,
                                reset_in_syn_between_batches=True,
                                batch_size=BATCH_SIZE)'''
compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                 losses="sparse_categorical_crossentropy",
                                batch_size=BATCH_SIZE, dt=DT,
                                 kernel_profiling=False)
compiled_net = compiler.compile(network)



with compiled_net:
    compiled_net.save_connectivity((NUM_EPOCHS - 1,), serialiser)
    # Evaluate model on numpy dataset
    metrics, cb  = compiled_net.train({input: spikes[:]},
                                        {output: labels[:]}, callbacks=callbacks, num_epochs=1)

with compiled_net:
    compiled_net.save_connectivity((NUM_EPOCHS - 1,), serialiser)
    #print(cb["E_hidden"][0].shape)
    exc_spike_ids = cb["E_hidden"][1][0]
    exc_spike_times = cb["E_hidden"][0][0]
    #frames = np.array(cb["E_hidden"]).reshape(-1, input_side, input_side)


side = 280#int(np.sqrt(NUM_EXCITATORY))  # should be 900
print(side)  # 900

crop = 0#side // 6          # 900 // 6 = 150
r0, r1 = crop, side - crop
c0, c1 = crop, side - crop

cropped_side = r1 - r0    # should be 600


'''exc_spike_times = np.array(exc_spike_times)  # shape (n_spikes,)
exc_spike_ids   = np.array(exc_spike_ids)    # shape (n_spikes,)'''

t_min = 0#exc_spike_times.min()
#t_max = pool_volt.shape[0]#exc_spike_times.max()
t_max = exc_spike_times.max()
dt = 1.0 # ms per frame (adjust to taste)
time_bins = np.arange(t_min, t_max + dt, dt)
print(t_min, t_max)

n_frames = len(time_bins) - 1

frames = np.zeros((n_frames, cropped_side, cropped_side), dtype=np.float32)


# Find which frame each spike belongs to
frame_indices = np.digitize(exc_spike_times, time_bins) - 1

valid = (frame_indices >= 0) & (frame_indices < n_frames)
frame_indices = frame_indices[valid]
ids = exc_spike_ids[valid]

rows = ids // side
cols = ids % side

# Keep only spikes inside central region
inside = (rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1)

rows = rows[inside] - r0   # shift so cropped grid starts at 0
cols = cols[inside] - c0
frame_indices = frame_indices[inside]

# Accumulate spikes into frames
for f, r, c in zip(frame_indices, rows, cols):
    frames[f, r, c] += 1

'''# Log scale helps a lot with spike data
frames = np.log1p(frames)

# Normalize to [0, 1]
frames /= frames.max()'''


from scipy.ndimage import gaussian_filter1d

# Smooth along time axis
frames = gaussian_filter1d(frames, sigma=5, axis=0)

from scipy.ndimage import gaussian_filter

import skimage.measure
frames_pool = np.zeros((n_frames, int(cropped_side), int(cropped_side)), dtype=np.float32)


for i in range(n_frames):
    frames_pool[i] = gaussian_filter(frames[i], sigma=10)
    #frames_pool[i] = skimage.measure.block_reduce(frames[i], (5,5), np.mean)

max_val = np.max(frames_pool)
min_val = np.min(frames_pool)

import matplotlib.pyplot as plt
import imageio

gif_path = "spiking_activity.gif"
images = []

for i in range(frames.shape[0]):
    print(i)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(frames_pool[i], cmap="gist_gray", vmin=0, vmax=max_val)
    ax.set_title(f"Time: {i:.1f}–{i+1:.1f} ms")
    ax.axis("off")

    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    images.append(image)

    plt.close(fig)

imageio.mimsave(gif_path, images, duration=0.1, loop=10 )  # seconds per frame
