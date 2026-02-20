import numpy as np
import matplotlib.pyplot as plt
from time import time

from pygenn import (GeNNModel, VarLocation, init_var,
                    init_postsynaptic, init_weight_update, create_sparse_connect_init_snippet, create_var_init_snippet,  create_weight_update_model, create_neuron_model,
                    init_sparse_connectivity)


# Parameters
time_div = 10
TIMESTEP = 1.0 / time_div
NUM_NEURONS = 1_012_500


K = 3000

conduction_delay = 200 #μm/ms


EXCITATORY_INHIBITORY_RATIO = 4.0
NUM_EXCITATORY = int(round((NUM_NEURONS * EXCITATORY_INHIBITORY_RATIO) / (1.0 + EXCITATORY_INHIBITORY_RATIO)))

NUM_INHIBITORY = NUM_NEURONS - NUM_EXCITATORY

PROBABILITY_CONNECTION_E = K*(EXCITATORY_INHIBITORY_RATIO/(1+EXCITATORY_INHIBITORY_RATIO))/NUM_EXCITATORY#0.1
PROBABILITY_CONNECTION_I = K*(1-(EXCITATORY_INHIBITORY_RATIO/(1+EXCITATORY_INHIBITORY_RATIO)))/NUM_INHIBITORY#0.1

SCALE_E = (4000.0 / NUM_NEURONS) * (0.02 / (PROBABILITY_CONNECTION_E))
SCALE_I = (4000.0 / NUM_NEURONS) * (0.02 / (PROBABILITY_CONNECTION_I))
EXCITATORY_WEIGHT = 4.0E-3 * SCALE_E * 2
INHIBITORY_WEIGHT = -51.0E-3 * SCALE_I * 2
model = GeNNModel("float", "va_benchmark")
model.dt = TIMESTEP
model.default_narrow_sparse_ind_enabled = True


lif_init = {"V": init_var("Uniform", {"min": 0, "max": 1}), "RefracTime": 0.0}
lif_params = {"C": 20.0, "TauM": 20.0, "Vrest": 0.0, "Vreset": 0.0, "Vthresh" : 1.0,
                      "Ioffset": 0.051*20, "TauRefrac": 0.0}

poisson_model = create_neuron_model(
    "poisson",

    threshold_condition_code="gennrand_uniform() >= exp(-rate * dt / 1000.0)",
    vars= [("rate", "scalar")]
)
poisson_init = {"rate": 20.0}
excitatory_pop = model.add_neuron_population("E", NUM_EXCITATORY, "LIF", lif_params, lif_init)
inhibitory_pop = model.add_neuron_population("I", NUM_INHIBITORY, "LIF", lif_params, lif_init)

excitatory_pop.spike_recording_enabled = True
inhibitory_pop.spike_recording_enabled = True



StaticPulseDendriticDelayConstantWeight = create_weight_update_model(
    "StaticPulseDendriticDelayConstantWeight",
    params=["g"],
    vars=[("d", "uint8_t")],

    pre_spike_syn_code=
        """
        addToPostDelay(g, d);""")


input_number_post = create_sparse_connect_init_snippet(
    "connect_input",
    params=[("num", "unsigned int"), ("start_loc_x", "int"), ("start_loc_y", "int"), ("grid_num_x", "unsigned int")],
    col_build_code=
        """
        const int xPost = id_post % grid_num_x;
        const int yPost = id_post/ grid_num_x;
        const int sqrt_num_post = round(sqrt((scalar)num));
        if (xPost >= start_loc_x && xPost < start_loc_x + sqrt_num_post && yPost >= start_loc_y && yPost < start_loc_y + sqrt_num_post) {
            int count = 0;
            while(count < num) {
                addSynapse(count);
                count++;
            }
        }
        """,   
        calc_max_col_len_func=lambda num_pre, num_post, pars: pars["num"])


fixed_number_post= create_sparse_connect_init_snippet(
    "fixed_number_post",
    params=[("num", "unsigned int"), ("sigma_space", "float"), ("grid_num_x", "unsigned int"), ("grid_num_x2", "unsigned int")],
    row_build_code=
        """
        const float ratio = (float)(grid_num_x2 -1 ) / (float)(grid_num_x-1);
        const float xPre = (id_pre % grid_num_x) * ratio;
        const float yPre = (id_pre / grid_num_x) * ratio;
        int count = num;
        while(count > 0) {
            const float distanceX = gennrand_normal() * sigma_space;
            const float distanceY = gennrand_normal() * sigma_space;
            int xPost = (int)round(xPre + distanceX);
            int yPost = (int)round(yPre + distanceY);
            // periodic wrapping
            while(xPost < 0) xPost += grid_num_x2;
            while(xPost >= grid_num_x2) xPost -= grid_num_x2;
            while(xPost < 0) xPost += grid_num_x2;
            while(yPost < 0) yPost += grid_num_x2;
            while(yPost >= grid_num_x2) yPost -= grid_num_x2;
            count--;
            const int id_post = (yPost * grid_num_x2) + xPost;
            addSynapse(id_post);
        }
        """,   
        calc_max_row_len_func=lambda num_pre, num_post, pars: pars["num"])


calc_dist = create_var_init_snippet(
    "calc_dist",
    
    params=[("delay", "float"),("grid_num_x", "unsigned int"), ("grid_num_x2", "unsigned int")],
    var_init_code=
        """
        const float ratio = (float)(grid_num_x2 -1 ) / (float)(grid_num_x-1);
        const float xPre = (id_pre % grid_num_x) * ratio;
        const float yPre = (id_pre / grid_num_x) * ratio;
        const float xPost = id_post % grid_num_x2;
        const float yPost = id_post / grid_num_x2;
        float dx = fabs(xPre - xPost);
        float dy = fabs(yPre - yPost);

        // Periodic boundary conditions
        if (dx > 0.5 * grid_num_x2){
            dx = dx - grid_num_x2;
        };
        if (dy > 0.5 * grid_num_x2){
            dy = dy - grid_num_x2;
        };

        float dist = sqrt(dx * dx + dy * dy);
        value = dist * delay;
        """
    )
input_num = 40*40
input_pop1 = model.add_neuron_population("Input1", input_num, poisson_model, {}, poisson_init)
input_pop2 = model.add_neuron_population("Input2", input_num, poisson_model, {}, poisson_init)


L = NUM_NEURONS/1012500*6000#6000.0
sigma = NUM_NEURONS/1012500*400.0
n_side_E = int(np.ceil(np.sqrt(NUM_EXCITATORY)))
n_side_I = int(np.ceil(np.sqrt(NUM_INHIBITORY)))
E_dist = L/n_side_E
I_dist = L/n_side_I


EE_synapse_init = init_weight_update(StaticPulseDendriticDelayConstantWeight, {"g": EXCITATORY_WEIGHT}, { "d": init_var(calc_dist, {"delay": E_dist/conduction_delay, "grid_num_x": int(n_side_E), "grid_num_x2": int(n_side_E)})})
EI_synapse_init = init_weight_update(StaticPulseDendriticDelayConstantWeight, {"g": EXCITATORY_WEIGHT}, {"d": init_var(calc_dist, {"delay": I_dist/conduction_delay, "grid_num_x": int(n_side_E), "grid_num_x2": int(n_side_I)})})
II_synapse_init = init_weight_update(StaticPulseDendriticDelayConstantWeight, {"g": INHIBITORY_WEIGHT}, { "d": init_var(calc_dist, {"delay": I_dist/conduction_delay, "grid_num_x": int(n_side_I), "grid_num_x2": int(n_side_I)})})
IE_synapse_init = init_weight_update(StaticPulseDendriticDelayConstantWeight, {"g": INHIBITORY_WEIGHT}, {"d": init_var(calc_dist, {"delay": E_dist/conduction_delay, "grid_num_x": int(n_side_I), "grid_num_x2": int(n_side_E)})})

excitatory_postsynaptic_init = init_postsynaptic("ExpCurr", {"tau": 5.0})
inhibitory_postsynaptic_init = init_postsynaptic("ExpCurr", {"tau": 10.0})


'''start_loc_x = np.random.randint(0, n_side_E-int(np.sqrt(input_num)))
start_loc_y = np.random.randint(0, n_side_E-int(np.sqrt(input_num)))

InputE_syn_pop = model.add_synapse_population("InputE", "SPARSE",
    input_pop1, excitatory_pop,
    init_weight_update("StaticPulseConstantWeight", {"g": 0.0}),
    init_postsynaptic("ExpCurr", {"tau": 5.0}),
    init_sparse_connectivity(input_number_post, {"num": int(input_num), "start_loc_x": start_loc_x, "start_loc_y": start_loc_y, "grid_num_x": int(n_side_E)}))

InputI_syn_pop = model.add_synapse_population("InputI", "SPARSE",
    input_pop2, inhibitory_pop,
    init_weight_update("StaticPulseConstantWeight", {"g": 0.0}),
    init_postsynaptic("ExpCurr", {"tau": 5.0}),
    init_sparse_connectivity(input_number_post, {"num": int(input_num/2), "start_loc_x": start_loc_x/2, "start_loc_y": start_loc_y/2, "grid_num_x": int(n_side_I)}))'''

EE_syn_pop = model.add_synapse_population("EE", "SPARSE",
    excitatory_pop, excitatory_pop,
    #excitatory_weight_init, excitatory_postsynaptic_init,
    EE_synapse_init, excitatory_postsynaptic_init,
    #init_sparse_connectivity("FixedProbabilityNoAutapse", fixed_prob))
    init_sparse_connectivity(fixed_number_post, {"num": int(K*EXCITATORY_INHIBITORY_RATIO/(1+EXCITATORY_INHIBITORY_RATIO)), "sigma_space": sigma/L*n_side_E, "grid_num_x": int(n_side_E), "grid_num_x2": int(n_side_E)}))

EI_syn_pop = model.add_synapse_population("EI", "SPARSE",
    excitatory_pop, inhibitory_pop,
    #excitatory_weight_init, excitatory_postsynaptic_init,
    EI_synapse_init, excitatory_postsynaptic_init,
    #init_sparse_connectivity("FixedProbability", fixed_prob))
    #init_sparse_connectivity("FixedNumberPostWithReplacement", {"num": 3000*0.2}))
    init_sparse_connectivity(fixed_number_post, {"num": int(K*(1-(EXCITATORY_INHIBITORY_RATIO/(1+EXCITATORY_INHIBITORY_RATIO)))), "sigma_space": sigma/L*n_side_I, "grid_num_x": int(n_side_E), "grid_num_x2": int(n_side_I)}))
II_syn_pop = model.add_synapse_population("II", "SPARSE",
    inhibitory_pop, inhibitory_pop,
    #inhibitory_weight_init, inhibitory_postsynaptic_init,
    II_synapse_init, inhibitory_postsynaptic_init,
    #init_sparse_connectivity("FixedProbabilityNoAutapse", fixed_prob))
    #init_sparse_connectivity("FixedNumberPostWithReplacement", {"num": 3000*0.2}))
    init_sparse_connectivity(fixed_number_post, {"num": int(K*(1-(EXCITATORY_INHIBITORY_RATIO/(1+EXCITATORY_INHIBITORY_RATIO)))), "sigma_space": sigma/L*n_side_I, "grid_num_x": int(n_side_I), "grid_num_x2": int(n_side_I)}))
    
IE_syn_pop = model.add_synapse_population("IE", "SPARSE",
    inhibitory_pop, excitatory_pop,
    #inhibitory_weight_init, inhibitory_postsynaptic_init,
    IE_synapse_init, inhibitory_postsynaptic_init,
    #init_sparse_connectivity("FixedProbability", fixed_prob))
    #init_sparse_connectivity("FixedNumberPostWithReplacement", {"num": 3000*0.8}))
    init_sparse_connectivity(fixed_number_post, {"num": int(K*EXCITATORY_INHIBITORY_RATIO/(1+EXCITATORY_INHIBITORY_RATIO)), "sigma_space": sigma/L*n_side_E, "grid_num_x": int(n_side_I), "grid_num_x2": int(n_side_E)}))

print(int((2*(400**2))**0.5 / (conduction_delay * model.dt)))
EE_syn_pop.max_dendritic_delay_timesteps = int((2*(400**2))**0.5 / (conduction_delay * model.dt))
EI_syn_pop.max_dendritic_delay_timesteps = int((2*(400**2))**0.5 / (conduction_delay * model.dt))
II_syn_pop.max_dendritic_delay_timesteps = int((2*(400**2))**0.5 / (conduction_delay * model.dt))
IE_syn_pop.max_dendritic_delay_timesteps = int((2*(400**2))**0.5 / (conduction_delay * model.dt))
print("Building Model")
model.build()
print("Loading Model")
model.load(num_recording_timesteps=1000)

sim_start_time = time()

while model.timestep < 200:
    model.step_time()


input_pop1.vars["rate"].pull_from_device()
input_pop1.vars["rate"].values = 0
input_pop1.vars["rate"].push_to_device()

input_pop2.vars["rate"].pull_from_device()
input_pop2.vars["rate"].values = 0
input_pop2.vars["rate"].push_to_device()

while model.timestep < 1000:
    model.step_time()

sim_end_time = time()
print("Simulation time:%fs" % (sim_end_time - sim_start_time))

# Download recording data
model.pull_recording_buffers_from_device()

exc_spike_times, exc_spike_ids = excitatory_pop.spike_recording_data[0]
inh_spike_times, inh_spike_ids = inhibitory_pop.spike_recording_data[0]
print(len(exc_spike_times))

fig, axes = plt.subplots(3, sharex=True, figsize=(20, 10))

# Define some bins to calculate spike rates
bin_size = 10.0
rate_bins = np.arange(0, 1000.0/ time_div, bin_size)
rate_bin_centres = rate_bins[:-1] + (bin_size / 2.0)

# Plot excitatory and inhibitory spikes on first axis
axes[0].scatter(exc_spike_times, exc_spike_ids, s=0.01)
axes[0].scatter(inh_spike_times, inh_spike_ids + NUM_EXCITATORY, s=0.01)

# Plot excitatory rates on second axis
exc_rate = np.histogram(exc_spike_times, bins=rate_bins)[0]
axes[1].plot(rate_bin_centres, exc_rate * (1000.0 / bin_size) * (1.0 / NUM_EXCITATORY))

# Plot inhibitory rates on third axis
inh_rate = np.histogram(inh_spike_times, bins=rate_bins)[0]
axes[2].plot(rate_bin_centres, inh_rate * (100.0 / bin_size) * (1.0 / NUM_INHIBITORY))

# Label axes
axes[0].set_ylabel("Neuron ID")
axes[1].set_ylabel("Excitatory rate [Hz]")
axes[2].set_ylabel("Inhibitory rate [Hz]")
axes[2].set_xlabel("Time [ms]");

plt.savefig("wave.png")

side = int(np.sqrt(NUM_EXCITATORY))  # should be 900
print(side)  # 900

crop = side // 6          # 900 // 6 = 150
r0, r1 = crop, side - crop
c0, c1 = crop, side - crop

cropped_side = r1 - r0    # should be 600


exc_spike_times = np.array(exc_spike_times)  # shape (n_spikes,)
exc_spike_ids   = np.array(exc_spike_ids)    # shape (n_spikes,)

t_min = exc_spike_times.min()
t_max = exc_spike_times.max()

#dt = 10.0  # ms per frame (adjust to taste)
time_bins = np.arange(t_min, t_max + model.dt*10, model.dt*10)
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

# Log scale helps a lot with spike data
frames = np.log1p(frames)

# Normalize to [0, 1]
frames /= frames.max()

import matplotlib.pyplot as plt
import imageio

gif_path = "spiking_activity.gif"
images = []

for i in range(n_frames):
    print(i)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(frames[i], cmap="hot", vmin=0, vmax=1)
    ax.set_title(f"Time: {time_bins[i]:.1f}–{time_bins[i+1]:.1f} ms")
    ax.axis("off")

    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    images.append(image)

    plt.close(fig)

imageio.mimsave(gif_path, images, duration=0.1, loop=10 )  # seconds per frame
