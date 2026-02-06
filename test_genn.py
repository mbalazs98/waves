
import numpy as np
import matplotlib.pyplot as plt

from pygenn import GeNNModel, init_postsynaptic, init_sparse_connectivity, init_var, init_weight_update, create_var_ref, create_sparse_connect_init_snippet, create_var_init_snippet

model = GeNNModel("float", "tutorial_wave")
model.dt = 0.1

lif_params = {"C": 0.2, "TauM": 20.0, "Vrest": -65.0, "Vreset": -70.0,
              "Vthresh": -50.0, "Ioffset": 0.0, "TauRefrac": 5.0}


lif_init = {"V": init_var("Normal", {"mean": -68.0901, "sd": 5.7296}),
            "RefracTime": 0.0}





exc_ratio = 0.8
inh_ratio = 0.2
all_neuron = 1_012_500
sigma = 400.0
K = 3000
exc_num = int(all_neuron * exc_ratio)
inh_num = int(all_neuron * inh_ratio)
L = 6000.0


n_side_E = int(np.ceil(np.sqrt(exc_num)))
n_side_I = int(np.ceil(np.sqrt(inh_num)))

EE_dist = L/n_side_E





fixed_number_post = create_sparse_connect_init_snippet(
    "fixed_number_post",
    params=[("num", "unsigned int"), ("sigma_space", "float"), ("grid_num_x", "unsigned int")],
    row_build_code=
        """
        const int xPre = id_pre % grid_num_x;
        const int yPre = id_pre / grid_num_x;
        int count = num;
        while(count > 0) {
            const int distanceX = (int)round(gennrand_normal() * sigma_space);
            const int distanceY = (int)round(gennrand_normal() * sigma_space);
            int xPost = xPre + distanceX;
            int yPost = yPre + distanceY;
            if((distanceX == 0 && distanceY == 0) || (xPost < 0 || xPost >= grid_num_x || yPost < 0 || yPost >= grid_num_x)){
                continue;
             }
            count--;
            const int id_post = (yPost * grid_num_x) + xPost;
            addSynapse(id_post);
        }
        """,   
        calc_max_row_len_func=lambda num_pre, num_post, pars: pars["num"])

fixed_number_post_resize = create_sparse_connect_init_snippet(
    "fixed_number_post_resize",
    params=[("num", "unsigned int"), ("sigma_space", "float"), ("grid_num_x", "unsigned int"), ("grid_num_x2", "unsigned int")],
    row_build_code=
        """
        const int xPre = id_pre % grid_num_x;
        const int yPre = id_pre / grid_num_x;
        int count = num;
        while(count > 0) {
            const float distanceX = gennrand_normal() * sigma_space;
            const float distanceY = gennrand_normal() * sigma_space;
            int xPost = (int)round((xPre + distanceX)/grid_num_x/grid_num_x2);
            int yPost = (int)round((yPre + distanceX)/grid_num_x/grid_num_x2);
            if((xPost < 0 || xPost >= grid_num_x2 || yPost < 0 || yPost >= grid_num_x2)){
                continue;
            }
            count--;
            const int id_post = (yPost * grid_num_x2) + xPost;
            addSynapse(id_post);
        }
        """,   
        calc_max_row_len_func=lambda num_pre, num_post, pars: pars["num"])



exc_pop = model.add_neuron_population("E", exc_num, "LIF", lif_params, lif_init)
inh_pop = model.add_neuron_population("I", inh_num, "LIF", lif_params, lif_init)

exc_pop.spike_recording_enabled = True
inh_pop.spike_recording_enabled = True


conduction_delay = 0.2 #mm/ms

exc_synapse_init = {"g": 0.001}
inh_synapse_init = {"g": -0.01}


exc_post_syn_params = {"tau": 5.0, "E": 0.0}
inh_post_syn_params = {"tau": 5.0, "E": -80.0}





EE_syn_pop = model.add_synapse_population("EE", "SPARSE",
    exc_pop, exc_pop,
    init_weight_update("StaticPulseConstantWeight", exc_synapse_init),
    #init_weight_update("StaticPulseDendriticDelay", EE_synapse_init),
    init_postsynaptic("ExpCond", exc_post_syn_params, var_refs={"V": create_var_ref(exc_pop, "V")}),
    init_sparse_connectivity(fixed_number_post, {"num": int(K*exc_ratio), "sigma_space": sigma/L*n_side_E, "grid_num_x": int(n_side_E)}))



EI_syn_pop = model.add_synapse_population("EI", "SPARSE",
    exc_pop, inh_pop,
    init_weight_update("StaticPulseConstantWeight", exc_synapse_init),
    init_postsynaptic("ExpCond", exc_post_syn_params, var_refs={"V": create_var_ref(inh_pop, "V")}),
    init_sparse_connectivity(fixed_number_post_resize, {"num": int(K*inh_ratio), "sigma_space": sigma/L*n_side_I, "grid_num_x": int(n_side_E), "grid_num_x2": int(n_side_I)}))

II_syn_pop = model.add_synapse_population("II", "SPARSE",
    inh_pop, inh_pop,
    init_weight_update("StaticPulseConstantWeight", inh_synapse_init),
    init_postsynaptic("ExpCond", inh_post_syn_params, var_refs={"V": create_var_ref(inh_pop, "V")}),
    init_sparse_connectivity(fixed_number_post, {"num": int(K*inh_ratio), "sigma_space": sigma/L*n_side_I, "grid_num_x": int(n_side_I)}))

IE_syn_pop = model.add_synapse_population("IE", "SPARSE",
    inh_pop, exc_pop,
    init_weight_update("StaticPulseConstantWeight", inh_synapse_init),
    init_postsynaptic("ExpCond", inh_post_syn_params, var_refs={"V": create_var_ref(exc_pop, "V")}),
    init_sparse_connectivity(fixed_number_post_resize, {"num": int(K*exc_ratio), "sigma_space": sigma/L*n_side_E, "grid_num_x": int(n_side_I), "grid_num_x2": int(n_side_E)}))


model.build()


model.load(num_recording_timesteps=10000)


while model.timestep < 10000:
    model.step_time()


model.pull_recording_buffers_from_device()

exc_spike_times, exc_spike_ids = exc_pop.spike_recording_data[0]
inh_spike_times, inh_spike_ids = inh_pop.spike_recording_data[0]


fig, axes = plt.subplots(3, sharex=True, figsize=(20, 10))

# Define some bins to calculate spike rates
bin_size = 20.0
rate_bins = np.arange(0, 1000.0, bin_size)
rate_bin_centres = rate_bins[:-1] + (bin_size / 2.0)

# Plot excitatory and inhibitory spikes on first axis
axes[0].scatter(exc_spike_times, exc_spike_ids, s=1)
axes[0].scatter(inh_spike_times, inh_spike_ids + exc_num, s=1)

# Plot excitatory rates on second axis
exc_rate = np.histogram(exc_spike_times, bins=rate_bins)[0]
axes[1].plot(rate_bin_centres, exc_rate * (1000.0 / bin_size) * (1.0 / exc_num))

# Plot inhibitory rates on third axis
inh_rate = np.histogram(inh_spike_times, bins=rate_bins)[0]
axes[2].plot(rate_bin_centres, inh_rate * (1000.0 / bin_size) * (1.0 / exc_num))

# Label axes
axes[0].set_ylabel("Neuron ID")
axes[1].set_ylabel("Excitatory rate [Hz]")
axes[2].set_ylabel("Inhibitory rate [Hz]")
axes[2].set_xlabel("Time [ms]");

plt.savefig("wave.png")

