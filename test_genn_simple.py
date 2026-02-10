
import numpy as np
import matplotlib.pyplot as plt
import pygenn
from pygenn import GeNNModel, init_postsynaptic, init_sparse_connectivity, init_var, init_weight_update, create_var_ref, create_sparse_connect_init_snippet, create_var_init_snippet, create_neuron_model

model = GeNNModel("float", "tutorial_wave")
model.dt = 1.0

lif_params = {"C": 0.2*15, "TauM": 20.0, "Vrest": 0.0, "Vreset": 0.0,
              "Vthresh": 1.0, "Ioffset": 0.0, "TauRefrac": 0.0}



lif_init = {"V": init_var("Normal", {"mean": -0.21, "sd": 0.38}),
            "RefracTime": 0.0}



poisson_model = create_neuron_model(
    "poisson",

    threshold_condition_code="gennrand_uniform() >= exp(-rate * dt / 1000.0)",
    vars= [("rate", "scalar")]
)
poisson_init = {"rate": 20.0}


exc_ratio = 0.8
inh_ratio = 0.2
all_neuron = 1_012_500/2
sigma = 400.0
K = 3000
exc_num = int(all_neuron * exc_ratio)
inh_num = int(all_neuron * inh_ratio)
input_num1 = 10 * 10
input_num2 = 4 * 4
L = 6000.0


n_side_E = int(np.ceil(np.sqrt(exc_num)))
n_side_I = int(np.ceil(np.sqrt(inh_num)))
EE_dist = L/n_side_E
II_dist = L/n_side_I




input_to_hid = create_sparse_connect_init_snippet(
    "input_to_hid",
    params=[("num", "unsigned int"), ("loc", "int"), ("grid_num_x", "unsigned int")],
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
        const float ratio = (float)grid_num_x2 / (float)grid_num_x;
        const int xPre = (id_pre % grid_num_x) * ratio;
        const int yPre = (id_pre / grid_num_x) * ratio;
        int count = num;
        while(count > 0) {
            const int distanceX = (int)round(gennrand_normal() * sigma_space);
            const int distanceY = (int)round(gennrand_normal() * sigma_space);
            int xPost = xPre + distanceX;
            int yPost = yPre + distanceY;
            if((xPost < 0 || xPost >= grid_num_x2 || yPost < 0 || yPost >= grid_num_x2)){
                continue;
            }
            count--;
            const int id_post = (yPost * grid_num_x2) + xPost;
            addSynapse(id_post);
        }
        """,   
        calc_max_row_len_func=lambda num_pre, num_post, pars: pars["num"])



calc_dist = create_var_init_snippet(
    "calc_dist",
    
    params=[("delay", "float"),("grid_num_x", "unsigned int")],
    var_init_code=
        """
        const float xPre = id_pre % grid_num_x;
        const float yPre = id_pre / grid_num_x;
        const float xPost = id_post % grid_num_x;
        const float yPost = id_post / grid_num_x;
        float dist = (float)sqrt(pow(xPre - xPost, 2) + pow(yPre - yPost, 2));
        value = dist * delay;
        """
    )

calc_dist_resize = create_var_init_snippet(
    "calc_dist_resize",
    
    params=[("delay", "float"),("grid_num_x", "unsigned int"), ("grid_num_x2", "unsigned int")],
    var_init_code=
        """
        const float ratio = (float)grid_num_x2 / (float)grid_num_x;
        const int xPre = (id_pre % grid_num_x) * ratio;
        const int yPre = (id_pre / grid_num_x) * ratio;
        const float xPost = id_post % grid_num_x2;
        const float yPost = id_post / grid_num_x2;
        float dist = (float)sqrt(pow(xPre - xPost, 2) + pow(yPre - yPost, 2));
        value = dist * delay;
        """
    )


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

input_pop1 = model.add_neuron_population("Input1", input_num1, poisson_model, {}, poisson_init)
input_pop2 = model.add_neuron_population("Input2", input_num2, poisson_model, {}, poisson_init)
exc_pop = model.add_neuron_population("E", exc_num, "LIF", lif_params, lif_init)
inh_pop = model.add_neuron_population("I", inh_num, "LIF", lif_params, lif_init)

exc_pop.spike_recording_enabled = True
inh_pop.spike_recording_enabled = True


conduction_delay = 0.2 #mm/ms
EE_synapse_init = {"g": 0.001*4.5, "d": init_var(calc_dist, {"delay": EE_dist*conduction_delay, "grid_num_x": int(n_side_E)})}
EI_synapse_init = {"g": 0.001*4.5, "d": init_var(calc_dist_resize, {"delay": II_dist*conduction_delay, "grid_num_x": int(n_side_E), "grid_num_x2": int(n_side_I)})}
II_synapse_init = {"g": -0.01, "d": init_var(calc_dist, {"delay": II_dist*conduction_delay, "grid_num_x": int(n_side_I)})}
IE_synapse_init = {"g": -0.01, "d": init_var(calc_dist_resize, {"delay": EE_dist*conduction_delay, "grid_num_x": int(n_side_I), "grid_num_x2": int(n_side_E)})}

start_loc_x = np.random.randint(0, n_side_E-int(np.sqrt(input_num1)))
start_loc_y = np.random.randint(0, n_side_E-int(np.sqrt(input_num1)))
InputE_syn_pop = model.add_synapse_population("InputE", "SPARSE",
    input_pop1, exc_pop,
    init_weight_update("StaticPulseConstantWeight", {"g": 5.0}),
    init_postsynaptic("ExpCurr", {"tau": 5.0}),
    init_sparse_connectivity(input_number_post, {"num": int(input_num1), "start_loc_x": start_loc_x, "start_loc_y": start_loc_y, "grid_num_x": int(n_side_E)}))

InputI_syn_pop = model.add_synapse_population("InputI", "SPARSE",
    input_pop2, inh_pop,
    init_weight_update("StaticPulseConstantWeight", {"g": 5.0}),
    init_postsynaptic("ExpCurr", {"tau": 5.0}),
    init_sparse_connectivity(input_number_post, {"num": int(input_num2), "start_loc_x": start_loc_x/2, "start_loc_y": start_loc_y/2, "grid_num_x": int(n_side_I)}))



EE_syn_pop = model.add_synapse_population("EE", "SPARSE",
    exc_pop, exc_pop,
    init_weight_update("StaticPulseDendriticDelay", {}, EE_synapse_init),
    init_postsynaptic("ExpCurr", {"tau": 5.0}),
    init_sparse_connectivity(fixed_number_post, {"num": int(K*exc_ratio), "sigma_space": sigma/L*n_side_E, "grid_num_x": int(n_side_E)}))




EI_syn_pop = model.add_synapse_population("EI", "SPARSE",
    exc_pop, inh_pop,
    init_weight_update("StaticPulseDendriticDelay", {}, EI_synapse_init),
    init_postsynaptic("ExpCurr", {"tau": 5.0}),
    init_sparse_connectivity(fixed_number_post_resize, {"num": int(K*inh_ratio), "sigma_space": sigma/L*n_side_I, "grid_num_x": int(n_side_E), "grid_num_x2": int(n_side_I)}))

II_syn_pop = model.add_synapse_population("II", "SPARSE",
    inh_pop, inh_pop,
    init_weight_update("StaticPulseDendriticDelay", {}, II_synapse_init),
    init_postsynaptic("ExpCurr", {"tau": 5.0}),
    init_sparse_connectivity(fixed_number_post, {"num": int(K*inh_ratio), "sigma_space": sigma/L*n_side_I, "grid_num_x": int(n_side_I)}))


IE_syn_pop = model.add_synapse_population("IE", "SPARSE",
    inh_pop, exc_pop,
    init_weight_update("StaticPulseDendriticDelay", {}, IE_synapse_init),
    init_postsynaptic("ExpCurr", {"tau": 5.0}),
    init_sparse_connectivity(fixed_number_post_resize, {"num": int(K*exc_ratio), "sigma_space": sigma/L*n_side_E, "grid_num_x": int(n_side_I), "grid_num_x2": int(n_side_E)}))

EE_syn_pop.max_dendritic_delay_timesteps = int((2*(400**2))**0.5 * (conduction_delay / model.dt))
EI_syn_pop.max_dendritic_delay_timesteps = int((2*(400**2))**0.5 * (conduction_delay / model.dt))
II_syn_pop.max_dendritic_delay_timesteps = int((2*(400**2))**0.5 * (conduction_delay / model.dt))
IE_syn_pop.max_dendritic_delay_timesteps = int((2*(400**2))**0.5 * (conduction_delay / model.dt))

model.build()


model.load(num_recording_timesteps=1000)


while model.timestep < 10:
    model.step_time()


input_pop1.vars["rate"].pull_from_device()
input_pop1.vars["rate"].values = 0
input_pop1.vars["rate"].push_to_device()

input_pop2.vars["rate"].pull_from_device()
input_pop2.vars["rate"].values = 0
input_pop2.vars["rate"].push_to_device()

while model.timestep < 1000:
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

