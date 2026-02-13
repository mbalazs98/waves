from pygenn import (create_sparse_connect_init_snippet, create_var_init_snippet,  create_weight_update_model,
                    init_sparse_connectivity)


from typing import Optional
from pygenn import SynapseMatrixType
from typing import TYPE_CHECKING
from ml_genn.connectivity import Connectivity
from ml_genn.utils.snippet import ConnectivitySnippet
from ml_genn.utils.value import InitValue
TYPE_CHECKING = True
if TYPE_CHECKING:
    from ml_genn import Connection, Population
    from ml_genn.compilers.compiler import SupportedMatrixType

from pygenn import (create_sparse_connect_init_snippet, init_sparse_connectivity)

from ml_genn.initializers import Wrapper



StaticPulseDendriticDelayConstantWeight = create_weight_update_model(
    "StaticPulseDendriticDelayConstantWeight",
    params=["g"],
    vars=[("d", "uint8_t")],

    pre_spike_syn_code=
        """
        addToPostDelay(g, d);""")


fixed_number_post = create_sparse_connect_init_snippet(
    "fixed_number_post_resize",
    params=[("num", "unsigned int"), ("sigma_space", "float"), ("grid_num_x", "unsigned int"), ("grid_num_x2", "unsigned int")],
    row_build_code=
        """
        const float ratio = (float)(grid_num_x2-1) / (float)(grid_num_x-1);
        const float xPre = (id_pre % grid_num_x) * ratio;
        const float yPre = (id_pre / grid_num_x) * ratio;
        int count = num;
        while(count > 0) {
            const float distanceX = (float)gennrand_normal() * sigma_space;
            const float distanceY = (float)gennrand_normal() * sigma_space;
            int xPost = (int)round(xPre + distanceX);
            int yPost = (int)round(yPre + distanceY);
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
    
    params=[("delay", "float"),("grid_num_x", "unsigned int"), ("grid_num_x2", "unsigned int")],
    var_init_code=
        """
        const float ratio = (float)(grid_num_x2 -1 ) / (float)(grid_num_x-1);
        const float xPre = (id_pre % grid_num_x) * ratio;
        const float yPre = (id_pre / grid_num_x) * ratio;
        const float xPost = id_post % grid_num_x2;
        const float yPost = id_post / grid_num_x2;
        float dist = (float)sqrt(pow(xPre - xPost, 2) + pow(yPre - yPost, 2));
        delays = dist * delay;
        """
    )



class TopoGraphic(Connectivity):
    """Topographic connectivity with fixed number of post-synaptic connections.
    
    Creates spatially localized random connections where each neuron connects to
    a fixed number of nearby neurons. Can optionally resize between source and 
    target populations.
    
    Args:
        weight:         Synaptic weights. Must be either a constant value,
                        a :class:`ml_genn.initializers.Initializer` or a numpy array.
        num:            Number of post-synaptic connections per neuron.
        sigma_space:    Spatial spread (standard deviation) for connection placement.
        grid_num_x:     Number of neurons in a column/row of source population.
        grid_num_x2:    Optional number of neurons in a column/row of target population. If provided,
                        uses resize version for different source/target grid sizes.
        delay:          Connection delays
    """
    def __init__(self, weight: InitValue, num: int, sigma_space: float, 
                 grid_num_x: int, grid_num_x2: Optional[int] = None, spatial_delay: bool = True, cond_vel: Optional[float] = None, 
                 delay: InitValue = 0):
        super(TopoGraphic, self).__init__(weight, delay)
        
        self.num = int(num)
        self.sigma_space = sigma_space
        self.grid_num_x = grid_num_x
        if grid_num_x2 is None:
            grid_num_x2 = grid_num_x
        self.grid_num_x2 = grid_num_x2
        self.spatial_delay = spatial_delay
        self.cond_vel = cond_vel
        print(self.cond_vel)

    def connect(self, source: Population, target: Population):
        pass

    def get_snippet(self, connection: Connection,
                    supported_matrix_type: SupportedMatrixType) -> ConnectivitySnippet:
        snippet = fixed_number_post
        params = {
            "num": self.num,
            "sigma_space": self.sigma_space,
            "grid_num_x": self.grid_num_x,
            "grid_num_x2": self.grid_num_x2
        }
        conn_init = init_sparse_connectivity(snippet, params)

        if self.spatial_delay:
            delay_snippet = calc_dist
            delay_params = {
                "delay": self.cond_vel,
                "grid_num_x": self.grid_num_x,
                "grid_num_x2": self.grid_num_x2
            }
            delay_init = Wrapper(delay_snippet, delay_params, {"delays": self.delay})
        else:
            delay_init = self.delay
        return ConnectivitySnippet(
            snippet=conn_init,
            matrix_type=SynapseMatrixType.SPARSE,
            weight=self.weight,
            delay=delay_init)
