from pygenn import (create_sparse_connect_init_snippet, create_var_init_snippet,  create_weight_update_model,
                    init_sparse_connectivity)

from __future__ import annotations

from pygenn import Optional, SynapseMatrixType
from typing import TYPE_CHECKING
from ml_genn.connectivity import Connectivity
from ml_genn.utils.snippet import ConnectivitySnippet
from ..utils.value import InitValue

if TYPE_CHECKING:
    from .. import Connection, Population
    from ..compilers.compiler import SupportedMatrixType

from pygenn import (create_sparse_connect_init_snippet, init_sparse_connectivity)



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
                 grid_num_x: int, grid_num_x2: Optional[int] = None, 
                 delay: InitValue = 0):
        super(TopoGraphic, self).__init__(weight, delay)
        
        self.num = num
        self.sigma_space = sigma_space
        self.grid_num_x = grid_num_x
        if grid_num_x2 is None:
            grid_num_x2 = grid_num_x
        self.grid_num_x2 = grid_num_x2

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

        
        return ConnectivitySnippet(
            snippet=conn_init,
            matrix_type=SynapseMatrixType.SPARSE,
            weight=self.weight,
            delay=self.delay,
            conn_init_params=params)
