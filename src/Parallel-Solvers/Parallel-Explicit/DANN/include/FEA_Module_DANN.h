/**********************************************************************************************
 � 2020. Triad National Security, LLC. All rights reserved.
 This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
 National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
 Department of Energy/National Nuclear Security Administration. All rights in the program are
 reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
 Security Administration. The Government is granted for itself and others acting on its behalf a
 nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
 derivative works, distribute copies to the public, perform publicly and display publicly, and
 to permit others to do so.
 This program is open source under the BSD-3 License.
 Redistribution and use in source and binary forms, with or without modification, are permitted
 provided that the following conditions are met:
 1.  Redistributions of source code must retain the above copyright notice, this list of
 conditions and the following disclaimer.
 2.  Redistributions in binary form must reproduce the above copyright notice, this list of
 conditions and the following disclaimer in the documentation and/or other materials
 provided with the distribution.
 3.  Neither the name of the copyright holder nor the names of its contributors may be used
 to endorse or promote products derived from this software without specific prior
 written permission.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **********************************************************************************************/

#ifndef FEA_MODULE_DANN_H
#define FEA_MODULE_DANN_H

#include "mesh.h"
#include "state.h"
#include "matar.h"
#include "elements.h"
#include "node_combination.h"
#include "dynamic_checkpoint.h"
#include "FEA_Module.h"
#include "material_models.h"
#include <set>

class Explicit_Solver;

class Solver;

class Simulation_Parameters_Explicit;

class DANN_Parameters;

struct material_t;

struct boundary_t;

/////////////////////////////////////////////////////////////////////////////
///
/// \class FEA_Module_DANN
///
/// \brief Class for containing functions required to perform SGH
///
/// This class containts the requisite functions requited to perform
/// staggered grid hydrodynamics (SGH) which is equivalent to a lumped
/// mass finite element (FE) scheme.
///
/////////////////////////////////////////////////////////////////////////////
class FEA_Module_DANN : public FEA_Module
{
public:

    FEA_Module_DANN(SGH_Parameters& params, Solver* Solver_Pointer, std::shared_ptr<mesh_t> mesh_in, const int my_fea_module_index = 0);
    ~FEA_Module_DANN();

    // initialize data for boundaries of the model and storage for boundary conditions and applied loads
    void dann_interface_setup(node_t& node, elem_t& elem, corner_t& corner);

    void setup();

    void cleanup_material_models();

    int solve();

    void checkpoint_solve(std::set<Dynamic_Checkpoint>::iterator start_checkpoint, size_t bounding_timestep);

    void module_cleanup();

    void dann_solve();

    void update_state_dann(double rk_alpha,
                             const size_t num_nodes,
                             DViewCArrayKokkos<double>& node_states);

    void init_assembly();

    void rk_init(DViewCArrayKokkos<double>& node_states,
                 const size_t num_elems,
                 const size_t num_nodes);

    void get_timestep(mesh_t& mesh,
                      DViewCArrayKokkos<double>& node_coords,
                      DViewCArrayKokkos<double>& node_vel,
                      DViewCArrayKokkos<double>& elem_sspd,
                      DViewCArrayKokkos<double>& elem_vol);

    void update_state(const DCArrayKokkos<material_t>& material,
                      const mesh_t& mesh,
                      const DViewCArrayKokkos<double>& node_states,
                      const double rk_alpha,
                      const size_t cycle);

    virtual void update_forward_solve(Teuchos::RCP<const MV> zp, bool print_design=false);

    void comm_adjoint_vector(int cycle);

    void comm_variables(Teuchos::RCP<const MV> zp);

    void init_output();

    void compute_output();

    void output_control();

    void sort_output(Teuchos::RCP<Tpetra::Map<LO, GO, node_type>> sorted_map);

    void sort_element_output(Teuchos::RCP<Tpetra::Map<LO, GO, node_type>> sorted_map);

    void collect_output(Teuchos::RCP<Tpetra::Map<LO, GO, node_type>> global_reduce_map);

    void write_data(std::map<std::string, const double*>& point_data_scalars_double,
                    std::map<std::string, const double*>& point_data_vectors_double,
                    std::map<std::string, const double*>& cell_data_scalars_double,
                    std::map<std::string, const int*>&    cell_data_scalars_int,
                    std::map<std::string, std::pair<const double*, size_t>>& cell_data_fields_double);

    void write_outputs(const mesh_t& mesh,
                       DViewCArrayKokkos<double>& node_states);

    void ensight(const mesh_t& mesh,
                 const DViewCArrayKokkos<double>& node_states);

    void state_file(const mesh_t& mesh,
                    const DViewCArrayKokkos<double>& node_states);

    void weight_constraints(host_vec_array weights_lower_bound);

    void compute_optimization_adjoint_full(Teuchos::RCP<const MV> design_weights_distributed); // Force depends on node coords, velocity, and sie

    void compute_optimization_gradient_full(Teuchos::RCP<const MV> design_weights_distributed, Teuchos::RCP<MV> design_gradients_distributed);

    void compute_optimization_gradient_tally(Teuchos::RCP<const MV> design_weights_distributed, Teuchos::RCP<MV> design_gradients_distributed,
                                                      unsigned long cycle, real_t global_dt);


    Simulation_Parameters_Explicit* simparam;
    DANN_Parameters*  module_params;
    Explicit_Solver* Explicit_Solver_Pointer_;

    elements::ref_element* ref_elem;

    std::shared_ptr<mesh_t> mesh;
    // shallow copies of mesh class views
    size_t num_weights_in_node;

    // elem ids in node
    RaggedRightArrayKokkos<size_t> weights_in_node;

    // node ids in node
    RaggedRightArrayKokkos<size_t> nodes_in_node;
    CArrayKokkos<size_t> num_nodes_in_node;

    // Global FEA data
    Teuchos::RCP<MV> previous_node_states_distributed;
    Teuchos::RCP<MV> initial_node_states_distributed;
    Teuchos::RCP<MV> all_node_states_distributed;
    Teuchos::RCP<MV> all_cached_node_states_distributed;
    Teuchos::RCP<MV> cached_design_gradients_distributed;
    Teuchos::RCP<MV> all_adjoint_vector_distributed, adjoint_vector_distributed;
    Teuchos::RCP<MV> previous_adjoint_vector_distributed;
    Teuchos::RCP<std::vector<Teuchos::RCP<MV>>> forward_solve_state_data;
    Teuchos::RCP<std::vector<Teuchos::RCP<MV>>> adjoint_vector_data;
    //TpetraMVArray<real_t, array_layout, device_type, memory_traits> mtr_node_velocities_distributed;
    // TpetraPartitionMap<long long int, array_layout, device_type, memory_traits> mtr_map;
    // TpetraPartitionMap<long long int, array_layout, device_type, memory_traits> mtr_local_map;

    // Local FEA data
    DCArrayKokkos<size_t, array_layout, device_type, memory_traits>      Global_Gradient_Matrix_Assembly_Map; // Maps element local nodes to columns on ragged right node connectivity graph
    RaggedRightArrayKokkos<LO, array_layout, device_type, memory_traits> Graph_Matrix; // stores local indices
    DCArrayKokkos<size_t, array_layout, device_type, memory_traits>          Gradient_Matrix_Strides;
    DCArrayKokkos<size_t, array_layout, device_type, memory_traits>          Graph_Matrix_Strides;
    RaggedRightArrayKokkos<real_t, array_layout, device_type, memory_traits> Original_Gradient_Entries;
    RaggedRightArrayKokkos<LO, array_layout, device_type, memory_traits>     Original_Gradient_Entry_Indices;
    DCArrayKokkos<size_t, array_layout, device_type, memory_traits>          Original_Gradient_Entries_Strides;

    // distributed matrices
    Teuchos::RCP<MAT> distributed_weights;

    std::vector<real_t> time_data;
    int max_time_steps, last_time_step;

    // ---------------------------------------------------------------------
    //    state data type declarations (must stay in scope for output after run)
    // ---------------------------------------------------------------------
    node_t  node_interface;

    // Dual View wrappers
    // Dual Views of the individual node struct variables
    DViewCArrayKokkos<double> node_states;

    // Boundary Conditions Data
    DCArrayKokkos<size_t> Local_Index_Boundary_Patches;
    // CArray <Nodal_Combination> Patch_Nodes;
    enum bc_type { NONE, POINT_LOADING_CONDITION, LINE_LOADING_CONDITION, SURFACE_LOADING_CONDITION };

    // Boundary Conditions Data
    int max_boundary_sets;

    // output dof data
    // Global arrays with collected data used to print
    int output_state_index;

    // parameters
    double time_value, time_final, dt, dt_max, dt_min, dt_cfl, graphics_time, graphics_dt_ival;
    size_t graphics_cyc_ival, cycle_stop, rk_num_stages, graphics_id;
    double fuzz, tiny, small;
    CArray<double> graphics_times;
    int rk_num_bins;

    // optimization flags and data
    Teuchos::RCP<std::set<Dynamic_Checkpoint>> dynamic_checkpoint_set;
    Teuchos::RCP<std::vector<Dynamic_Checkpoint>> cached_dynamic_checkpoints;
    int num_active_checkpoints;
    enum vector_name { S_DATA=0 };
};

#endif // end HEADER_H
