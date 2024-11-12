/**********************************************************************************************
 © 2020. Triad National Security, LLC. All rights reserved.
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

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // fmin, fmax, abs note: fminl is long
#include <sys/stat.h>
#include <mpi.h>
#include <chrono>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_Tuple.hpp>
#include <Teuchos_VerboseObject.hpp>

#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include "Tpetra_Import.hpp"
#include "Tpetra_Import_Util2.hpp"

#include "elements.h"
#include "swage.h"
#include "matar.h"
#include "utilities.h"
#include "node_combination.h"
#include "dynamic_checkpoint.h"
#include "Simulation_Parameters/Simulation_Parameters_Explicit.h"
#include "Simulation_Parameters/FEA_Module/DANN_Parameters.h"
#include "FEA_Module_DANN.h"
#include "Explicit_Solver.h"

// optimization
#include "ROL_Solver.hpp"
#include "Fierro_Optimization_Objective.hpp"

#define MAX_ELEM_NODES 8
#define STRAIN_EPSILON 0.000000001
#define DENSITY_EPSILON 0.0001
#define BC_EPSILON 1.0e-6
#define BUFFER_GROW 100

// #define DEBUG

using namespace utils;

FEA_Module_DANN::FEA_Module_DANN(
    DANN_Parameters& params, Solver* Solver_Pointer,
    std::shared_ptr<mesh_t> mesh_in, const int my_fea_module_index)
    : FEA_Module(Solver_Pointer)
{
    // assign interfacing index
    my_fea_module_index_ = my_fea_module_index;
    Module_Type = FEA_MODULE_TYPE::DANN;

    // recast solver pointer for non-base class access
    Explicit_Solver_Pointer_ = dynamic_cast<Explicit_Solver*>(Solver_Pointer);
    simparam = &(Explicit_Solver_Pointer_->simparam);
    module_params = &params;

    // create ref element object
    // ref_elem = new elements::ref_element();
    // create mesh objects
    // init_mesh = new swage::mesh_t(simparam);
    // mesh = new swage::mesh_t(simparam);

    mesh = mesh_in;

    // set Tpetra vector pointers
    // initial_node_states_distributed = Explicit_Solver_Pointer_->initial_node_states_distributed;
    // node_states_distributed     = Explicit_Solver_Pointer_->node_states_distributed;
    // all_node_states_distributed = Explicit_Solver_Pointer_->all_node_states_distributed;
    
    initial_node_states_distributed = Teuchos::rcp(new MV(map, 1));
    all_node_states_distributed = Teuchos::rcp(new MV(all_node_map, 1));
    all_previous_node_states_distributed = Teuchos::rcp(new MV(all_node_map, 1));
    previous_node_states_distributed = Teuchos::rcp(new MV(*all_previous_node_states_distributed, map));
    node_states_distributed = Teuchos::rcp(new MV(*all_node_states_distributed, map));

}

FEA_Module_DANN::~FEA_Module_DANN()
{
    // delete simparam;
}

/////////////////////////////////////////////////////////////////////////////
///
/// \fn output_control
///
/// \brief Output field settings and file settings
///
/////////////////////////////////////////////////////////////////////////////
void FEA_Module_DANN::output_control()
{
}


/////////////////////////////////////////////////////////////////////////////
///
/// \fn sort_output
///
/// \brief Prompts sorting for elastic response output data. For now, nodal strains.
///        Inactive in this context
///
/////////////////////////////////////////////////////////////////////////////
void FEA_Module_DANN::sort_output(Teuchos::RCP<Tpetra::Map<LO, GO, node_type>> sorted_map)
{
}

/////////////////////////////////////////////////////////////////////////////
///
/// \fn write_data
///
/// \brief Populate requests this module makes for output data
///
/// \param Scalar point data
/// \param Vector point data
/// \param Scalar cell data (double)
/// \param Scalar cell data (int)
/// \param Cell field data
///
/////////////////////////////////////////////////////////////////////////////
void FEA_Module_DANN::write_data(std::map<std::string, const double*>& point_data_scalars_double,
    std::map<std::string, const double*>& point_data_vectors_double,
    std::map<std::string, const double*>& cell_data_scalars_double,
    std::map<std::string, const int*>&    cell_data_scalars_int,
    std::map<std::string, std::pair<const double*, size_t>>& cell_data_fields_double)
{
    const size_t rk_level = simparam->dynamic_options.rk_num_bins - 1;

    for (const auto& field_name : simparam->output_options.output_fields) {
        switch (field_name) {
        case FIELD::state:
            // node "state"
            node_states.update_host();
            point_data_scalars_double["state"] = &node_states.host(rk_level, 0);
            break;


        default:
            break;
        } // end switch
    } // end if

}

/////////////////////////////////////////////////////////////////////////////
///
/// \fn collect_output
///
/// \brief Inactive in this context
///
/////////////////////////////////////////////////////////////////////////////
void FEA_Module_DANN::collect_output(Teuchos::RCP<Tpetra::Map<LO, GO, node_type>> global_reduce_map)
{
}

/////////////////////////////////////////////////////////////////////////////
///
/// \fn compute_output
///
/// \brief Inactive in this context
///
/////////////////////////////////////////////////////////////////////////////
void FEA_Module_DANN::compute_output()
{
}


/////////////////////////////////////////////////////////////////////////////
///
/// \fn comm_variables
///
/// \brief Communicate ghosts using the current optimization design data
///
/////////////////////////////////////////////////////////////////////////////
void FEA_Module_DANN::comm_variables(Teuchos::RCP<const MV> zp)
{

}

/////////////////////////////////////////////////////////////////////////////
///
/// \fn solve
///
/// \brief Solve function called by solver
///
/////////////////////////////////////////////////////////////////////////////
int FEA_Module_DANN::solve()
{
    dann_solve();

    return 0;
}

/////////////////////////////////////////////////////////////////////////////
///
/// \fn dann_solve
///
/// \brief DANN solver loop
///
/////////////////////////////////////////////////////////////////////////////
void FEA_Module_DANN::dann_solve()
{
    Dynamic_Options dynamic_options = simparam->dynamic_options;

    const int    num_dim  = simparam->num_dims;
    const size_t rk_level = dynamic_options.rk_num_bins - 1;

    const DCArrayKokkos<boundary_t> boundary = module_params->boundary;
    int batch_size = module_params->batch_size;

    int print_cycle = dynamic_options.print_cycle;

    graphics_time    = simparam->output_options.graphics_step;
    graphics_dt_ival = simparam->output_options.graphics_step;
    cycle_stop     = dynamic_options.cycle_stop;
    graphics_times = simparam->output_options.graphics_times;
    graphics_id    = simparam->output_options.graphics_id;

    if(myrank==0){
        std::cout << "DANN SOLVER CALLED " << std::endl;
    }
    distributed_weights->setAllToScalar(1);
    previous_node_states_distributed->putScalar(1);
    //comm to all here
    all_previous_node_states_distributed->doImport(*previous_node_states_distributed, *importer, Tpetra::INSERT);
    for(int istep = 0; istep < cycle_stop; istep++){
        distributed_weights->apply(*previous_node_states_distributed,*node_states_distributed);
        //comm to all here
        all_node_states_distributed->doImport(*node_states_distributed, *importer, Tpetra::INSERT);
        all_previous_node_states_distributed->assign(*all_node_states_distributed);
    }

    //output state vector
    node_states_distributed->describe(*fos, Teuchos::VERB_EXTREME);
    
} // end of DANN solve
