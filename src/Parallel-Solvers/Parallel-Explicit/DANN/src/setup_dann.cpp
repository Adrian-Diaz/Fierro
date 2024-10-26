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
#include "matar.h"
#include "state.h"
#include "FEA_Module_DANN.h"
#include "Simulation_Parameters/Simulation_Parameters_Explicit.h"
#include "Simulation_Parameters/FEA_Module/DANN_Parameters.h"
#include "Explicit_Solver.h"

// #define DEBUG

/////////////////////////////////////////////////////////////////////////////
///
/// \fn setup
///
/// \brief Setup DANN solver data
///
/////////////////////////////////////////////////////////////////////////////
void FEA_Module_DANN::setup()
{
    const size_t rk_level         = simparam->dynamic_options.rk_num_bins - 1;
    const size_t num_fills        = simparam->regions.size();
    const size_t rk_num_bins      = simparam->dynamic_options.rk_num_bins;
    const int    num_dim          = simparam->num_dims;
    bool topology_optimization_on = simparam->topology_optimization_on;
    bool shape_optimization_on    = simparam->shape_optimization_on;
    
} // end of setup

/////////////////////////////////////////////////////////////////////////////
///
/// \fn dann_interface_setup
///
/// \brief Interfaces read in data with the DANN solver data; currently a hack to streamline
///
/// \param State data for the nodes
/// \param State data for the elements
/// \param State data for the corners
///
/////////////////////////////////////////////////////////////////////////////
void FEA_Module_DANN::dann_interface_setup(node_t& node, elem_t& elem, corner_t& corner)
{
    const size_t num_dim     = simparam->num_dims;
    const size_t rk_num_bins = simparam->dynamic_options.rk_num_bins;


}

