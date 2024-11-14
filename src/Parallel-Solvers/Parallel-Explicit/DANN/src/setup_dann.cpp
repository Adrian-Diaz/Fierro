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
#include <Tpetra_CrsMatrix.hpp>
#include "Tpetra_Details_makeColMap.hpp"
#include "Tpetra_Import_Util2.hpp"

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

    init_assembly();

    //read in training and test data
    read_training_data();
    read_testing_data();

    
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

/* ----------------------------------------------------------------------
   Initialize global vectors and array maps needed for matrix assembly
------------------------------------------------------------------------- */
void FEA_Module_DANN::init_assembly(){
  const_host_elem_conn_array nodes_in_elem = global_nodes_in_elem_distributed->getLocalView<HostSpace> (Tpetra::Access::ReadOnly);
  CArrayKokkos<size_t, array_layout, device_type, memory_traits> Graph_Fill(nall_nodes, "nall_nodes");
  CArrayKokkos<size_t, array_layout, device_type, memory_traits> current_row_nodes_scanned;
  int current_row_n_nodes_scanned;
  int local_node_index, global_node_index, current_column_index;
  int max_stride = 0;
  size_t nodes_per_element;
  
  //allocate stride arrays
  CArrayKokkos <size_t, array_layout, device_type, memory_traits> Graph_Matrix_Strides_initial(nlocal_nodes, "Graph_Matrix_Strides_initial");
  Graph_Matrix_Strides = DCArrayKokkos<size_t, array_layout, device_type, memory_traits>(nlocal_nodes, "Graph_Matrix_Strides");

//   //allocate storage for the sparse conductivity matrix map used in the assembly process
//   Global_Conductivity_Matrix_Assembly_Map = CArrayKokkos<size_t, array_layout, device_type, memory_traits>(rnum_elem,
//                                          max_nodes_per_element,max_nodes_per_element, "Global_Conductivity_Matrix_Assembly_Map");

  //allocate array used to determine global node repeats in the sparse graph later
  CArrayKokkos <int, array_layout, device_type, memory_traits> node_indices_used(nall_nodes, "node_indices_used");

  /*allocate array that stores which column the node index occured on for the current row
    when removing repeats*/
  CArrayKokkos <size_t, array_layout, device_type, memory_traits> column_index(nall_nodes, "column_index");
  
  //initialize nlocal arrays
  for(int inode = 0; inode < nlocal_nodes; inode++){
    Graph_Matrix_Strides_initial(inode) = 0;
    Graph_Matrix_Strides(inode) = 0;
    Graph_Fill(inode) = 0;
  }

  //initialize nall arrays
  for(int inode = 0; inode < nall_nodes; inode++){
    node_indices_used(inode) = 0;
    column_index(inode) = 0;
  }
  
  //count upper bound of strides for Sparse Pattern Graph by allowing repeats due to connectivity
  if(num_dim == 2)
  for (int ielem = 0; ielem < rnum_elem; ielem++){
    element_select->choose_2Delem_type(Element_Types(ielem), elem2D);
    nodes_per_element = elem2D->num_nodes();
    for (int lnode = 0; lnode < nodes_per_element; lnode++){
      global_node_index = nodes_in_elem(ielem, lnode);
      if(map->isNodeGlobalElement(global_node_index)){
        local_node_index = map->getLocalElement(global_node_index);
        Graph_Matrix_Strides_initial(local_node_index) += nodes_per_element;
      }
    }
  }

  if(num_dim == 3)
  for (int ielem = 0; ielem < rnum_elem; ielem++){
    element_select->choose_3Delem_type(Element_Types(ielem), elem);
    nodes_per_element = elem->num_nodes();
    for (int lnode = 0; lnode < nodes_per_element; lnode++){
      global_node_index = nodes_in_elem(ielem, lnode);
      if(map->isNodeGlobalElement(global_node_index)){
        local_node_index = map->getLocalElement(global_node_index);
        Graph_Matrix_Strides_initial(local_node_index) += nodes_per_element;
      }
    }
  }
  
  //equate strides for later
  for(int inode = 0; inode < nlocal_nodes; inode++)
    Graph_Matrix_Strides(inode) = Graph_Matrix_Strides_initial(inode);
  
  //compute maximum stride
  for(int inode = 0; inode < nlocal_nodes; inode++)
    if(Graph_Matrix_Strides_initial(inode) > max_stride) max_stride = Graph_Matrix_Strides_initial(inode);
  
  //allocate array used in the repeat removal process
  current_row_nodes_scanned = CArrayKokkos<size_t, array_layout, device_type, memory_traits>(max_stride, "current_row_nodes_scanned");

  //allocate sparse graph with node repeats
  RaggedRightArrayKokkos<size_t, array_layout, device_type, memory_traits> Repeat_Graph_Matrix(Graph_Matrix_Strides_initial);
  RaggedRightArrayofVectorsKokkos<size_t, array_layout, device_type, memory_traits> Element_local_indices(Graph_Matrix_Strides_initial,num_dim);
  
  //Fill the initial Graph with repeats
  if(num_dim == 2)
  for (int ielem = 0; ielem < rnum_elem; ielem++){
    element_select->choose_2Delem_type(Element_Types(ielem), elem2D);
    nodes_per_element = elem2D->num_nodes();
    for (int lnode = 0; lnode < nodes_per_element; lnode++){
      global_node_index = nodes_in_elem(ielem, lnode);
      if(map->isNodeGlobalElement(global_node_index)){
        local_node_index = map->getLocalElement(global_node_index);
        for (int jnode = 0; jnode < nodes_per_element; jnode++){
          current_column_index = Graph_Fill(local_node_index)+jnode;
          Repeat_Graph_Matrix(local_node_index, current_column_index) = nodes_in_elem(ielem,jnode);

        //   //fill inverse map
        //   Element_local_indices(local_node_index,current_column_index,0) = ielem;
        //   Element_local_indices(local_node_index,current_column_index,1) = lnode;
        //   Element_local_indices(local_node_index,current_column_index,2) = jnode;

        //   //fill forward map
        //   Global_Conductivity_Matrix_Assembly_Map(ielem,lnode,jnode) = current_column_index;
        }
        Graph_Fill(local_node_index) += nodes_per_element;
      }
    }
  }
  
  if(num_dim == 3)
  for (int ielem = 0; ielem < rnum_elem; ielem++){
    element_select->choose_3Delem_type(Element_Types(ielem), elem);
    nodes_per_element = elem->num_nodes();
    for (int lnode = 0; lnode < nodes_per_element; lnode++){
      global_node_index = nodes_in_elem(ielem, lnode);
      if(map->isNodeGlobalElement(global_node_index)){
        local_node_index = map->getLocalElement(global_node_index);
        for (int jnode = 0; jnode < nodes_per_element; jnode++){
          current_column_index = Graph_Fill(local_node_index)+jnode;
          Repeat_Graph_Matrix(local_node_index, current_column_index) = nodes_in_elem(ielem,jnode);

        //   //fill inverse map
        //   Element_local_indices(local_node_index,current_column_index,0) = ielem;
        //   Element_local_indices(local_node_index,current_column_index,1) = lnode;
        //   Element_local_indices(local_node_index,current_column_index,2) = jnode;

        //   //fill forward map
        //   Global_Conductivity_Matrix_Assembly_Map(ielem,lnode,jnode) = current_column_index;
        }
        Graph_Fill(local_node_index) += nodes_per_element;
      }
    }
  }
  
  //debug statement
  //std::cout << "started run" << std::endl;
  //std::cout << "Graph Matrix Strides Repeat on task " << myrank << std::endl;
  //for (int inode = 0; inode < nlocal_nodes; inode++)
    //std::cout << Graph_Matrix_Strides(inode) << std::endl;
  
  //remove repeats from the inital graph setup
  int current_node, current_element_index, element_row_index, element_column_index, current_stride;
  for (int inode = 0; inode < nlocal_nodes; inode++){
    current_row_n_nodes_scanned = 0;
    for (int istride = 0; istride < Graph_Matrix_Strides(inode); istride++){
      //convert global index in graph to its local index for the flagging array
      current_node = all_node_map->getLocalElement(Repeat_Graph_Matrix(inode,istride));
      //debug
      //if(current_node==-1)
      //std::cout << "Graph Matrix node access on task " << myrank << std::endl;
      //std::cout << Repeat_Graph_Matrix(inode,istride) << std::endl;
      if(node_indices_used(current_node)){
        //set global assembly map index to the location in the graph matrix where this global node was first found
        // current_element_index = Element_local_indices(inode,istride,0);
        // element_row_index = Element_local_indices(inode,istride,1);
        // element_column_index = Element_local_indices(inode,istride,2);
        // Global_Conductivity_Matrix_Assembly_Map(current_element_index,element_row_index, element_column_index) 
        //     = column_index(current_node);   

        
        //swap current node with the end of the current row and shorten the stride of the row
        //first swap information about the inverse and forward maps

        current_stride = Graph_Matrix_Strides(inode);
        if(istride!=current_stride-1){
        // Element_local_indices(inode,istride,0) = Element_local_indices(inode,current_stride-1,0);
        // Element_local_indices(inode,istride,1) = Element_local_indices(inode,current_stride-1,1);
        // Element_local_indices(inode,istride,2) = Element_local_indices(inode,current_stride-1,2);
        // current_element_index = Element_local_indices(inode,istride,0);
        // element_row_index = Element_local_indices(inode,istride,1);
        // element_column_index = Element_local_indices(inode,istride,2);

        // Global_Conductivity_Matrix_Assembly_Map(current_element_index,element_row_index, element_column_index) 
        //     = istride;

        //now that the element map information has been copied, copy the global node index and delete the last index

        Repeat_Graph_Matrix(inode,istride) = Repeat_Graph_Matrix(inode,current_stride-1);
        }
        istride--;
        Graph_Matrix_Strides(inode)--;
      }
      else{
        /*this node hasn't shown up in the row before; add it to the list of nodes
          that have been scanned uniquely. Use this list to reset the flag array
          afterwards without having to loop over all the nodes in the system*/
        node_indices_used(current_node) = 1;
        column_index(current_node) = istride;
        current_row_nodes_scanned(current_row_n_nodes_scanned) = current_node;
        current_row_n_nodes_scanned++;
      }
    }
    //reset nodes used list for the next row of the sparse list
    for(int node_reset = 0; node_reset < current_row_n_nodes_scanned; node_reset++)
      node_indices_used(current_row_nodes_scanned(node_reset)) = 0;

  }

  //copy reduced content to non_repeat storage
  Graph_Matrix = RaggedRightArrayKokkos<GO, array_layout, device_type, memory_traits>(Graph_Matrix_Strides);
  for(int inode = 0; inode < nlocal_nodes; inode++)
    for(int istride = 0; istride < Graph_Matrix_Strides(inode); istride++)
      Graph_Matrix(inode,istride) = Repeat_Graph_Matrix(inode,istride);

  //deallocate repeat matrix
  
  /*At this stage the sparse graph should have unique global indices on each row.
    The constructed Assembly map (to the global sparse matrix)
    is used to loop over each element's local conductivity matrix in the assembly process.*/
  
  Gradient_Matrix_Strides = Graph_Matrix_Strides;
  Weight_Matrix = RaggedRightArrayKokkos<real_t, Kokkos::LayoutRight, device_type, memory_traits, array_layout>(Graph_Matrix_Strides);
  Gradient_Matrix = RaggedRightArrayKokkos<real_t, Kokkos::LayoutRight, device_type, memory_traits, array_layout>(Gradient_Matrix_Strides);

  //construct distributed conductivity matrix and force vector from local kokkos data
  
  //build column map for the global conductivity matrix
  Teuchos::RCP<const Tpetra::Map<LO,GO,node_type> > colmap;
  const Teuchos::RCP<const Tpetra::Map<LO,GO,node_type> > dommap = map;

  Tpetra::Details::makeColMap<LO,GO,node_type>(colmap,dommap,Graph_Matrix.get_kokkos_view(), nullptr);

  size_t nnz = Graph_Matrix.size();

  //debug print
  //std::cout << "DOF GRAPH SIZE ON RANK " << myrank << " IS " << nnz << std::endl;
  
  //local indices in the graph using the constructed column map
  CArrayKokkos<LO, array_layout, device_type, memory_traits> weight_local_indices(nnz, "weight_local_indices");
  
  //row offsets with compatible template arguments
    Kokkos::View<size_t *,array_layout, device_type, memory_traits> row_offsets = Graph_Matrix.start_index_;
    row_pointers row_offsets_pass("row_offsets", nlocal_nodes + 1);
    for(int ipass = 0; ipass < nlocal_nodes + 1; ipass++){
      row_offsets_pass(ipass) = row_offsets(ipass);
    }

  size_t entrycount = 0;
  for(int irow = 0; irow < nlocal_nodes; irow++){
    for(int istride = 0; istride < Graph_Matrix_Strides(irow); istride++){
      weight_local_indices(entrycount) = colmap->getLocalElement(Graph_Matrix(irow,istride));
      entrycount++;
    }
  }
  
  //sort values and indices
  Tpetra::Import_Util::sortCrsEntries<row_pointers, indices_array, values_array>(row_offsets_pass, weight_local_indices.get_kokkos_view(), Weight_Matrix.get_kokkos_view());

  distributed_weights = Teuchos::rcp(new MAT(map, colmap, row_offsets_pass, weight_local_indices.get_kokkos_view(), Weight_Matrix.get_kokkos_view()));
  distributed_weights->fillComplete();
  
}

/* ----------------------------------------------------------------------
   Read in model training data
------------------------------------------------------------------------- */
void FEA_Module_DANN::read_training_data()
{

    
} // end of read_training_data

/* ----------------------------------------------------------------------
   Read in model testing data
------------------------------------------------------------------------- */
void FEA_Module_DANN::read_testing_data()
{

    
} // end of read_testing_data

