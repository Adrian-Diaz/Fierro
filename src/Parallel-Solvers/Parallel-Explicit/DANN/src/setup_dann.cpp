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

#define MAX_WORD 30
#define BC_EPSILON 1.0e-6
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

    init_boundaries();
    init_assembly();
    generate_bcs();
    
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
      if(node_indices_used(current_node)||inode==current_node){
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
  weight_indices = Kokkos::DualView<GO*, Kokkos::LayoutLeft, device_type, memory_traits>("weight_indices", nnz);
  //set global indices for each weight in 1D span by looping over contents of each row of matrix

  global_nnz = 0;
  //global nonzero count
  MPI_Allreduce(&nnz, &global_nnz, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

  //partition map of weight indices
  weight_map = Teuchos::rcp(new Tpetra::Map<LO, GO, node_type>(global_nnz, nnz, 0, comm));

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

  //initialize weight matrix to random values
  if(simparam->dann_training_on){
    //randomize
    // random generator
    std::default_random_engine gen;
    std::uniform_real_distribution<double> distribution(simparam->optimization_options.min_weight_value, simparam->optimization_options.max_weight_value);
 
    for(int irow = 0; irow < nlocal_nodes; irow++){
      for(int istride = 0; istride < Graph_Matrix_Strides(irow); istride++){
        Weight_Matrix(irow,istride) = distribution(gen);
      }
    }
  }
  else{
    //randomize
    // random generator
    std::default_random_engine gen;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
 
    for(int irow = 0; irow < nlocal_nodes; irow++){
      for(int istride = 0; istride < Graph_Matrix_Strides(irow); istride++){
        Weight_Matrix(irow,istride) = distribution(gen);
      }
    }
    //distributed_weights->setAllToScalar(1.0);
  }

  
}

/* ----------------------------------------------------------------------
   Read in model training data
------------------------------------------------------------------------- */
void FEA_Module_DANN::read_training_data(size_t current_batch_size, bool last_batch)
{
  char ch;
  std::string skip_line, read_line, substring;
  std::stringstream line_parse;

  size_t num_input_nodes = module_params->num_input_nodes;
  bool sparse_categories = module_params->sparse_categories;
  size_t num_output_nodes = (sparse_categories) ? 1 : module_params->num_output_nodes;
  size_t num_training_data = module_params->num_training_data;
  std::string training_input_data_filename = module_params->training_input_data_filename;
  std::string training_output_data_filename = module_params->training_output_data_filename;
  size_t buffer_size = module_params->read_buffer_size;
  size_t batch_size = current_batch_size;
  int local_node_index;
  size_t batch_id, patch_id;
  int buffer_loop, buffer_iteration, buffer_iterations, scan_loop;
  size_t num_local_bc_nodes;
  long long int num_bc_input_nodes;
  size_t read_index_start, node_rid, elem_gid;
  size_t strain_count;
  CArray<GO> Surface_Nodes;
  std::set<long long int> bc_node_set;

  GO     node_gid;
  real_t dof_value;

  CArrayKokkos<char, array_layout, HostSpace, memory_traits> read_buffer;

  if(num_boundary_conditions){
    //get unique node set for this condition since storage is in patches
    for(int ipatch=0; ipatch < NBoundary_Condition_Patches(0); ipatch++){
      patch_id = Boundary_Condition_Patches(0, ipatch);
      Surface_Nodes = Boundary_Patches(patch_id).node_set;
      for(int inode=0; inode < Surface_Nodes.size(); inode++){
        if(map->isNodeGlobalElement(Surface_Nodes(inode))){
          bc_node_set.insert(Surface_Nodes(inode));
        }
      }
    }

    num_local_bc_nodes = bc_node_set.size();

    //TODO: use subcommunicator

    //find how many ranks have bc nodes
    int local_have_bc_nodes = (NBoundary_Condition_Patches(0)) ? 1 : 0;
    int global_have_bc_nodes = 0;
    
    MPI_Allreduce(&local_have_bc_nodes, &global_have_bc_nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    //find total unique bc nodes
    long long int num_global_bc_nodes;
    MPI_Allreduce(&num_local_bc_nodes, &num_global_bc_nodes, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

    if(num_global_bc_nodes < num_input_nodes){
      if(myrank==0){
        std::cout << "Not enough BC nodes to apply training data" << std::endl;
      }
      Solver_Pointer_->exit_solver(0);
    }

    //compute avg bc nodes being assigned an input per rank
    long long int avg_num_input_nodes = num_input_nodes/global_have_bc_nodes;
    long long int rem_num_input_nodes = num_input_nodes%global_have_bc_nodes;
    long long int *inputs_per_rank, *bc_nodes_per_rank;
    //gather bc nodes per rank

    if(myrank==0){
      inputs_per_rank = new long long int[nranks];
      bc_nodes_per_rank = new long long int[nranks];
    }
    
    MPI_Gather(&num_local_bc_nodes,1,MPI_LONG_LONG_INT,inputs_per_rank,1,MPI_LONG_LONG_INT,0,MPI_COMM_WORLD);

    //assign upper bound assignment to each rank in a scatter
    if(myrank==0){
      //copy
      for(int irank=0; irank < nranks; irank++){
        bc_nodes_per_rank[irank] = inputs_per_rank[irank];
      }
      long long int remainder = num_input_nodes;
      for(int irank=0; irank < nranks; irank++){
        if(inputs_per_rank[irank] < avg_num_input_nodes){
          remainder -= inputs_per_rank[irank];
        }
        else if(inputs_per_rank[irank] >= avg_num_input_nodes){
          inputs_per_rank[irank] = avg_num_input_nodes;
          remainder -= inputs_per_rank[irank];
        }
      }
      long long int remainder_contrib;
      //assign remainder wherever it fits
      for(int irank=0; irank < nranks; irank++){
        if(bc_nodes_per_rank[irank] > avg_num_input_nodes){
          remainder_contrib = bc_nodes_per_rank[irank] - inputs_per_rank[irank];
          if(remainder_contrib > remainder) remainder_contrib = remainder;
          inputs_per_rank[irank] += remainder_contrib;
          remainder -= remainder_contrib;
        }
        if(remainder==0) break;
      }
    }

    MPI_Scatter(inputs_per_rank,1,MPI_LONG_LONG_INT,&num_bc_input_nodes,1,MPI_LONG_LONG_INT,0,MPI_COMM_WORLD);
    std::cout << "BC INPUT NODES " << num_bc_input_nodes << std::endl;

    if(myrank==0){
      delete[] inputs_per_rank;
      delete[] bc_nodes_per_rank;
    }
  }

  // open the input and output training data files
  if (myrank == 0&&first_training_batch_read)
  {
      std::cout << " INPUT DATA DIM IS " << num_input_nodes << std::endl;
      input_training_file = new std::ifstream();
      input_training_file->open(training_input_data_filename);

      std::cout << " OUTPUT DATA DIM IS " << num_output_nodes << std::endl;
      output_training_file = new std::ifstream();
      output_training_file->open(training_output_data_filename);
      first_training_batch_read = false;
  } // end if(myrank==0)

  // scope ensures view is destroyed for now to avoid calling a device view with an active host view later
  {
      host_vec_array node_states = previous_node_states_distributed->getLocalView<HostSpace>(Tpetra::Access::ReadWrite);
      /*only process 0 reads in data from the input file
      stores data in a buffer and communicates once the buffer cap is reached
      or the data ends*/

      // allocate read buffer
      read_buffer = CArrayKokkos<char, array_layout, HostSpace, memory_traits>(buffer_size, num_input_nodes, MAX_WORD);

      buffer_iterations = batch_size / buffer_size;
      if (batch_size % buffer_size != 0)
      {
          buffer_iterations++;
      }

      size_t read_limit = batch_size; //alter to choose remainder between current batch read and available training points later

      // read data
      read_index_start = 0;
      for (buffer_iteration = 0; buffer_iteration < buffer_iterations; buffer_iteration++)
      {
          // pack buffer on rank 0
          if (myrank == 0 && buffer_iteration < buffer_iterations - 1)
          {
              for (buffer_loop = 0; buffer_loop < buffer_size; buffer_loop++)
              {
                  getline(*input_training_file, read_line);
                  line_parse.clear();
                  line_parse.str(read_line);

                  for (int iword = 0; iword < num_input_nodes; iword++)
                  {
                      // read portions of the line into the substring variable
                      line_parse >> substring;
                      // debug print
                      // std::cout<<" "<< substring <<std::endl;
                      // assign the substring variable as a word of the read buffer
                      strcpy(&read_buffer(buffer_loop, iword, 0), substring.c_str());
                  }
              }
          }
          else if (myrank == 0)
          {
              buffer_loop = 0;
              while (buffer_iteration * buffer_size + buffer_loop < read_limit) {
                  getline(*input_training_file, read_line);
                  line_parse.clear();
                  line_parse.str(read_line);
                  for (int iword = 0; iword < num_input_nodes; iword++)
                  {
                      // read portions of the line into the substring variable
                      line_parse >> substring;
                      // debug print
                      // std::cout<<" "<< substring <<std::endl;
                      // assign the substring variable as a word of the read buffer
                      strcpy(&read_buffer(buffer_loop, iword, 0), substring.c_str());
                  }
                  buffer_loop++;
              }
          }

          // broadcast buffer to all ranks; each rank will determine which nodes in the buffer belong
          MPI_Bcast(read_buffer.pointer(), buffer_size * num_input_nodes * MAX_WORD, MPI_CHAR, 0, world);
          // broadcast how many nodes were read into this buffer iteration
          MPI_Bcast(&buffer_loop, 1, MPI_INT, 0, world);

          // debug_print
          // std::cout << "NODE BUFFER LOOP IS: " << buffer_loop << std::endl;
          // for(int iprint=0; iprint < buffer_loop; iprint++)
          // std::cout<<"buffer packing: " << std::string(&read_buffer(iprint,0,0)) << std::endl;
          // return;

          // determine which data to store in the swage mesh members (the local node data)
          // loop through read buffer
          
          for (scan_loop = 0; scan_loop < buffer_loop; scan_loop++)
          {
              // set global node id (ensight specific order)
              batch_id = read_index_start + scan_loop;
              // let map decide if this node id belongs locally; if yes store data
              if(num_boundary_conditions){
                auto it     = bc_node_set.begin();
                // while (it != bc_node_set.end()) {
                //     node_gid = *it;
                //     it++;
                // }
                
                for (int inode = 0; inode < num_bc_input_nodes; inode++)
                {
                  node_gid = *it; //assign to the first num_input nodes of the first BC set
                  it++;
                  // set local node index in this mpi rank
                  node_rid = map->getLocalElement(node_gid);
                  dof_value = atof(&read_buffer(scan_loop, inode, 0));
                  node_states(node_rid, batch_id) = dof_value;
                  
                  //std::cout << "BC INPUT NODES " << node_gid << " VALUE " << node_states(node_rid, batch_id) << std::endl;
                }
              }
              else{
                for (int inode = 0; inode < num_input_nodes; inode++)
                {
                  node_gid = inode; //we assume input nodes will be the first 0:num_input_nodes-1 global nodes
                  if (map->isNodeGlobalElement(node_gid))
                  {
                      // set local node index in this mpi rank
                      node_rid = map->getLocalElement(node_gid);
                      dof_value = atof(&read_buffer(scan_loop, node_gid, 0));
                      node_states(node_rid, batch_id) = dof_value;
                  
                      //std::cout << "BC INPUT NODES " << node_gid << " VALUE " << node_states(node_rid, batch_id) << std::endl;
                  }
                }
              }
          }
          read_index_start += buffer_size;
      }
  } // end of input data readin

  // allocate read buffer
  read_buffer = CArrayKokkos<char, array_layout, HostSpace, memory_traits>(buffer_size, num_output_nodes, MAX_WORD);

  buffer_iterations = batch_size / buffer_size;
  if (batch_size % buffer_size != 0)
  {
      buffer_iterations++;
  }

  size_t read_limit = batch_size; //alter to choose remainder between current batch read and available training points later

  // read data
  read_index_start = 0;
  for (buffer_iteration = 0; buffer_iteration < buffer_iterations; buffer_iteration++)
  {
      // pack buffer on rank 0
      if (myrank == 0 && buffer_iteration < buffer_iterations - 1)
      {
          for (buffer_loop = 0; buffer_loop < buffer_size; buffer_loop++)
          {
              getline(*output_training_file, read_line);
              line_parse.clear();
              line_parse.str(read_line);

              for (int iword = 0; iword < num_output_nodes; iword++)
              {
                  // read portions of the line into the substring variable
                  line_parse >> substring;
                  // debug print
                  // std::cout<<" "<< substring <<std::endl;
                  // assign the substring variable as a word of the read buffer
                  strcpy(&read_buffer(buffer_loop, iword, 0), substring.c_str());
              }
          }
      }
      else if (myrank == 0)
      {
          buffer_loop = 0;
          while (buffer_iteration * buffer_size + buffer_loop < read_limit) {
              getline(*output_training_file, read_line);
              line_parse.clear();
              line_parse.str(read_line);
              for (int iword = 0; iword < num_output_nodes; iword++)
              {
                  // read portions of the line into the substring variable
                  line_parse >> substring;
                  // debug print
                  // std::cout<<" "<< substring <<std::endl;
                  // assign the substring variable as a word of the read buffer
                  strcpy(&read_buffer(buffer_loop, iword, 0), substring.c_str());
              }
              buffer_loop++;
          }
      }

      // broadcast buffer to all ranks; each rank will determine which nodes in the buffer belong
      MPI_Bcast(read_buffer.pointer(), buffer_size * num_output_nodes * MAX_WORD, MPI_CHAR, 0, world);
      // broadcast how many nodes were read into this buffer iteration
      MPI_Bcast(&buffer_loop, 1, MPI_INT, 0, world);

      // debug_print
      // std::cout << "NODE BUFFER LOOP IS: " << buffer_loop << std::endl;
      // for(int iprint=0; iprint < buffer_loop; iprint++)
      // std::cout<<"buffer packing: " << std::string(&read_buffer(iprint,0,0)) << std::endl;
      // return;

      // determine which data to store in the swage mesh members (the local node data)
      // loop through read buffer
      for (scan_loop = 0; scan_loop < buffer_loop; scan_loop++)
      {
          // set global node id (ensight specific order)
          batch_id = read_index_start + scan_loop;
          // let map decide if this node id belongs locally; if yes store data
          for (int inode = 0; inode < num_output_nodes; inode++)
          {
            dof_value = atof(&read_buffer(scan_loop, inode, 0));
            //we assume output nodes will be the last num_node-1:num_input_nodes-1-num_output_nodes global nodes
            //node_gid = (sparse_categories) num_nodes-1-dof_value : num_nodes-1-inode;
            Output_Training_Data_Batch(inode,batch_id) = dof_value;
          }
      }
      read_index_start += buffer_size;
  }
  // end of output data readin

  //debug print of output
  // if(myrank==0)
  // for(int ibatch = 0; ibatch < batch_size; ibatch++){
  //   std::cout << "Output for batch index " << ibatch << " is " << Output_Training_Data_Batch(0,ibatch) << std::endl;
  // }

  //close files if the last batch has been read in
  if (myrank == 0&&last_batch)
  {
      input_training_file->close();
      output_training_file->close();
  }
    
} // end of read_training_data

/* ----------------------------------------------------------------------
   Read in model testing data
------------------------------------------------------------------------- */
void FEA_Module_DANN::read_testing_data(size_t current_batch_size, bool last_batch)
{

    
} // end of read_testing_data

/////////////////////////////////////////////////////////////////////////////
///
/// \fn init_boundaries
///
/// \brief Initialize sets of element boundary surfaces and arrays for input conditions
///
/////////////////////////////////////////////////////////////////////////////
void FEA_Module_DANN::init_boundaries()
{
    max_boundary_sets = module_params->boundary_conditions.size();
    int num_dim = simparam->num_dims;

    // set the number of boundary sets
    if (myrank == 0) {
        std::cout << "building boundary sets " << std::endl;
    }

    // initialize to 1 since there must be at least 1 boundary set anyway; read in may occure later
    if (max_boundary_sets == 0) {
        max_boundary_sets = 1;
    }
    // std::cout << "NUM BOUNDARY CONDITIONS ON RANK " << myrank << " FOR INIT " << num_boundary_conditions <<std::endl;
    init_boundary_sets(max_boundary_sets);

    // allocate nodal data
    Node_DOF_Boundary_Condition_Type = CArrayKokkos<int, array_layout, device_type, memory_traits>(nall_nodes * num_dim, "Node_DOF_Boundary_Condition_Type");

    // initialize
    for (int init = 0; init < nall_nodes * num_dim; init++) {
        Node_DOF_Boundary_Condition_Type(init) = NONE;
    }

    Number_DOF_BCS = 0;
}

/* ----------------------------------------------------------------------
   Assign sets of element boundary surfaces corresponding to user BCs
------------------------------------------------------------------------- */

void FEA_Module_DANN::generate_bcs(){
  int num_dim = simparam->num_dims;
  int num_bcs;
  int bc_tag;
  real_t value;
  real_t surface_limits[4];

  for (auto bc : module_params->boundary_conditions) {
    switch (bc.surface.type) {
      case BOUNDARY_TYPE::x_plane:
        bc_tag = 0;
        break;
      case BOUNDARY_TYPE::y_plane:
        bc_tag = 1;
        break;
      case BOUNDARY_TYPE::z_plane:
        bc_tag = 2;
        break;
      default:
        throw std::runtime_error("Invalid surface type: " + to_string(bc.surface.type));
    }
    value = bc.surface.plane_position * simparam->get_unit_scaling();

    //determine if the surface has finite limits
    if(num_boundary_conditions + 1>max_boundary_sets) grow_boundary_sets(num_boundary_conditions+1);
    //tag_boundaries(bc_tag, value, num_boundary_conditions, surface_limits);
    if(bc.surface.use_limits){
      surface_limits[0] = bc.surface.surface_limits_sl;
      surface_limits[1] = bc.surface.surface_limits_su;
      surface_limits[2] = bc.surface.surface_limits_tl;
      surface_limits[3] = bc.surface.surface_limits_tu;
      tag_boundaries(bc_tag, value, num_boundary_conditions, surface_limits);
    }
    else{
      tag_boundaries(bc_tag, value, num_boundary_conditions);
    }
    *fos << "tagging " << bc_tag << " at " << value <<  std::endl;
    
    *fos << "tagged a set " << std::endl;
    std::cout << "number of bdy patches in this bc set = " << NBoundary_Condition_Patches(num_boundary_conditions) << std::endl;
    *fos << std::endl;
    num_boundary_conditions++;
  }
} // end generate_bcs

/* ----------------------------------------------------------------------
   find which boundary patches correspond to the given BC.
   bc_tag = 0 xplane, 1 yplane, 2 zplane, 3 cylinder, 4 is shell
   val = plane value, cylinder radius, shell radius
------------------------------------------------------------------------- */

void FEA_Module_DANN::tag_boundaries(int bc_tag, real_t val, int bdy_set, real_t* patch_limits)
{
    int is_on_set;
    /*
    if (bdy_set == num_bdy_sets_){
      std::cout << " ERROR: number of boundary sets must be increased by "
        << bdy_set-num_bdy_sets_+1 << std::endl;
      exit(0);
    }
    */

    // test patch limits for feasibility
    if (patch_limits != NULL)
    {
        // test for upper bounds being greater than lower bounds
        if (patch_limits[1] <= patch_limits[0])
        {
            std::cout << " Warning: patch limits for boundary condition are infeasible " << patch_limits[0] << " and " << patch_limits[1] << std::endl;
        }
        if (patch_limits[3] <= patch_limits[2])
        {
            std::cout << " Warning: patch limits for boundary condition are infeasible " << patch_limits[2] << " and " << patch_limits[3] << std::endl;
        }
    }

    // save the boundary vertices to this set that are on the plane
    int counter = 0;
    for (int iboundary_patch = 0; iboundary_patch < nboundary_patches; iboundary_patch++)
    {
        // check to see if this patch is on the specified plane
        is_on_set = check_boundary(Boundary_Patches(iboundary_patch), bc_tag, val, patch_limits); // no=0, yes=1

        if (is_on_set == 1)
        {
            Boundary_Condition_Patches(bdy_set, counter) = iboundary_patch;
            counter++;
        }
    } // end for bdy_patch

    // save the number of bdy patches in the set
    NBoundary_Condition_Patches(bdy_set) = counter;

    *fos << " tagged boundary patches " << std::endl;
}

/* ----------------------------------------------------------------------
   routine for checking to see if a patch is on a boundary set
   bc_tag = 0 xplane, 1 yplane, 3 zplane, 4 cylinder, 5 is shell
   val = plane value, radius, radius
------------------------------------------------------------------------- */

int FEA_Module_DANN::check_boundary(Node_Combination& Patch_Nodes, int bc_tag, real_t val, real_t* patch_limits)
{
    int is_on_set = 1;
    const_host_vec_array all_node_coords = all_node_coords_distributed->getLocalView<HostSpace>(Tpetra::Access::ReadOnly);

    // Nodes on the Patch
    auto   node_list = Patch_Nodes.node_set;
    int    num_dim   = simparam->num_dims;
    size_t nnodes    = node_list.size();
    size_t node_rid;
    real_t node_coord[num_dim];
    int    dim_other1, dim_other2;
    CArrayKokkos<size_t, array_layout, device_type, memory_traits> node_on_flags(nnodes, "node_on_flags");

    // initialize
    for (int inode = 0; inode < nnodes; inode++)
    {
        node_on_flags(inode) = 0;
    }

    if (bc_tag == 0)
    {
        dim_other1 = 1;
        dim_other2 = 2;
    }
    else if (bc_tag == 1)
    {
        dim_other1 = 0;
        dim_other2 = 2;
    }
    else if (bc_tag == 2)
    {
        dim_other1 = 0;
        dim_other2 = 1;
    }

    // test for planes
    if (bc_tag < 3)
    {
        for (int inode = 0; inode < nnodes; inode++)
        {
            node_rid = all_node_map->getLocalElement(node_list(inode));
            for (int init = 0; init < num_dim; init++)
            {
                node_coord[init] = all_node_coords(node_rid, init);
            }
            if (fabs(node_coord[bc_tag] - val) <= BC_EPSILON)
            {
                node_on_flags(inode) = 1;

                // test if within patch segment if user specified
                if (patch_limits != NULL)
                {
                    if (node_coord[dim_other1] - patch_limits[0] <= -BC_EPSILON)
                    {
                        node_on_flags(inode) = 0;
                    }
                    if (node_coord[dim_other1] - patch_limits[1] >= BC_EPSILON)
                    {
                        node_on_flags(inode) = 0;
                    }
                    if (node_coord[dim_other2] - patch_limits[2] <= -BC_EPSILON)
                    {
                        node_on_flags(inode) = 0;
                    }
                    if (node_coord[dim_other2] - patch_limits[3] >= BC_EPSILON)
                    {
                        node_on_flags(inode) = 0;
                    }
                }
            }
            // debug print of node id and node coord
            // std::cout << "node coords on task " << myrank << " for node " << node_rid << std::endl;
            // std::cout << "coord " <<node_coord << " flag " << node_on_flags(inode) << " bc_tag " << bc_tag << std::endl;
        }
    }

    /*
    // cylinderical shell where radius = sqrt(x^2 + y^2)
    else if (this_bc_tag == 3){

        real_t R = sqrt(these_patch_coords[0]*these_patch_coords[0] +
                        these_patch_coords[1]*these_patch_coords[1]);

        if ( fabs(R - val) <= 1.0e-8 ) is_on_bdy = 1;


    }// end if on type

    // spherical shell where radius = sqrt(x^2 + y^2 + z^2)
    else if (this_bc_tag == 4){

        real_t R = sqrt(these_patch_coords[0]*these_patch_coords[0] +
                        these_patch_coords[1]*these_patch_coords[1] +
                        these_patch_coords[2]*these_patch_coords[2]);

        if ( fabs(R - val) <= 1.0e-8 ) is_on_bdy = 1;

    } // end if on type
    */
    // check if all nodes lie on the boundary set
    for (int inode = 0; inode < nnodes; inode++)
    {
        if (!node_on_flags(inode))
        {
            is_on_set = 0;
        }
    }

    // debug print of return flag
    // std::cout << "patch flag on task " << myrank << " is " << is_on_set << std::endl;
    return is_on_set;
} // end method to check bdy

/////////////////////////////////////////////////////////////////////////////
///
/// \fn grow_boundary_sets
///
/// \brief Grow boundary conditions sets of element boundary surfaces
///
/// \param Number of boundary sets
///
/////////////////////////////////////////////////////////////////////////////
void FEA_Module_DANN::grow_boundary_sets(int num_sets)
{
    int num_dim = simparam->num_dims;

    if (num_sets == 0) {
        std::cout << " Warning: number of boundary conditions being set to 0";
        return;
    }

    // std::cout << " DEBUG PRINT "<<num_sets << " " << nboundary_patches << std::endl;
    if (num_sets > max_boundary_sets) {
        // temporary storage for previous data
        CArrayKokkos<int, array_layout, HostSpace, memory_traits> Temp_Boundary_Condition_Type_List     = Boundary_Condition_Type_List;
        CArrayKokkos<size_t, array_layout, device_type, memory_traits> Temp_NBoundary_Condition_Patches = NBoundary_Condition_Patches;
        CArrayKokkos<size_t, array_layout, device_type, memory_traits> Temp_Boundary_Condition_Patches  = Boundary_Condition_Patches;

        max_boundary_sets = num_sets + 5; // 5 is an arbitrary buffer
        Boundary_Condition_Type_List = CArrayKokkos<int, array_layout, HostSpace, memory_traits>(max_boundary_sets, "Boundary_Condition_Type_List");
        NBoundary_Condition_Patches  = CArrayKokkos<size_t, array_layout, device_type, memory_traits>(max_boundary_sets, "NBoundary_Condition_Patches");
        // std::cout << "NBOUNDARY PATCHES ON RANK " << myrank << " FOR GROW " << nboundary_patches <<std::endl;
        Boundary_Condition_Patches = CArrayKokkos<size_t, array_layout, device_type, memory_traits>(max_boundary_sets, nboundary_patches, "Boundary_Condition_Patches");

        // copy previous data back over
#ifdef DEBUG
        std::cout << "NUM BOUNDARY CONDITIONS ON RANK " << myrank << " FOR COPY " << max_boundary_sets << std::endl;
#endif
        for (int iset = 0; iset < num_boundary_conditions; iset++) {
            Boundary_Condition_Type_List(iset) = Temp_Boundary_Condition_Type_List(iset);
            NBoundary_Condition_Patches(iset)  = Temp_NBoundary_Condition_Patches(iset);
            for (int ipatch = 0; ipatch < nboundary_patches; ipatch++) {
                Boundary_Condition_Patches(iset, ipatch) = Temp_Boundary_Condition_Patches(iset, ipatch);
            }
        }

        // initialize data
        for (int iset = num_boundary_conditions; iset < max_boundary_sets; iset++) {
            NBoundary_Condition_Patches(iset) = 0;
        }

        // initialize
        for (int ibdy = num_boundary_conditions; ibdy < max_boundary_sets; ibdy++) {
            Boundary_Condition_Type_List(ibdy) = NONE;
        }
    }
}

/* ----------------------------------------------------------------------
   initialize storage for element boundary surfaces corresponding to user BCs
------------------------------------------------------------------------- */

void FEA_Module_DANN::init_boundary_sets (int num_sets){

  if(num_sets == 0){
    std::cout << " Warning: number of boundary conditions = 0";
    return;
  }
  //initialize maximum
  max_boundary_sets = num_sets;
  //std::cout << " DEBUG PRINT "<<num_sets << " " << nboundary_patches << std::endl;
  Boundary_Condition_Type_List = CArrayKokkos<int, array_layout, HostSpace, memory_traits>(num_sets, "Boundary_Condition_Type_List");
  NBoundary_Condition_Patches = CArrayKokkos<size_t, array_layout, device_type, memory_traits>(num_sets, "NBoundary_Condition_Patches");
  //std::cout << "NBOUNDARY PATCHES ON RANK " << myrank << " FOR INIT IS " << nboundary_patches <<std::endl;
  Boundary_Condition_Patches = CArrayKokkos<size_t, array_layout, device_type, memory_traits>(num_sets, nboundary_patches, "Boundary_Condition_Patches");

  //initialize data
  for(int iset = 0; iset < num_sets; iset++) NBoundary_Condition_Patches(iset) = 0;

   //initialize
  for(int ibdy=0; ibdy < num_sets; ibdy++) Boundary_Condition_Type_List(ibdy) = NONE;
}
