#pragma once
#include "Simulation_Parameters/FEA_Module/FEA_Module_Parameters.h"
#include "yaml-serializable.h"

struct DANN_Parameters 
    : FEA_Module_Parameters::Register<DANN_Parameters, FEA_MODULE_TYPE::DANN> {
    double damping_constant = 0.0000001;
    size_t batch_size = 1;
    size_t read_buffer_size = 20000;
    size_t num_input_nodes = 1;
    size_t num_output_nodes = 1;
    size_t num_training_data = 1;
    size_t num_testing_data = 1;
    std::string training_input_data_filename = "training_input_data.txt";
    std::string testing_input_data_filename = "testing_input_data.txt";
    std::string training_output_data_filename = "training_output_data.txt";
    std::string testing_output_data_filename = "testing_output_data.txt";

    DANN_Parameters() : FEA_Module_Parameters({
        FIELD::state
    }) { }

    void derive() {
        requires_conditions = false;
    }
};
IMPL_YAML_SERIALIZABLE_WITH_BASE(DANN_Parameters, FEA_Module_Parameters,
                                 batch_size, training_input_data_filename, training_output_data_filename,
                                 testing_input_data_filename, testing_output_data_filename, num_input_nodes,
                                 num_output_nodes, num_training_data, num_testing_data, read_buffer_size)