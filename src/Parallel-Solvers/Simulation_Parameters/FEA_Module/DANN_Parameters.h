#pragma once
#include "Simulation_Parameters/FEA_Module/FEA_Module_Parameters.h"
#include "yaml-serializable.h"

struct DANN_Parameters 
    : FEA_Module_Parameters::Register<DANN_Parameters, FEA_MODULE_TYPE::SGH> {
    double damping_constant = 0.0000001;

    DANN_Parameters() : FEA_Module_Parameters({
        FIELD::state
    }) { }
};
IMPL_YAML_SERIALIZABLE_WITH_BASE(DANN_Parameters, FEA_Module_Parameters)