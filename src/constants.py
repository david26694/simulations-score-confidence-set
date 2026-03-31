from src.confidence_sets import (
    DMLConfidenceSetCalculator,
    RobustConfidenceSetCalculator,
    ARConfidenceSetCalculator,
)
from src.dgp import generate_nonlinear_weakiv_data, generate_weakiv_data

confidence_set_methods = {
    "DRML": DMLConfidenceSetCalculator(),
    "Score": RobustConfidenceSetCalculator(),
    "AR": ARConfidenceSetCalculator(),
}

data_generation_functions = {
    "linear": generate_weakiv_data,
    "nonlinear": generate_nonlinear_weakiv_data,
}
