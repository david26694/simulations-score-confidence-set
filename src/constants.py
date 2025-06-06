from src.confidence_sets import (
    DMLConfidenceSetCalculator,
    RobustConfidenceSetCalculator,
)
from src.dgp import generate_weakiv_data

confidence_set_methods = {
    "DRML": DMLConfidenceSetCalculator(),
    "Score": RobustConfidenceSetCalculator(),
}

data_generation_functions = {"linear": generate_weakiv_data}
