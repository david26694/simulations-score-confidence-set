import numpy as np


def generate_weakiv_data(n_samples, slope=0, instrument_strength=5, u_std=1):
    """
    Generate synthetic data for weak instrument simulation
    
    Args:
        n_samples: Number of samples to generate
        slope: Treatment effect (ATE)
        instrument_strength: Strength of the instrument
        u_std: Standard deviation of the error term
        
    Returns:
        Dictionary containing generated data
    """
    u = np.random.normal(0, u_std, size=n_samples)
    X = np.random.normal(0, 1, size=n_samples)
    Z = np.random.binomial(1, 0.5, size=n_samples)
    A = instrument_strength * Z * (X > 0) + u
    A = np.array(A > 0, dtype=int)
    Y = slope * A + 2 * np.sign(u)
    return {"Y": Y, "Z": Z, "A": A, "X": X, "true_ATE": slope}