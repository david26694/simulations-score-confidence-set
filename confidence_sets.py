from abc import ABC, abstractmethod

import numpy as np
from linearmodels import IV2SLS


class ConfidenceSetCalculator(ABC):
    """
    Abstract base class for confidence set calculators.
    """

    @abstractmethod
    def get_confidence_set(self, data, dml_model, alpha=0.05):
        """
        Abstract method to compute the confidence set.

        Args:
            data: Dictionary containing the data
            dml_model: Fitted DoubleMLIIVM model
            alpha: Significance level (1 - confidence level)

        Returns:
            List of intervals representing the confidence set
        """
        pass


class RobustConfidenceSetCalculator(ConfidenceSetCalculator):
    """Class containing different methods for computing confidence sets"""

    def get_confidence_set(self, data, dml_model, alpha=0.05):
        """
        Compute the confidence set obtained by inverting the score test

        Args:
            data: Dictionary containing the data
            dml_model: Fitted DoubleMLIIVM model
            alpha: Significance level (1 - confidence level)

        Returns:
            List of intervals representing the confidence set
        """
        return dml_model.robust_confset(level=1 - alpha)


class DMLConfidenceSetCalculator(ConfidenceSetCalculator):
    """Class containing methods for computing confidence sets using DML"""

    def get_confidence_set(self, data, dml_model, alpha=0.05):
        """
        Compute the standard DML confidence interval

        Args:
            data: Dictionary containing the data
            dml_model: Fitted DoubleMLIIVM model
            alpha: Significance level (1 - confidence level)

        Returns:
            List containing a single interval representing the confidence set
        """
        dml_confidence_set = dml_model.confint(joint=False)
        return [
            (dml_confidence_set["2.5 %"].iloc[0], dml_confidence_set["97.5 %"].iloc[0])
        ]
