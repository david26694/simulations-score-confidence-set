from abc import ABC, abstractmethod

import numpy as np
from linearmodels import IV2SLS
from ivmodels import KClass
from ivmodels.summary import Summary
from src.utils import  confidence_sets_to_tuples


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

class ARConfidenceSetCalculator(ConfidenceSetCalculator):

    @staticmethod
    def _fit_ar_summary(data, alpha):
        kclass_model = KClass()

        kclass_summary = Summary(kclass_model, "anderson-rubin", alpha=alpha).fit(
            X=data['A'].reshape(-1, 1),
            y=data['Y'],
            Z=data['Z'].reshape(-1, 1),
            C=data['X'].reshape(-1, 1)
        )

        return kclass_summary

    def get_confidence_set(self, data, dml_model, alpha=0.05):
        kclass_summary = self._fit_ar_summary(data, alpha)
        conf_set = confidence_sets_to_tuples(kclass_summary.coefficient_table_.confidence_sets[1])

        return conf_set