"""
Phase Averaging Module
=======================

This module contains the `PhaseAverageResult` class for storing phase-averaged results,
and the `phase_average_array` function to compute phase averages from time and value arrays.

The `phase_average_array` function bins data based on calculated phase angles and computes
the mean and standard deviation for each bin. There are options for phase offset removal
and custom binning.

Classes:
    PhaseAverageResult: Stores phase-averaged data, including mean, standard deviation,
    and bin counts.

Functions:
    phase_average_array: Computes phase averages for time-series data and returns a
    `PhaseAverageResult` object containing the results.

Example:
    Example usage of the `phase_average_array` function:

    ```python
    sample_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    sample_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sample_result = phase_average_array(sample_times, sample_values, frequency=1.0)
    print(sample_result.phase_averaged_mean)
    ```

"""

import numpy as np


class PhaseAverageResult:
    """
    A class to hold the results of a phase averaging operation.

    Attributes:
        bin_midpoints (np.ndarray): Midpoints of the bins used for phase averaging.
        phase_averaged_mean (np.ndarray): The mean of the phase averaged values for each bin.
        phase_averaged_std (np.ndarray): The standard deviation of the phase averaged values
        for each bin.
        bin_counts (np.ndarray): The number of values that were assigned to each bin.
    """

    def __init__(self):
        """
        Initializes a PhaseAverageResult object with empty arrays for the attributes.
        """
        self.bin_midpoints = np.array([])
        self.phase_averaged_mean = np.array([])
        self.phase_averaged_std = np.array([])
        self.phase_averaged_covariance = np.array([])
        self.bin_counts = np.array([])

    def get_bins(self):
        """
        Returns the midpoints of the bins used for phase averaging.

        Returns:
            np.ndarray: Array of bin midpoints.
        """
        if self.bin_midpoints.max() > 1.0:
            return self.bin_midpoints / 360
        return self.bin_midpoints

    def get_bins_in_degrees(self):
        """
        Returns the midpoints of the bins used for the phase averaging in degrees.

        Returns:
            np.ndarray: Array of bin midpoints (degrees)
        """
        return self.bin_midpoints

    def get_mean(self):
        """
        Returns the phase averaged mean for each bin.

        Returns:
            np.ndarray: Array of phase averaged means.
        """
        return self.phase_averaged_mean

    def get_std(self):
        """
        Returns the phase averaged standard deviation for each bin.

        Returns:
            np.ndarray: Array of phase averaged standard deviations.
        """
        return self.phase_averaged_std

    def get_variance(self):
        """
        Returns the phase averaged variance for each bin (square of standard deviation).

        Returns:
            np.ndarray: Array of phase averaged variances.
        """
        return self.phase_averaged_covariance


def phase_average_field(
    t_arr: np.ndarray,
    y_arr: np.ndarray,
    frequency: float,
    phase_offset: float = 0,
    number_of_bins: int = 200,
    bin_center_offset: float = None,
    include_0_and_1: bool = True,
) -> PhaseAverageResult:
    """Compute phase averages and covariance for scalar or vector time-series data."""
    del phase_offset  # Retained for backwards-compatible call signatures.
    phase_result = PhaseAverageResult()

    # Calculate phase array
    phase_arr = (frequency * t_arr) % 1.0
    bins = np.linspace(0, 1, number_of_bins + 1)
    bin_midpoints = (bins[1:] + bins[0:-1]) * 0.5

    if bin_center_offset is None:
        bin_center_offset = 0.5 / number_of_bins

    if bin_center_offset != 0:
        if bin_center_offset < 0:
            raise ValueError("Please use a positive bin_center_offset")
        if bin_center_offset > 1.0 / (number_of_bins - 1):
            raise ValueError("bin_center_offset is larger than the bin width")

        # Adjust bin midpoints and bins based on offset
        bins = bins + bin_center_offset
        bin_midpoints = (bin_midpoints + bin_center_offset) % 1.0

        # Ensure first and last values are 0 and 360
        bins = np.insert(bins, 0, 0)
        bins[-1] = 1.0

    # Bin values based on original phase
    bin_inds = np.digitize(phase_arr, bins)

    # Create binned data
    binned_y_arr = [y_arr[bin_inds == i] for i in range(1, number_of_bins + 2)]

    # Adjust the first and last bins if there is an offset
    if bin_center_offset != 0:
        binned_y_arr[-1] = np.concatenate((binned_y_arr[0], binned_y_arr[-1]))
        binned_y_arr = binned_y_arr[1:]

    cov_matrices = []
    mean_matrices = []
    for group in binned_y_arr:
        if len(group) > 1:
            # Center the data by subtracting the mean from each component (column)
            group_mean = np.mean(group, axis=0)

            centered_group = group - group_mean

            # Compute the covariance matrix (3x3 in case of 3 components)
            if y_arr.ndim > 1:
                cov_matrix = np.cov(centered_group, rowvar=False)
            else:
                cov_matrix = np.var(centered_group)

            cov_matrices.append(cov_matrix)
            mean_matrices.append(group_mean)

        else:
            field_value_shape = y_arr[0].shape[0]

            dummy_mean_value = np.full((field_value_shape), np.nan)
            dummy_covariance_value = np.full(
                (field_value_shape, field_value_shape), np.nan
            )

            # print(dummy_covariance_value.shape)
            cov_matrices.append(dummy_covariance_value)
            mean_matrices.append(dummy_mean_value)

            print("Warning: empty bin")

    phase_averaged_covariance = np.array(cov_matrices)
    phase_averaged_y_arr = np.array(mean_matrices)

    # print(phase_averaged_y_arr)

    # Sort bin midpoints and results
    sorted_bin_midpoints_inds = np.argsort(bin_midpoints)
    bin_midpoints = bin_midpoints[sorted_bin_midpoints_inds]
    phase_averaged_y_arr = np.array(phase_averaged_y_arr)[sorted_bin_midpoints_inds]
    phase_averaged_covariance = np.array(phase_averaged_covariance)[
        sorted_bin_midpoints_inds
    ]

    # Optionally include 0 and 360 as bin midpoints
    if include_0_and_1:
        if 0 in bin_midpoints:
            bin_midpoints = np.append(bin_midpoints, 1)
            phase_averaged_y_arr = np.concatenate(
                [phase_averaged_y_arr, phase_averaged_y_arr[0:1]], axis=0
            )
            phase_averaged_covariance = np.concatenate(
                [phase_averaged_covariance, phase_averaged_covariance[0:1]], axis=0
            )
            binned_y_arr.append(binned_y_arr[0])

    # Store the results in the result object
    phase_result.bin_midpoints = np.array(bin_midpoints)
    phase_result.phase_averaged_mean = np.array(phase_averaged_y_arr)
    phase_result.phase_averaged_covariance = np.array(phase_averaged_covariance)

    if y_arr.ndim > 1:
        phase_result.phase_averaged_std = np.sqrt(
            np.diagonal(phase_averaged_covariance, axis1=1, axis2=2)
        )
    else:
        phase_result.phase_averaged_std = np.sqrt(
		    phase_averaged_covariance
		)

    phase_result.bin_counts = np.array([len(arr) for arr in binned_y_arr])

    return phase_result

