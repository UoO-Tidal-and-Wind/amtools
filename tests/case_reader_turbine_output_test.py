from pathlib import Path

import numpy as np

from amtools.input_output import CaseReader

EXAMPLE_CASE_PATH = Path(__file__).parent / "data" / "example_case"

EXAMPLE_THRUST_DATA = [[10.0], [20.0], [30.0]]
EXAMPLE_AXIAL_FORCE_DATA = [
    [10.0, 20.0, 30.0],
    [0.0, 0.0, 0.0],
    [-10.0, -20.0, -30.0],
    [20.0, 40.0, 60.0],
    [0.0, 0.0, 0.0],
    [-20.0, -40.0, -60.0],
    [30.0, 60.0, 90.0],
    [0.0, 0.0, 0.0],
    [-30.0, -60.0, -90.0],
]


def test_turbine_output_data_and_metadata():
    output = CaseReader(EXAMPLE_CASE_PATH).turbine_output("thrust", "first")

    np.testing.assert_array_equal(output.time, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(output.dt, [0.1, 0.1, 0.1])
    np.testing.assert_array_equal(output.turbine, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(output.data, EXAMPLE_THRUST_DATA)


def test_turbine_output_with_blade_metadata():
    output = CaseReader(EXAMPLE_CASE_PATH).turbine_output("axialForce", "first")

    np.testing.assert_array_equal(
        output.time, [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3]
    )
    np.testing.assert_array_equal(output.blade, [0, 1, 2] * 3)
    np.testing.assert_array_equal(output.data, EXAMPLE_AXIAL_FORCE_DATA)


def test_turbine_output_time_directory_selection():
    case = CaseReader(EXAMPLE_CASE_PATH)

    first = case.turbine_output("thrust", "first")
    latest = case.turbine_output("thrust", "latest")
    exact = case.turbine_output("thrust", "exactly", "0")
    closest = case.turbine_output("thrust", "closest to", "0.25")

    np.testing.assert_array_equal(first.time, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(latest.time, [0.3, 0.4, 0.5])
    np.testing.assert_array_equal(exact.time, first.time)
    np.testing.assert_array_equal(closest.time, latest.time)


def test_combined_turbine_output_drops_restart_overlap():
    output = CaseReader(EXAMPLE_CASE_PATH).turbine_output("thrust", "combined")

    np.testing.assert_array_equal(output.time, [0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_array_equal(output.data, [[10.0], [20.0], [30.0], [40.0], [50.0]])


def test_crop_turbine_output_by_time_and_blade():
    output = CaseReader(EXAMPLE_CASE_PATH).turbine_output("axialForce", "first")
    output.crop_time(lower_limit=0.15, upper_limit=0.25)
    np.testing.assert_array_equal(output.time, [0.2, 0.2, 0.2])
    np.testing.assert_array_equal(output.blade, [0, 1, 2])

    output.crop_blade(2)
    np.testing.assert_array_equal(output.time, [0.2])
    np.testing.assert_array_equal(output.data, [[-20.0, -40.0, -60.0]])
