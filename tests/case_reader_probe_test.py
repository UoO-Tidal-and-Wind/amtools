from pathlib import Path

import numpy as np

from amtools.input_output import CaseReader
from amtools.input_output.post_processing.probe_file import ProbeFile

EXAMPLE_CASE_PATH = Path(__file__).parent / "data" / "example_case"

EXAMPLE_U_DATA = [
    [
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [3.0, 0.0, -3.0],
    ],
    [
        [2.0, 0.0, -2.0],
        [4.0, 0.0, -4.0],
        [6.0, 0.0, -6.0],
    ],
    [
        [3.0, 0.0, -3.0],
        [6.0, 0.0, -6.0],
        [9.0, 0.0, -9.0],
    ],
]

EXAMPLE_P_DATA = [
    [1.0, 0.0, -1.0],
    [2.0, 0.0, -2.0],
    [3.0, 0.0, -3.0],
]


def test_probe_file_position():
    case = CaseReader(EXAMPLE_CASE_PATH)
    probe_data = case.probe("example_probes", "U", "first")
    np.testing.assert_array_equal(probe_data.x, [1.0, 4.0, 7.0])
    np.testing.assert_array_equal(probe_data.y, [2.0, 5.0, 8.0])
    np.testing.assert_array_equal(probe_data.z, [3.0, 6.0, 9.0])


def test_probe_file_time():
    case = CaseReader(EXAMPLE_CASE_PATH)
    probe_data = case.probe("example_probes", "U", "first")
    np.testing.assert_array_equal(probe_data.time, [0.1, 0.2, 0.3])


def test_probe_file_data():
    case = CaseReader(EXAMPLE_CASE_PATH)
    probe_data = case.probe("example_probes", "U", "first")
    np.testing.assert_array_equal(probe_data.data, EXAMPLE_U_DATA)


def test_scalar_probe_file_data():
    case = CaseReader(EXAMPLE_CASE_PATH)
    probe_data = case.probe("example_probes", "p", "first")
    np.testing.assert_array_equal(probe_data.time, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(probe_data.data, EXAMPLE_P_DATA)


def test_compressed_probe_file_data():
    case = CaseReader(EXAMPLE_CASE_PATH)
    probe_data = case.probe("example_probes", "U_compressed", "first")
    np.testing.assert_array_equal(probe_data.data, EXAMPLE_U_DATA)


def test_combined_probe_read():
    case = CaseReader(EXAMPLE_CASE_PATH)
    probe_data = case.probe("example_probes", "U", "combined")
    np.testing.assert_array_equal(probe_data.time, [0.1, 0.2, 0.3, 0.4, 0.5])


def test_probe_time_directory_selection():
    case = CaseReader(EXAMPLE_CASE_PATH)

    first = case.probe("example_probes", "p", "first")
    latest = case.probe("example_probes", "p", "latest")
    exact = case.probe("example_probes", "p", "exactly", "0")
    closest = case.probe("example_probes", "p", "closest to", "0.1")

    np.testing.assert_array_equal(first.time, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(latest.time, [0.3, 0.4, 0.5])
    np.testing.assert_array_equal(exact.time, first.time)
    np.testing.assert_array_equal(closest.time, first.time)


def test_crop_time():
    case = CaseReader(EXAMPLE_CASE_PATH)
    probe_data = case.probe("example_probes", "U", "first")
    probe_data.crop_by_time(lower_limit=0.15, upper_limit=0.25)
    np.testing.assert_array_equal(probe_data.time, [0.2])


def test_sort_by_position_keeps_probe_data_aligned():
    probe_data = CaseReader(EXAMPLE_CASE_PATH).probe("example_probes", "U", "first")
    probe_data.sort_by_position("y")

    np.testing.assert_array_equal(probe_data.y, [2.0, 5.0, 8.0])
    np.testing.assert_array_equal(probe_data.data, EXAMPLE_U_DATA)


def test_remove_duplicate_probe_positions_keeps_first():
    probe_data = ProbeFile("unused")
    probe_data.time = np.array([0.1, 0.2])
    probe_data.x = np.array([1.0, 2.0, 1.0])
    probe_data.y = np.array([3.0, 4.0, 3.0])
    probe_data.z = np.array([5.0, 6.0, 5.0])
    probe_data.data = np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])

    probe_data.rm_duplicates()

    np.testing.assert_array_equal(probe_data.x, [1.0, 2.0])
    np.testing.assert_array_equal(probe_data.y, [3.0, 4.0])
    np.testing.assert_array_equal(probe_data.z, [5.0, 6.0])
    np.testing.assert_array_equal(probe_data.data, [[10.0, 20.0], [11.0, 21.0]])


def test_crop_vector_probe_components():
    probe_data = CaseReader(EXAMPLE_CASE_PATH).probe("example_probes", "U", "first")
    probe_data.crop_by_component_index([0, 2])

    np.testing.assert_array_equal(
        probe_data.data,
        [
            [[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]],
            [[2.0, -2.0], [4.0, -4.0], [6.0, -6.0]],
            [[3.0, -3.0], [6.0, -6.0], [9.0, -9.0]],
        ],
    )


def test_probe_write_read_round_trip(tmp_path):
    original = CaseReader(EXAMPLE_CASE_PATH).probe("example_probes", "U", "first")

    for filename in ("probe_output", "probe_output.gz"):
        output_path = tmp_path / filename
        original.write(output_path)
        reread = ProbeFile(output_path)
        reread.read()

        np.testing.assert_array_equal(reread.time, original.time)
        np.testing.assert_array_equal(reread.data, original.data)
        np.testing.assert_array_equal(reread.x, original.x)
        np.testing.assert_array_equal(reread.y, original.y)
        np.testing.assert_array_equal(reread.z, original.z)


def test_scalar_probe_write_read_round_trip(tmp_path):
    original = CaseReader(EXAMPLE_CASE_PATH).probe("example_probes", "p", "first")
    output_path = tmp_path / "scalar_probe_output"
    original.write(output_path)

    reread = ProbeFile(output_path)
    reread.read()

    np.testing.assert_array_equal(reread.time, original.time)
    np.testing.assert_array_equal(reread.data, original.data)
    np.testing.assert_array_equal(reread.x, original.x)
    np.testing.assert_array_equal(reread.y, original.y)
    np.testing.assert_array_equal(reread.z, original.z)


def test_probe_copy():
    original = CaseReader(EXAMPLE_CASE_PATH).probe("example_probes", "U", "first")
    copy = original.copy()

    np.testing.assert_array_equal(copy.time, original.time)
    np.testing.assert_array_equal(copy.data, original.data)
    np.testing.assert_array_equal(copy.x, original.x)
    np.testing.assert_array_equal(copy.y, original.y)
    np.testing.assert_array_equal(copy.z, original.z)

