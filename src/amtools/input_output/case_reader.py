"""
Case-level readers for OpenFOAM actuator-method output.

The :class:`CaseReader` class stores the paths for one simulation case and
provides convenience methods for common output under ``turbineOutput`` and
``postProcessing``. Reader methods support the same time-directory selection
modes where applicable: ``"latest"``, ``"first"``, ``"exactly"``,
``"closest to"``, and ``"combined"``.

Example:
    .. code-block:: python

       case = CaseReader("/path/to/case")
       turbine_data = case.turbine_output("bladeForce", time_dir="latest")
       velocity_probes = case.probe("probes", "U", time_dir="combined")
       planar_u = case.planar_average("planarAveraging", "U", time_dir="combined")
"""

from pathlib import Path
from typing import Literal
import numpy as np

from .turbine_output.turbine_output_file import TurbineOutputFile
from .post_processing.probe_file import ProbeFile
from .post_processing.surface_files import SurfaceFiles
from .post_processing.force_file import ForceFile
from .post_processing.time_file import TimeFile
from .post_processing.phase_probe_file import PhaseProbeFile
from .post_processing.planar_average_file import PlanarAverageFile
from .wake_finder.sheared_wake_finder_file import ShearedWakeFinderFile


class CaseReader:
    """
    Reader facade for one OpenFOAM case directory.

    The class does not load data at construction time. Each public reader method
    locates a specific output file, constructs the corresponding file-reader class,
    loads it, and returns the populated object.

    Attributes:
        path: Root path of the case.
        name: Name of the case directory.
        turbine_output_path: Path to ``turbineOutput`` under the case.
        post_processing_path: Path to ``postProcessing`` under the case.
        turbine_output_time_dir: Name of the most recently selected time directory.
    """

    def __init__(self, root_path: str) -> None:
        """
        Initialize a case reader.

        Args:
            root_path: Path to the root directory of the OpenFOAM case.

        Raises:
            FileNotFoundError: If ``root_path`` does not exist.
        """
        self.path: Path = Path(root_path)  # Convert to Path object
        if not self.path.exists():
            raise FileNotFoundError(
                f"The path '{self.post_processing_path / surfaces_name}' does not exist."
            )
        surface_files = SurfaceFiles(self.post_processing_path / surfaces_name)
        return surface_files

    def force(
        self,
        force_name: str,
        time_dir: Literal[
            "latest", "first", "exactly", "closest to", "combined"
        ] = "latest",
        time_dir_value: str = "",
    ):
        """
        Load ``force.dat`` from a post-processing force directory.

        Files are resolved as ``<case>/postProcessing/<force_name>/<time_dir>/force.dat``.
        When ``time_dir="combined"``, later time directories are appended after rows
        that overlap the current maximum time are dropped.

        Args:
            force_name: Name of the force output directory under ``postProcessing``.
            time_dir: Time-directory selection mode. Accepted values are ``"latest"``,
                ``"first"``, ``"exactly"``, ``"closest to"``, and ``"combined"``.
            time_dir_value: Required target value for ``"exactly"`` and
                ``"closest to"``.

        Returns:
            A populated :class:`~amtools.input_output.post_processing.force_file.ForceFile`.

        Raises:
            FileNotFoundError: If the force directory or ``force.dat`` file is missing.
            ValueError: If ``time_dir_value`` is required but not provided.
        """

        force_path = self.post_processing_path / force_name

        if not force_path.exists():
            raise FileNotFoundError(f"The path '{force_path}' does not exist.")

        combine_files = False
        if time_dir == "combined":
            # read in the first time directory to then combine with others later
            time_dir = "first"
            combine_files = True

        time_dirs = [f.name for f in force_path.iterdir()]
        if time_dir == "latest":
            time_dirs_float = [float(f) for f in time_dirs]
            index = np.argmax(time_dirs_float)
            self.turbine_output_time_dir = time_dirs[index]
        elif time_dir == "first":
            time_dirs_float = [float(f) for f in time_dirs]
            index = np.argmin(time_dirs_float)
            self.turbine_output_time_dir = time_dirs[index]
        elif time_dir == "exactly":
            if not time_dir_value:
                raise ValueError(
                    "time_dir_value must be provided when using 'exactly'."
                )
            self.turbine_output_time_dir = time_dir_value
        elif time_dir == "closest to":
            if not time_dir_value:
                raise ValueError(
                    "time_dir_value must be provided when using 'closest to'."
                )
            time_dirs_float = [float(f) for f in time_dirs]
            index = np.argmin(np.abs(time_dirs_float - float(time_dir_value)))
            self.turbine_output_time_dir = time_dirs[index]

        file_path = force_path / self.turbine_output_time_dir / "force.dat"

        if not file_path.exists():
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        file_reader = ForceFile(file_path)
        file_reader.read()

        if combine_files:
            time_dirs_to_combine = time_dirs.copy()
            time_dirs_to_combine.remove(self.turbine_output_time_dir)

            if len(time_dirs_to_combine) != 0:
                # sort the time dirs by numerical value
                def convert_to_float(item):
                    return float(item)

                time_dirs_to_combine.sort(key=convert_to_float)

                for new_time_dir in time_dirs_to_combine:
                    new_file_path = force_path / new_time_dir / "force.dat"
                    new_file_reader = ForceFile(new_file_path)
                    new_file_reader.read()

                    current_max_time = file_reader.time.max()
                    # crop the new file
                    new_file_reader.crop_by_time(lower_limit=current_max_time)

                    # now concatenate the data together
                    file_reader.data = np.concatenate(
                        (file_reader.data, new_file_reader.data)
                    )
                    file_reader.time = np.concatenate(
                        (file_reader.time, new_file_reader.time)
                    )

                    file_reader.split_data()


        return file_reader

    def time(
        self,
        time_name: str,
        time_dir: Literal[
            "latest", "first", "exactly", "closest to", "combined"
        ] = "latest",
        time_dir_value: str = "",
    ):
        """
        Load ``timeInfo.dat`` from a post-processing time directory.

        Files are resolved as ``<case>/postProcessing/<time_name>/<time_dir>/timeInfo.dat``.
        When ``time_dir="combined"``, later time directories are appended after rows
        that overlap the current maximum time are dropped.

        Args:
            time_name: Name of the time output directory under ``postProcessing``.
            time_dir: Time-directory selection mode. Accepted values are ``"latest"``,
                ``"first"``, ``"exactly"``, ``"closest to"``, and ``"combined"``.
            time_dir_value: Required target value for ``"exactly"`` and
                ``"closest to"``.

        Returns:
            A populated :class:`~amtools.input_output.post_processing.time_file.TimeFile`.

        Raises:
            FileNotFoundError: If the time output directory or ``timeInfo.dat`` file is missing.
            ValueError: If ``time_dir_value`` is required but not provided.
        """

        time_path = self.post_processing_path / time_name

        if not time_path.exists():
            raise FileNotFoundError(f"The path '{time_path}' does not exist.")

        combine_files = False
        if time_dir == "combined":
            # read in the first time directory to then combine with others later
            time_dir = "first"
            combine_files = True

        time_dirs = [f.name for f in time_path.iterdir()]
        if time_dir == "latest":
            time_dirs_float = [float(f) for f in time_dirs]
            index = np.argmax(time_dirs_float)
            self.turbine_output_time_dir = time_dirs[index]
        elif time_dir == "first":
            time_dirs_float = [float(f) for f in time_dirs]
            index = np.argmin(time_dirs_float)
            self.turbine_output_time_dir = time_dirs[index]
        elif time_dir == "exactly":
            if not time_dir_value:
                raise ValueError(
                    "time_dir_value must be provided when using 'exactly'."
                )
            self.turbine_output_time_dir = time_dir_value
        elif time_dir == "closest to":
            if not time_dir_value:
                raise ValueError(
                    "time_dir_value must be provided when using 'closest to'."
                )
            time_dirs_float = [float(f) for f in time_dirs]
            index = np.argmin(np.abs(time_dirs_float - float(time_dir_value)))
            self.turbine_output_time_dir = time_dirs[index]

        file_path = time_path / self.turbine_output_time_dir / "timeInfo.dat"

        if not file_path.exists():
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        file_reader = TimeFile(file_path)
        file_reader.read()

        if combine_files:
            time_dirs_to_combine = time_dirs.copy()
            time_dirs_to_combine.remove(self.turbine_output_time_dir)

            if len(time_dirs_to_combine) != 0:
                # sort the time dirs by numerical value
                def convert_to_float(item):
                    return float(item)

                time_dirs_to_combine.sort(key=convert_to_float)

                for new_time_dir in time_dirs_to_combine:
                    new_file_path = time_path / new_time_dir / "timeInfo.dat"
                    new_file_reader = TimeFile(new_file_path)
                    new_file_reader.read()

                    current_max_time = file_reader.time.max()
                    # crop the new file
                    new_file_reader.crop_by_time(lower_limit=current_max_time)

                    # now concatenate the data together
                    file_reader.data = np.concatenate(
                        (file_reader.data, new_file_reader.data)
                    )
                    file_reader.time = np.concatenate(
                        (file_reader.time, new_file_reader.time)
                    )

                    file_reader.split_data()


        return file_reader
