"""
Helpers for loading sampled surface files from ``postProcessing``.
"""

import gzip
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv


class SurfaceFiles:
    """
    Helper for a sampled-surface output directory.

    The directory is expected to contain numeric time subdirectories. Individual
    surface files are read with :func:`pyvista.read`.

    Attributes:
        data_dir: Directory containing the sampled-surface time directories.
        times_paths: Paths to numeric time directories.
        times_names: Time directory names as strings.
        times_floats: Time directory names converted to floats.
    """

    def __init__(self, surfaces_data_dir):
        """
        Initialize the surface-file helper.

        Args:
            surfaces_data_dir: Path to a sampled-surface directory under
                ``postProcessing``.
        """
        self.data_dir = surfaces_data_dir

        self.times_paths = [
            p
            for p in self.data_dir.glob("*")
            if p.name.replace(".", "", 1).isdigit()  # allow numbers like 0.001
        ]
        self.times_names = np.array([path.name for path in self.times_paths])
        self.times_floats = np.array([float(name) for name in self.times_names])

    def read_file(self, time: float, name: str) -> pv.DataObject:
        """
        Read a named surface file at an exact output time.

        Args:
            time: Exact time directory to read from.
            name: File name inside the selected time directory.

        Returns:
        The PyVista data object returned by :func:`pyvista.read`.

        Raises:
            ValueError: If ``time`` is not one of the available time directories.
            FileNotFoundError: If neither the named file nor its ``.gz`` version exists.
        """
        if time not in self.times_floats:
            raise ValueError(
                f"Time {time} not found in available times: {self.times_floats}"
            )

        time_index = np.where(self.times_floats == time)[0][0]
        file_path = self.times_paths[time_index]

        surface_path = file_path / name
        if surface_path.exists():
            return pv.read(surface_path)

        compressed_path = (
            surface_path if surface_path.suffix == ".gz" else Path(f"{surface_path}.gz")
        )
        if not compressed_path.exists():
            raise FileNotFoundError(
                f"Neither '{surface_path}' nor '{compressed_path}' exists."
            )

        # PyVista/VTK readers do not consistently accept gzip-compressed inputs.
        # Decompress to a temporary path that retains the original file extension.
        decompressed_name = compressed_path.name[:-3]
        with tempfile.TemporaryDirectory() as temporary_directory:
            decompressed_path = Path(temporary_directory) / decompressed_name
            with gzip.open(compressed_path, "rb") as compressed_file:
                with decompressed_path.open("wb") as output_file:
                    shutil.copyfileobj(compressed_file, output_file)
            return pv.read(decompressed_path)

    def read_nearest_file(self, time: float, name: str) -> pv.DataObject:
        """
        Read a named surface file at the nearest available output time.

        Args:
            time: Target time value.
            name: File name inside the nearest time directory.

        Returns:
            A tuple of ``(mesh, nearest_time)``.
        """
        nearest_time = min(self.times_floats, key=lambda t: abs(t - time))
        return self.read_file(nearest_time, name), nearest_time

    def get_time_floats(self) -> np.ndarray:
        """
        Return available surface output times as floats.

        Returns:
            One-dimensional array of available time-directory values.
        """
        return self.times_floats
