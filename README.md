# amtools
[![lint-format](https://github.com/UoO-Tidal-and-Wind/amtools/actions/workflows/CI-lint.yml/badge.svg)](https://github.com/UoO-Tidal-and-Wind/amtools/actions/workflows/CI-lint.yml) [![CI (pytest)](https://github.com/UoO-Tidal-and-Wind/amtools/actions/workflows/CI-pytest.yml/badge.svg)](https://github.com/UoO-Tidal-and-Wind/amtools/actions/workflows/CI-pytest.yml)

`amtools` is a Python package for working with actuator-method simulation output. It includes readers for OpenFOAM case data and utilities for phase averaging.

## Using amtools

`CaseReader` is the main interface for reading data from a case. Its methods cover outputs such as turbine data, probes, planar averages, forces, surfaces, and sheared wake finder results. Time directories can be selected by modes such as `latest`, `first`, `exactly`, `closest to`, or `combined`, where supported.

```python
from amtools.input_output import CaseReader

# define a case reader object
case = CaseReader("/path/to/case")

# use the case reader to read in some probe data (combined across checkpoints)
probe_data = case.probe("probes", "U", time_dir="combined")

# probe data is stored in handy np arrays
vel = probe_data.data  # the sampled data
time = probe_data.time  # the time of the samples
z = probe_data.z  # the z position of the probes

# some other useful utilities...
# Keep samples between 10 and 20 seconds, then order probes by height.
probe_data.crop_by_time(lower_limit=10, upper_limit=20)
probe_data.sort_by_position("z")

# For scalar probe data, average over time for each probe.
mean_by_probe = np.mean(probe_data.data, axis=0)
variance_by_probe = np.var(probe_data.data, axis=0)

# Or use an array's methods, for example:
mean_by_probe = probe_data.data.mean(axis=0)

# Read turbine output such as thrust from the turbineOutput directory.
thrust_data = case.turbine_output("thrust", time_dir="combined")
thrust_time = thrust_data.time
thrust_values = thrust_data.data
```

## Installation

From the project directory, install the package and its runtime dependencies with:

```bash
pip install .
```

The package requires Python 3.10 or later. To install the development dependencies (including pytest and Ruff), use:

```bash
pip install '.[dev]'
```

To ensure the examples you will need to ensure some additional optional dependencies are installed (i.e. matplotlib):


```bash
pip install '.[examples]'
```

## License

The project metadata specifies the MIT License. See `LICENSE` for the license terms.
