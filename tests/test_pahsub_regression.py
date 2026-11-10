import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from astropy.io import fits
from astropy.table import QTable
import astropy.units as u


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "tests" / "out"
REFERENCE_DIR = ROOT / "tests" / "pahsub" / "reference"
RELATIVE_TOLERANCE = 1e-6
FITS_ROWS_PER_CHUNK = 64


def _assert_values_match(actual, reference, label):
    actual_unit = getattr(actual, "unit", None)
    reference_unit = getattr(reference, "unit", None)
    if actual_unit is not None or reference_unit is not None:
        assert actual_unit is not None, f"{label}: actual values have no unit"
        assert reference_unit is not None, f"{label}: reference values have no unit"
        assert actual_unit.is_equivalent(reference_unit), (
            f"{label}: incompatible units {actual_unit} and {reference_unit}"
        )
        actual = u.Quantity(actual).to_value(reference_unit)
        reference = u.Quantity(reference).to_value(reference_unit)

    actual = np.asanyarray(actual)
    reference = np.asanyarray(reference)
    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual),
        np.ma.getmaskarray(reference),
        err_msg=f"{label}: masks differ",
    )
    actual = np.ma.filled(actual, np.nan)
    reference = np.ma.filled(reference, np.nan)

    if np.issubdtype(reference.dtype, np.number):
        np.testing.assert_allclose(
            actual,
            reference,
            rtol=RELATIVE_TOLERANCE,
            atol=0,
            equal_nan=True,
            err_msg=f"{label}: values differ",
        )
    else:
        np.testing.assert_array_equal(
            actual,
            reference,
            err_msg=f"{label}: values differ",
        )


def _assert_bgsub_matches_reference():
    filename = "ngc3351_convolved_pah_f360m_bgsub.fits"
    with (
        fits.open(OUTPUT_DIR / filename, memmap=True) as actual_hdul,
        fits.open(REFERENCE_DIR / filename, memmap=True) as reference_hdul,
    ):
        assert len(actual_hdul) == len(reference_hdul)
        for hdu_index, (actual_hdu, reference_hdu) in enumerate(
            zip(actual_hdul, reference_hdul, strict=True)
        ):
            assert actual_hdu.name == reference_hdu.name
            if reference_hdu.data is None:
                assert actual_hdu.data is None
                continue

            assert actual_hdu.data is not None
            assert actual_hdu.data.shape == reference_hdu.data.shape
            for row_start in range(0, reference_hdu.data.shape[0], FITS_ROWS_PER_CHUNK):
                row_stop = row_start + FITS_ROWS_PER_CHUNK
                _assert_values_match(
                    actual_hdu.data[row_start:row_stop],
                    reference_hdu.data[row_start:row_stop],
                    f"{filename} HDU {hdu_index}, rows {row_start}:{row_stop}",
                )


def _assert_catalogs_match_references():
    actual_paths = {path.name: path for path in OUTPUT_DIR.glob("*.ecsv")}
    reference_paths = {path.name: path for path in REFERENCE_DIR.glob("*.ecsv")}
    assert actual_paths.keys() == reference_paths.keys()

    for filename, reference_path in sorted(reference_paths.items()):
        actual = QTable.read(actual_paths[filename])
        reference = QTable.read(reference_path)
        assert actual.colnames == reference.colnames, f"{filename}: columns differ"
        assert len(actual) == len(reference), f"{filename}: row counts differ"

        for column_name in reference.colnames:
            _assert_values_match(
                actual[column_name],
                reference[column_name],
                f"{filename}, column {column_name}",
            )


def test_va_test1_outputs_match_reference():
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir()

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    pythonpath = str(ROOT / "src")
    if env.get("PYTHONPATH"):
        pythonpath = os.pathsep.join((pythonpath, env["PYTHONPATH"]))
    env["PYTHONPATH"] = pythonpath

    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "rphot" / "main.py")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"main.py failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    _assert_bgsub_matches_reference()
    _assert_catalogs_match_references()
