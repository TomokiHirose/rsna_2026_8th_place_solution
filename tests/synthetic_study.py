from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

DEFAULT_IOP = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # axial: normal = +z


def slice_pattern(index: int, shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    """A slice whose mean encodes its position, so the slice order can be checked."""
    rows, cols = shape
    ramp = np.linspace(0, 200, cols, dtype=np.float32)[None, :] + np.linspace(0, 100, rows, dtype=np.float32)[:, None]
    return (ramp + 1000.0 * (index + 1)).astype(np.uint16)


def write_slice(
    path: Path,
    array: np.ndarray,
    *,
    pixel_spacing: float | None = 2.0,
    ipp: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
    iop: tuple[float, ...] | None = DEFAULT_IOP,
    instance_number: int | None = 1,
    photometric: str = "MONOCHROME2",
    slope: float | None = None,
    intercept: float | None = None,
    truncate_bytes: int = 0,
    **extra: object,
) -> Path:
    """Write one uncompressed 16-bit MR slice (Explicit VR Little Endian); ``truncate_bytes`` corrupts the pixels."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "MR"
    ds.Rows, ds.Columns = array.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    if pixel_spacing is not None:
        ds.PixelSpacing = [float(pixel_spacing), float(pixel_spacing)]
    if ipp is not None:
        ds.ImagePositionPatient = [float(v) for v in ipp]
    if iop is not None:
        ds.ImageOrientationPatient = [float(v) for v in iop]
    if instance_number is not None:
        ds.InstanceNumber = int(instance_number)
    if slope is not None:
        ds.RescaleSlope = float(slope)
    if intercept is not None:
        ds.RescaleIntercept = float(intercept)
    for key, value in extra.items():
        setattr(ds, key, value)
    ds.PixelData = np.ascontiguousarray(array, dtype=np.uint16).tobytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)
    if truncate_bytes:
        raw = path.read_bytes()
        path.write_bytes(raw[:-truncate_bytes])
    return path


def make_series(
    directory: Path,
    n_slices: int = 5,
    shape: tuple[int, int] = (64, 64),
    *,
    pixel_spacing: float | None = 2.0,
    with_ipp: bool = True,
    with_iop: bool = True,
    with_instance_number: bool = True,
    reverse_instance_numbers: bool = False,
    spacing_mm: float = 4.0,
    photometric: str = "MONOCHROME2",
    slope: float | None = None,
    intercept: float | None = None,
    truncate_index: int | None = None,
    seed: int = 0,
) -> list[np.ndarray]:
    """Write a series with random file names so file order differs from slice order; returns the raw slices in true order."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arrays = [slice_pattern(i, shape) for i in range(n_slices)]
    for i, array in enumerate(arrays):
        name = f"{uuid.UUID(bytes=rng.bytes(16))}.dcm"
        instance = (n_slices - i) if reverse_instance_numbers else (i + 1)
        write_slice(
            directory / name,
            array,
            pixel_spacing=pixel_spacing,
            ipp=(0.0, 0.0, i * spacing_mm) if with_ipp else None,
            iop=DEFAULT_IOP if with_iop else None,
            instance_number=instance if with_instance_number else None,
            photometric=photometric,
            slope=slope,
            intercept=intercept,
            truncate_bytes=2000 if truncate_index == i else 0,
        )
    return arrays


def make_study(root: Path, study_id: str, series_specs: list[dict], seed: int = 0) -> list[dict]:
    """Write a study directory and return its ``{split}_series.csv`` rows.

    Each spec needs ``plane`` and ``fluid``; ``uid``, ``n_slices``, ``shape``, ``pixel_spacing``, ``truncate_index`` and ``empty`` are optional.
    """
    rows: list[dict] = []
    study_dir = root / study_id
    for i, spec in enumerate(series_specs):
        uid = str(spec.get("uid") or f"1.2.826.0.1.{study_id[-4:]}.{i}")
        series_dir = study_dir / uid
        if spec.get("empty"):
            series_dir.mkdir(parents=True, exist_ok=True)
        else:
            make_series(
                series_dir,
                n_slices=int(spec.get("n_slices", 5)),
                shape=tuple(spec.get("shape", (64, 64))),
                pixel_spacing=spec.get("pixel_spacing", 2.0),
                truncate_index=spec.get("truncate_index"),
                photometric=spec.get("photometric", "MONOCHROME2"),
                seed=seed * 97 + i,
            )
        rows.append({"StudyInstanceUID": study_id, "SeriesInstanceUID": uid, "Fluid_Sensitive": int(spec["fluid"]), "Fat_Suppression": int(spec.get("fat", spec["fluid"])), "Anatomical_Plane": spec["plane"]})
    return rows
