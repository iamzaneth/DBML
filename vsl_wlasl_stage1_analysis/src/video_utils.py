from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
)


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Metadata describing a single video file.

    Attributes:
        dataset: Name of the parent dataset (e.g. ``"VSL"``).
        class_name: Gloss / class label looked up from the dataset's
            label file (CSV for VSL, JSON for WLASL).
        file_path: Absolute path to the video file on disk.
        duration_seconds: Duration of the video in seconds, or ``None`` if
            it could not be determined.
        frame_count: Total number of frames, or ``None`` if unavailable.
        fps: Frames per second, or ``None`` if unavailable.
        width: Frame width in pixels, or ``None`` if unavailable.
        height: Frame height in pixels, or ``None`` if unavailable.
        codec: FourCC / codec name as reported by the reader, or ``None``.
        readable: Whether the video could be opened and at least
            partially read. Unreadable videos are still recorded so that
            corrupted-file rates can be reported, but they are excluded
            from numeric statistics by the caller.
        regional_variant: VSL-only. The Vietnamese regional dialect
            variant this recording belongs to, decoded from the filename
            suffix convention: ``N`` -> "Nam" [South], ``T`` -> "Trung"
            [Central], ``B`` -> "Bac" [North]. ``None`` for WLASL or for
            VSL videos with no regional suffix.

            IMPORTANT: per the dataset author, suffix letters map to
            region names as N=Nam, T=Trung, B=Bac (i.e. the letter is the
            first letter of the *Vietnamese* region name: Nam, Trung,
            Bac), NOT to signer identity or gender/age.
        signer_id: WLASL-only. The numeric signer identifier reported in
            the WLASL JSON metadata (``instances[i].signer_id``). ``None``
            for VSL, which has no signer identifier in its label file.
    """

    dataset: str
    class_name: str
    file_path: Path
    duration_seconds: Optional[float]
    frame_count: Optional[int]
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    codec: Optional[str]
    readable: bool
    regional_variant: Optional[str] = None
    signer_id: Optional[str] = None


_VSL_REGIONAL_SUFFIX_MAP: dict[str, str] = {
    "N": "Nam",  # Nam = South
    "T": "Trung",  # Trung = Central
    "B": "Bac",  # Bac = North
}

_VSL_FILENAME_PATTERN = re.compile(r"^[A-Za-z]+\d+([A-Za-z]*)\.[A-Za-z0-9]+$")


def parse_vsl_regional_variant(video_filename: str) -> Optional[str]:
    """Decodes the regional-dialect suffix from a VSL video filename.

    VSL filenames follow the pattern ``<prefix><digits><suffix?>.<ext>``
    (e.g. ``D0001N.mp4``, ``W01122B.mp4``, ``D0002.mp4``). The optional
    trailing letter, confirmed by the dataset author, encodes the
    Vietnamese regional dialect performed in that recording: ``N`` ->
    "Nam" (South), ``T`` -> "Trung" (Central), ``B`` -> "Bac" (North).

    Args:
        video_filename: The video filename as listed in the label CSV
            (e.g. ``"D0001N.mp4"``).

    Returns:
        ``"Nam"``, ``"Trung"``, or ``"Bac"`` if a recognized one-letter
        suffix is present; ``None`` if the filename has no suffix or the
        suffix is not one of the three known letters (in which case a
        debug log is emitted rather than raising, since unknown suffixes
        should not halt the pipeline).
    """
    match = _VSL_FILENAME_PATTERN.match(video_filename)
    if not match:
        logger.debug(
            "Filename '%s' does not match the expected VSL naming pattern; "
            "no regional variant assigned.",
            video_filename,
        )
        return None

    suffix = match.group(1).upper()
    if not suffix:
        return None

    variant = _VSL_REGIONAL_SUFFIX_MAP.get(suffix)
    if variant is None:
        logger.debug(
            "Unrecognized regional suffix '%s' in filename '%s'; "
            "no regional variant assigned.",
            suffix,
            video_filename,
        )
    return variant


def load_vsl_label_csv(label_csv_path: Path, videos_dir: Path) -> dict[Path, dict]:
    """Loads the VSL ``video filename -> gloss label`` mapping from a CSV.

    Expected CSV columns: ``ID``, ``VIDEO`` (filename, e.g.
    ``D0001N.mp4``), ``LABEL`` (gloss text, may contain Vietnamese
    diacritics and commas inside quoted fields).

    Each VSL filename may carry a one-letter regional-dialect suffix
    before the ``.mp4`` extension: ``N`` = "Nam" (South), ``T`` =
    "Trung" (Central), ``B`` = "Bac" (North). This convention was
    confirmed by the dataset author; the letter is the first letter of
    the Vietnamese region name, not a signer code or a gender/age marker.
    Filenames with no such suffix (e.g. ``D0002.mp4``) have
    ``regional_variant=None``.

    Args:
        label_csv_path: Path to the VSL label CSV file.
        videos_dir: Path to the directory containing the actual video
            files. Used to resolve each filename to an absolute path and
            to detect filenames listed in the CSV but missing on disk.

    Returns:
        A dict mapping each video's absolute :class:`Path` (only for
        files that actually exist in ``videos_dir``) to a metadata dict
        with keys ``class_name`` (str) and ``regional_variant``
        (``"Nam"``, ``"Trung"``, ``"Bac"``, or ``None``).

    Raises:
        FileNotFoundError: If ``label_csv_path`` or ``videos_dir`` does
            not exist.
    """
    if not label_csv_path.exists():
        raise FileNotFoundError(f"VSL label CSV not found: {label_csv_path}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"VSL videos directory not found: {videos_dir}")

    mapping: dict[Path, dict] = {}
    missing_on_disk: list[str] = []

    with label_csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"VIDEO", "LABEL"}
        if reader.fieldnames is None or not required_columns.issubset(
            set(reader.fieldnames)
        ):
            raise ValueError(
                f"VSL label CSV must contain columns {required_columns}, "
                f"found: {reader.fieldnames}"
            )

        for row in reader:
            video_name = row["VIDEO"].strip()
            label = row["LABEL"].strip()
            if not video_name or not label:
                logger.warning("Skipping row with empty VIDEO/LABEL: %s", row)
                continue

            video_path = videos_dir / video_name
            if not video_path.is_file():
                missing_on_disk.append(video_name)
                continue
            mapping[video_path.resolve()] = {
                "class_name": label,
                "regional_variant": parse_vsl_regional_variant(video_name),
            }

    if missing_on_disk:
        logger.warning(
            "%d video(s) listed in '%s' were not found in '%s' and will be "
            "excluded from analysis (showing up to 5): %s",
            len(missing_on_disk),
            label_csv_path.name,
            videos_dir,
            missing_on_disk[:5],
        )

    logger.info(
        "Loaded VSL label mapping: %d videos resolved out of CSV entries "
        "(%d missing on disk)",
        len(mapping),
        len(missing_on_disk),
    )
    return mapping


def load_wlasl_json(json_path: Path, videos_dir: Path) -> dict[Path, dict]:
    """Loads the WLASL ``video filename -> gloss label`` mapping from JSON.

    Expected JSON shape: a list of objects, each with a ``"gloss"`` string
    and an ``"instances"`` list of objects, each instance carrying a
    ``"video_id"`` string (e.g. ``"07085"``) that maps to a file named
    ``"<video_id>.mp4"`` in ``videos_dir``, and a ``"signer_id"`` integer
    identifying which signer performed that recording.

    Args:
        json_path: Path to the WLASL JSON metadata file
            (e.g. ``WLASL_v0_3.json``).
        videos_dir: Path to the directory containing the actual video
            files. Used to resolve each ``video_id`` to an absolute path
            and to detect ids listed in the JSON but missing on disk.

    Returns:
        A dict mapping each video's absolute :class:`Path` (only for
        files that actually exist in ``videos_dir``) to a metadata dict
        with keys ``class_name`` (str) and ``signer_id`` (str, or
        ``None`` if the instance has no ``signer_id`` field).

    Raises:
        FileNotFoundError: If ``json_path`` or ``videos_dir`` does not
            exist.
        ValueError: If the JSON content is not shaped as expected.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"WLASL JSON metadata not found: {json_path}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"WLASL videos directory not found: {videos_dir}")

    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected WLASL JSON to be a list of gloss entries, got {type(data)}"
        )

    mapping: dict[Path, dict] = {}
    missing_on_disk: list[str] = []

    for entry in data:
        gloss = entry.get("gloss")
        instances = entry.get("instances", [])
        if not gloss or not isinstance(instances, list):
            logger.warning("Skipping malformed gloss entry: %s", entry)
            continue

        for instance in instances:
            video_id = instance.get("video_id")
            if not video_id:
                logger.warning("Skipping instance with no video_id: %s", instance)
                continue

            video_path = videos_dir / f"{video_id}.mp4"
            if not video_path.is_file():
                missing_on_disk.append(f"{video_id}.mp4")
                continue

            signer_id = instance.get("signer_id")
            mapping[video_path.resolve()] = {
                "class_name": gloss,
                "signer_id": str(signer_id) if signer_id is not None else None,
            }

    if missing_on_disk:
        logger.warning(
            "%d video(s) listed in '%s' were not found in '%s' and will be "
            "excluded from analysis (showing up to 5): %s",
            len(missing_on_disk),
            json_path.name,
            videos_dir,
            missing_on_disk[:5],
        )

    logger.info(
        "Loaded WLASL label mapping: %d videos resolved out of JSON entries "
        "(%d missing on disk)",
        len(mapping),
        len(missing_on_disk),
    )
    return mapping


def _fourcc_to_str(fourcc_int: float) -> Optional[str]:
    """Converts an OpenCV FourCC integer code to a 4-character string.

    Args:
        fourcc_int: Raw FourCC value as returned by
            ``cv2.VideoCapture.get(cv2.CAP_PROP_FOURCC)``.

    Returns:
        A 4-character codec tag (e.g. ``"avc1"``), or ``None`` if the value
        is zero / unreadable / non-printable.
    """
    try:
        code = int(fourcc_int)
        if code <= 0:
            return None
        chars = bytes(
            [code & 0xFF, (code >> 8) & 0xFF, (code >> 16) & 0xFF, (code >> 24) & 0xFF]
        )
        text = chars.decode("ascii", errors="ignore").strip()
        return text if text else None
    except (ValueError, OverflowError):
        return None


def _read_with_opencv(video_path: Path) -> Optional[dict]:
    """Attempts to extract metadata from a video using OpenCV.

    Args:
        video_path: Path to the video file.

    Returns:
        A dict with keys ``duration_seconds``, ``frame_count``, ``fps``,
        ``width``, ``height``, ``codec`` if the file opened successfully
        and reported a positive frame count and fps; otherwise ``None``.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)

        if fps <= 0 or frame_count <= 0:
            return None

        return {
            "duration_seconds": float(frame_count / fps),
            "frame_count": int(frame_count),
            "fps": float(fps),
            "width": int(width) if width > 0 else None,
            "height": int(height) if height > 0 else None,
            "codec": _fourcc_to_str(fourcc),
        }
    finally:
        cap.release()


def _read_with_ffprobe(video_path: Path) -> Optional[dict]:
    """Attempts to extract metadata from a video using ``ffprobe``.

    This is used as a fallback when OpenCV fails to open the file or
    returns non-positive fps/frame_count, which happens for some codec /
    container combinations that the local OpenCV build was not compiled
    with support for.

    Args:
        video_path: Path to the video file.

    Returns:
        A dict with the same keys as :func:`_read_with_opencv`, or
        ``None`` if ``ffprobe`` is not installed, fails, or returns no
        usable video stream.
    """
    if shutil.which("ffprobe") is None:
        logger.debug("ffprobe not found on PATH; skipping fallback for %s", video_path)
        return None

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=True
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("ffprobe failed for %s: %s", video_path, exc)
        return None

    try:
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        if not streams:
            return None
        stream = streams[0]

        r_frame_rate = stream.get("r_frame_rate")
        fps = None
        if r_frame_rate and "/" in r_frame_rate:
            num, _, den = r_frame_rate.partition("/")
            den_val = float(den) if float(den) != 0 else None
            fps = float(num) / den_val if den_val else None

        duration = stream.get("duration")
        duration_seconds = float(duration) if duration is not None else None

        nb_frames = stream.get("nb_frames")
        frame_count = int(nb_frames) if nb_frames is not None else None
        if frame_count is None and duration_seconds is not None and fps:
            frame_count = int(round(duration_seconds * fps))
        if duration_seconds is None and frame_count is not None and fps:
            duration_seconds = frame_count / fps

        if fps is None or frame_count is None or not fps:
            return None

        return {
            "duration_seconds": duration_seconds,
            "frame_count": frame_count,
            "fps": fps,
            "width": stream.get("width"),
            "height": stream.get("height"),
            "codec": stream.get("codec_name"),
        }
    except (json.JSONDecodeError, KeyError, ValueError, ZeroDivisionError) as exc:
        logger.warning("Could not parse ffprobe output for %s: %s", video_path, exc)
        return None


def read_video_metadata(
    video_path: Path,
    dataset: str,
    class_name: str,
    regional_variant: Optional[str] = None,
    signer_id: Optional[str] = None,
) -> VideoMetadata:
    """Reads metadata for a single video, trying OpenCV then ffprobe.

    Args:
        video_path: Path to the video file.
        dataset: Name of the dataset this video belongs to.
        class_name: Gloss / class label for this video.
        regional_variant: VSL-only regional dialect label (see
            :class:`VideoMetadata`), or ``None``.
        signer_id: WLASL-only signer identifier (see
            :class:`VideoMetadata`), or ``None``.

    Returns:
        A :class:`VideoMetadata` instance. If neither OpenCV nor
        ``ffprobe`` could read the file, the returned instance has
        ``readable=False`` and all numeric fields set to ``None``.
    """
    info = _read_with_opencv(video_path)
    if info is None:
        logger.debug("OpenCV failed for %s, trying ffprobe fallback", video_path)
        info = _read_with_ffprobe(video_path)

    if info is None:
        logger.warning("Could not read metadata for %s", video_path)
        return VideoMetadata(
            dataset=dataset,
            class_name=class_name,
            file_path=video_path,
            duration_seconds=None,
            frame_count=None,
            fps=None,
            width=None,
            height=None,
            codec=None,
            readable=False,
            regional_variant=regional_variant,
            signer_id=signer_id,
        )

    return VideoMetadata(
        dataset=dataset,
        class_name=class_name,
        file_path=video_path,
        duration_seconds=info["duration_seconds"],
        frame_count=info["frame_count"],
        fps=info["fps"],
        width=info["width"],
        height=info["height"],
        codec=info["codec"],
        readable=True,
        regional_variant=regional_variant,
        signer_id=signer_id,
    )


def scan_dataset_from_mapping(
    video_to_metadata: dict[Path, dict], dataset_name: str
) -> list[VideoMetadata]:
    """Reads metadata for every video in a ``{path: metadata}`` mapping.

    This is the shared second half of the pipeline for both VSL (CSV
    label file) and WLASL (JSON label file): once a mapping from video
    path to label metadata has been built by :func:`load_vsl_label_csv` or
    :func:`load_wlasl_json`, this function reads each video's technical
    metadata identically regardless of which dataset it came from.

    Args:
        video_to_metadata: Mapping from absolute video file path to a
            metadata dict with key ``class_name`` (required) and
            optionally ``regional_variant`` (VSL) or ``signer_id``
            (WLASL).
        dataset_name: Human-readable name for the dataset (e.g. ``"VSL"``),
            stored on every resulting :class:`VideoMetadata` record.

    Returns:
        A list of :class:`VideoMetadata`, one per entry in
        ``video_to_metadata``, sorted by file path for reproducibility.

    Raises:
        ValueError: If ``video_to_metadata`` is empty.
    """
    if not video_to_metadata:
        raise ValueError(
            f"No videos to scan for dataset '{dataset_name}': the label "
            "mapping is empty (check that the label file paths and videos "
            "directory are correct)."
        )

    records: list[VideoMetadata] = []
    for video_path in sorted(video_to_metadata.keys()):
        entry = video_to_metadata[video_path]
        metadata = read_video_metadata(
            video_path,
            dataset=dataset_name,
            class_name=entry["class_name"],
            regional_variant=entry.get("regional_variant"),
            signer_id=entry.get("signer_id"),
        )
        records.append(metadata)

    n_unreadable = sum(1 for r in records if not r.readable)
    n_classes = len({r.class_name for r in records})
    logger.info(
        "Scanned dataset '%s': %d classes, %d videos found, %d unreadable",
        dataset_name,
        n_classes,
        len(records),
        n_unreadable,
    )
    return records
