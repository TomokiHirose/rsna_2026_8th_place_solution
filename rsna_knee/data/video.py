import json
from typing import Any

import numpy as np
import torch

from rsna_knee.config import ModelConfig

METADATA_FIELDS = ("slot_name", "SeriesDescription", "ScanningSequence", "ScanOptions", "RepetitionTime", "EchoTime", "SliceThickness", "SpacingBetweenSlices", "PixelSpacing_0", "PixelSpacing_1", "ImageOrientationPatient")

NUMERIC_METADATA = {"RepetitionTime", "EchoTime", "SliceThickness", "SpacingBetweenSlices", "PixelSpacing_0", "PixelSpacing_1"}


def metadata_value(header: dict[str, Any], key: str) -> str:
    """CSV と DICOM の数値表現を同じ文字列に揃える。"""
    value = header.get(key)
    if value is None or value == "":
        return "unknown"
    return f"{float(value):g}" if key in NUMERIC_METADATA else str(value)


def video_inputs(processor: Any, cfg: ModelConfig, pixels: np.ndarray, mask: np.ndarray, headers: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """選択系列を動画に変換し、判定基準と撮像メタデータを前置した Qwen 入力を返す。"""
    from transformers.video_utils import VideoMetadata

    assigned = {int(row["slot_index"]): row for row in headers if int(row["assigned"]) == 1}
    videos, metadata, descriptions, content = [], [], [], []
    offset = 0
    for slot, count in enumerate(cfg.preprocess.slot_table):
        real = np.flatnonzero(mask[offset : offset + count])
        if not len(real):
            descriptions.append({"slot": slot, "status": "missing"})
            offset += count
            continue
        header = assigned[slot]
        picks = json.loads(header["selected_indices"])
        indices = [picks[i] for i in real]
        frames = np.array(pixels[offset + real], copy=True)
        videos.append(torch.from_numpy(frames[:, None]).expand(-1, 3, -1, -1).contiguous())
        metadata.append(VideoMetadata(total_num_frames=int(header["n_files"]), fps=cfg.vlm.pseudo_fps, frames_indices=indices))
        descriptions.append({"video": len(videos), "slot": slot, **{key: metadata_value(header, key) for key in METADATA_FIELDS}})
        content.extend([{"type": "text", "text": f"Video {len(videos)} (slot {slot}):"}, {"type": "video"}])
        offset += count
    if not videos:
        raise ValueError("study contains no real video frames")
    instruction = cfg.vlm.prompt + "\nVideo timestamps encode ordered slice indices at a fixed pseudo frame rate; they are not acquisition times. TR/TE are in ms; slice and pixel spacing are in mm.\n" + json.dumps(descriptions, ensure_ascii=False)
    content.insert(0, {"type": "text", "text": instruction})
    text = processor.tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=False)
    result = processor(text=[text], videos=videos, video_metadata=metadata, do_sample_frames=False, do_resize=False, cap_pixels_per_frame=False, return_tensors="pt")
    return {key: value for key, value in result.items() if torch.is_tensor(value)}
