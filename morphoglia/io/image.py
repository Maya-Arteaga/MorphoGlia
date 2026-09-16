# morphoglia/io/image.py
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from typing import Iterable
import cv2

from ..core.models import ImageRecord, DatasetIndex
from ..metadata.nomenclature import NomenclatureExtractor

class ImageIndexer:
    """
    Discovers image files in the input directory and registers them
    in the dataset index using extracted metadata.
    """
    def __init__(self, input_dir: Path, extractor: NomenclatureExtractor,
                 exts=(".tif", ".tiff", ".png", ".jpg")):
        self.input_dir = input_dir
        self.extractor = extractor
        self.exts = tuple(e.lower() for e in exts)

    def discover_paths(self) -> Iterable[Path]:
        for p in self.input_dir.iterdir():
            if p.is_file() and p.suffix.lower() in self.exts:
                yield p

    def build_index(self, dataset: DatasetIndex):
        for img_path in self.discover_paths():
            meta = self.extractor.parse(img_path)
            image_id = img_path.stem
            dataset.add_image(ImageRecord(image_id=image_id, path=img_path, metadata=meta))

class ImageLoader:
    """
    Load grayscale images.

    ImageLoader performs I/O only.

    It does not invert, threshold, normalize, or otherwise reinterpret
    image polarity. Those operations belong to Preprocessing.

    The optional config argument is retained temporarily for call-site
    compatibility but is not used.
    """

    def __init__(
        self,
        config=None,
    ):

        self.config = config


    @lru_cache(
        maxsize=16
    )
    def _read(
        self,
        path: Path,
    ):

        img = cv2.imread(
            str(
                path
            ),
            cv2.IMREAD_GRAYSCALE,
        )


        if img is None:

            raise FileNotFoundError(
                f"Could not read image {path}"
            )


        return img


    def load(
        self,
        record: ImageRecord,
    ):

        img = self._read(
            record.path
        )


        if record.size is None:

            record.size = (
                img.shape
            )


        return img
