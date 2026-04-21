import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime

try:
    import gridfs
    from pymongo import MongoClient
except ImportError:
    gridfs = None
    MongoClient = None


class MongoHandler:
    def __init__(
        self,
        mongo_uri: str,
        db_name: str = "smart_crossing_db",
        collection_name: str = "violations"
    ):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name

        self.client = None
        self.db = None
        self.collection = None
        self.fs = None

    # =========================================================
    # CONNECT
    # =========================================================
    def connect(self) -> bool:
        if MongoClient is None or gridfs is None:
            print("MongoDB support disabled: install pymongo to enable database storage.")
            return False

        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            self.client.server_info()

            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            self.fs = gridfs.GridFS(self.db)

            print(
                f"MongoDB Atlas connected | db={self.db_name} | "
                f"collection={self.collection_name}"
            )
            return True

        except Exception as e:
            print(f"MongoDB unavailable: {e}")
            return False

    # =========================================================
    # SAVE VIOLATION (IMAGE → GRIDFS)
    # =========================================================
    def save_violation(
        self,
        frame: np.ndarray,
        track_id: int,
        vtype: str,
        speed: float = 0.0,
        metadata: dict = None
    ) -> bool:

        if self.fs is None:
            print("Save failed: MongoDB not connected")
            return False

        try:
            # Encode image
            _, encoded = cv2.imencode(".jpg", frame)
            image_id = self.fs.put(encoded.tobytes())

            doc = {
                "track_id": track_id,
                "violation_type": vtype,
                "speed_kmph": float(speed),
                "timestamp": datetime.now(),
                "image_id": image_id,
                "metadata": metadata or {}
            }

            result = self.collection.insert_one(doc)

            print(
                f"Saved {vtype} | ID={track_id} | Speed={speed:.1f} km/h | "
                f"db={self.db_name} | collection={self.collection_name} | "
                f"doc_id={result.inserted_id} | image_id={image_id}"
            )
            return True

        except Exception as e:
            print(f"Save failed: {e}")
            return False

    # =========================================================
    # SAVE VIDEO (SSD)
    # =========================================================
    def save_video_clip(self, video_frames: list, video_path: Path):
        try:
            if not video_frames:
                return False

            h, w = video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))

            for frame in video_frames:
                out.write(frame)

            out.release()

            print(f"Video saved: {video_path}")
            return True

        except Exception as e:
            print(f"Video save failed: {e}")
            return False

    # =========================================================
    # CLOSE
    # =========================================================
    def close(self):
        if self.client:
            self.client.close()


# =========================================================
# SINGLETON HANDLER
# =========================================================
_handler = None


def load_local_env():
    env_candidates = [
        Path(__file__).resolve().parents[1] / ".env",
        Path.cwd() / ".env",
    ]

    for env_path in env_candidates:
        if not env_path.exists():
            continue

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

        return env_path

    return None


def get_mongo_handler(
    mongo_uri: str = None,
    db_name: str = "smart_crossing_db",
    collection_name: str = "violations"
):
    global _handler

    if _handler is None:
        env_path = load_local_env()
        resolved_mongo_uri = mongo_uri or os.getenv("SMART_ZEBRA_MONGO_URI", "").strip()
        if not resolved_mongo_uri:
            if env_path is not None:
                print(
                    f"MongoDB support disabled: SMART_ZEBRA_MONGO_URI is missing in {env_path}."
                )
            else:
                print("MongoDB support disabled: SMART_ZEBRA_MONGO_URI is not set.")
            return None

        resolved_db_name = os.getenv("SMART_ZEBRA_MONGO_DB", db_name).strip() or db_name
        resolved_collection_name = (
            os.getenv("SMART_ZEBRA_MONGO_COLLECTION", collection_name).strip() or collection_name
        )

        _handler = MongoHandler(
            resolved_mongo_uri,
            resolved_db_name,
            resolved_collection_name
        )
        _handler.connect()

    return _handler


# =========================================================
# CONVENIENCE WRAPPER
# =========================================================
def save_violation(*args, **kwargs):
    handler = get_mongo_handler()
    return handler.save_violation(*args, **kwargs)
