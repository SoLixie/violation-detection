import cv2
import gridfs
import numpy as np
from pymongo import MongoClient
from pathlib import Path
from datetime import datetime


class MongoHandler:
    def __init__(
        self,
        mongo_uri: str,
        db_name: str = "smart_crossing_db",
        collection_name: str = "violation"
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
        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            self.client.server_info()

            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            self.fs = gridfs.GridFS(self.db)

            print("✅ MongoDB Atlas connected")
            return True

        except Exception as e:
            print(f"⚠️ MongoDB unavailable: {e}")
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
            print("❌ Save failed: MongoDB not connected")
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

            self.collection.insert_one(doc)

            print(f"💾 Saved {vtype} | ID={track_id} | Speed={speed:.1f} km/h")
            return True

        except Exception as e:
            print(f"❌ Save failed: {e}")
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

            print(f"📹 Video saved: {video_path}")
            return True

        except Exception as e:
            print(f"❌ Video save failed: {e}")
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


def get_mongo_handler(
    mongo_uri: str = None,
    db_name: str = "smart_crossing_db",
    collection_name: str = "violation"
):
    global _handler

    if _handler is None:
        _handler = MongoHandler(
            mongo_uri or
            "mongodb+srv://zebra_backend:backend_zebra@smart-zebra-crossing-cl.kahl9t8.mongodb.net/smart_crossing_db?retryWrites=true&w=majority",
            db_name,
            collection_name
        )
        _handler.connect()

    return _handler


# =========================================================
# CONVENIENCE WRAPPER
# =========================================================
def save_violation(*args, **kwargs):
    handler = get_mongo_handler()
    return handler.save_violation(*args, **kwargs)