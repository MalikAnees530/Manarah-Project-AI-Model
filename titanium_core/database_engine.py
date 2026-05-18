import qdrant_client
from qdrant_client.models import Distance, VectorParams
class TitaniumDB:
    def __init__(self):
        self.client = qdrant_client.QdrantClient(path="qdrant_local_db")
        self.collection_name = "titanium_gallery_v1"
    def setup_collection(self):
        try: self.client.delete_collection(self.collection_name)
        except Exception: pass
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "face": VectorParams(size=512, distance=Distance.COSINE),
                "body": VectorParams(size=768, distance=Distance.COSINE),
            },
        )
