import weaviate

import os
import asyncio
import logging

from typing import List, Dict, Any
from weaviate import WeaviateAsyncClient
from sentence_transformers import SentenceTransformer

from weaviate.classes.data import DataObject
from weaviate.classes.query import MetadataQuery
from weaviate.classes.config import Configure, Property, DataType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class WeaviateDatabaseManager:

    def __init__(self, model_name: str = "jinaai/jina-embeddings-v3"):
        self.model = SentenceTransformer(
            model_name, 
            trust_remote_code=True,
            device="cuda:0"
        )
        self.collection_name = "Documents"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding for a batch of sentences or documents"""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    
    async def create_collect(self, async_client: WeaviateAsyncClient):
        """Create a collection with vector configuration"""
        try:
            if await async_client.collections.exists(self.collection_name):
                await async_client.collections.delete(self.collection_name)
            
            collection = await async_client.collections.create(
                name=self.collection_name,
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="app_id", data_type=DataType.TEXT),
                    Property(name="document_path", data_type=DataType.TEXT),
                ],
                vector_config=Configure.Vectors.self_provided()
            )
            logger.info(f"Collection '{self.collection_name}' created successfully")
            return collection
            
        except Exception as e:
            logger.info(f"Error creating collections: {e}")
            raise weaviate.exceptions.WeaviateBaseError("Error spawning collections.")
        
    
    async def batch_insert(self, async_client: WeaviateAsyncClient, documents: List[Dict[str, Any]]):
        """Insert documents with vectors in batches"""
        try:
            collection = async_client.collections.get(self.collection_name)
            texts_to_embed = [document["content"] for document in documents]
            logger.info(f"Generating embeddings for {len(texts_to_embed)} documents.")
            vectors = self.embed_texts(texts_to_embed)

            data_objects = []
            for i, doc in enumerate(documents):
                data_objects.append(
                    DataObject(
                        properties={
                            "content": doc["content"],
                            "app_id": doc["app_id"],
                            "document_path": doc["document_path"]
                        },
                        vector=vectors[i]
                    )
                )
            
            logger.info(f"About to insert {len(data_objects)} elements.")
            response = await collection.data.insert_many(data_objects)

            if response.has_errors:
                logger.info(f"Some objects fail to be inserted.")
                for error in response.errors:
                    logger.info(f"- Insertion Error: {error}")
            
            else:
                logger.info(f"{len(data_objects)} objects successfully inserted into {self.collection_name}!")

        except Exception as e:
            logger.info(f"Batch insert process hasn't been successful.")
            raise weaviate.exceptions.WeaviateInsertManyAllFailedError("Batch insert has failed.")

        return response
    
    async def search_database(
            self, 
            async_client: WeaviateAsyncClient, 
            query_text: str, 
            limit: int = 5,
            distance_threshold: float = 0.25):
        try:
            collection = async_client.collections.get(self.collection_name)

            query_vector = self.embed_texts([query_text])[0]

            response = await collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                distance=distance_threshold,
                return_metadata=MetadataQuery(distance=True)
            )
            
            results = []
            for obj in response.objects:
                results.append({
                    "uuid": str(obj.uuid),
                    "content": obj.properties.get("content"),
                    "app_id": obj.properties.get("app_id"),
                    "document_path": obj.properties.get("document_path"),
                    "metadata": obj.properties.get("metadata"),
                    "distance": obj.metadata.distance
                })

            logger.info(f"Found {len(results)} similar documents")
            return results
        
        except Exception as e:
            logger.info(f"There appeared an error: {str(e)}")
            return
    
async def main():

    db_manager = WeaviateDatabaseManager()

    # sample_data = [
    #     {"content": "I like strawberry coconut juice", "app_id": "egune-test", "document_path": "path/to/doc1"},
    #     {"content": "I hate strawberry coconut juice", "app_id": "egune-test", "document_path": "path/to/doc1"},
    #     {"content": "I like strawberry but not coconut juice", "app_id": "egune-test", "document_path": "path/to/doc1"},
    #     {"content": "I like strawberry and coconut but I hate juice", "app_id": "egune-test", "document_path": "path/to/doc1"}
    # ]

    # async with weaviate.use_async_with_local() as async_client:

    #     readiness = await async_client.is_ready()
    #     logger.info(f"Weaviate client ready :D -> {readiness}")
        
    #     if not readiness:
    #         logger.info("Weaviate client not ready :(")
    #         return
        
    #     await db_manager.create_collect(async_client)
    #     await db_manager.batch_insert(async_client, sample_data)
    async with weaviate.use_async_with_local() as async_client:
        results = await db_manager.search_database(
            async_client,
            "Монгол улсын иргэн хагас автомат винтов буу эзэмшхэд юу шаардлагатай вэ?",
            distance_threshold=0.4
        )
        print(f"\n\n\n")
        for result in results:
            print(result)
            print("\n\n")

if __name__ == '__main__':
    asyncio.run(main())
    