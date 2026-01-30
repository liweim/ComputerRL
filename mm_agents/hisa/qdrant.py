from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct, 
    Filter, 
    FieldCondition,
    FilterSelector
)
from typing import List, Dict, Any, Optional
import os
import glob
import shutil
from .embedding import EmbeddingClient
import json

class QdrantManager:
    """
    A manager class for Qdrant vector database operations.
    Supports CRUD operations on collections and points.
    """
    
    def __init__(self, path: str = "../qdrant_storage", use_memory: bool = False,
                 use_server: bool = False, server_url: str = "http://localhost:6333"):
        """
        Initialize Qdrant client.

        Args:
            path: Path to store data on disk (ignored if use_memory=True or use_server=True)
            use_memory: If True, use in-memory storage (data lost on restart)
            use_server: If True, connect to Qdrant server (supports multi-process)
            server_url: Qdrant server URL (default: http://localhost:6333)
        """
        if use_memory:
            self.client = QdrantClient(":memory:")
        elif use_server:
            self.client = QdrantClient(url=server_url)
        else:
            self.client = QdrantClient(path=path)
    
    # ==================== Collection Operations ====================
    
    def create_collection(
        self, 
        collection_name: str, 
        vector_size: int, 
        distance: str = "Cosine"
    ) -> bool:
        """
        Create a new collection.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            distance: Distance metric ("Cosine", "Euclid", "Dot")
        
        Returns:
            True if successful
        """
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT
        }
        
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size, 
                distance=distance_map.get(distance, Distance.COSINE)
            )
        )
        return True
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
        
        Returns:
            True if successful
        """
        self.client.delete_collection(collection_name=collection_name)
        return True
    
    def list_collections(self) -> List[str]:
        """
        List all collection names.
        
        Returns:
            List of collection names
        """
        collections = self.client.get_collections()
        return [col.name for col in collections.collections]
    
    def get_collection_info(self, collection_name: str) -> Dict:
        """
        Get detailed information about a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary containing collection info
        """
        info = self.client.get_collection(collection_name=collection_name)
        return info.model_dump()
    
    # ==================== Insert/Update Operations ====================
    
    def insert_points(
        self, 
        collection_name: str, 
        points: List[Dict[str, Any]]
    ) -> bool:
        """
        Insert or update points in a collection.
        
        Args:
            collection_name: Name of the collection
            points: List of points, each containing:
                - id: Unique identifier
                - vector: List of floats
                - payload: Dictionary of metadata (optional)
        
        Example:
            points = [
                {"id": 1, "vector": [0.1, 0.2, ...], "payload": {"name": "item1"}},
                {"id": 2, "vector": [0.3, 0.4, ...], "payload": {"name": "item2"}}
            ]
        
        Returns:
            True if successful
        """
        point_structs = [
            PointStruct(
                id=point["id"],
                vector=point["vector"],
                payload=point.get("payload", {})
            )
            for point in points
        ]
        
        self.client.upsert(
            collection_name=collection_name,
            points=point_structs
        )
        return True
    
    def update_payload(
        self, 
        collection_name: str, 
        point_ids: List[int], 
        payload: Dict[str, Any]
    ) -> bool:
        """
        Update payload for specific points (keeps existing fields).
        
        Args:
            collection_name: Name of the collection
            point_ids: List of point IDs to update
            payload: Dictionary of fields to add/update
        
        Returns:
            True if successful
        """
        self.client.set_payload(
            collection_name=collection_name,
            payload=payload,
            points=point_ids
        )
        return True
    
    def overwrite_payload(
        self, 
        collection_name: str, 
        point_ids: List[int], 
        payload: Dict[str, Any]
    ) -> bool:
        """
        Overwrite payload for specific points (replaces all fields).
        
        Args:
            collection_name: Name of the collection
            point_ids: List of point IDs to update
            payload: Dictionary of new payload
        
        Returns:
            True if successful
        """
        self.client.overwrite_payload(
            collection_name=collection_name,
            payload=payload,
            points=point_ids
        )
        return True
    
    def delete_payload_fields(
        self, 
        collection_name: str, 
        point_ids: List[int], 
        keys: List[str]
    ) -> bool:
        """
        Delete specific fields from payload.
        
        Args:
            collection_name: Name of the collection
            point_ids: List of point IDs
            keys: List of field names to delete
        
        Returns:
            True if successful
        """
        self.client.delete_payload(
            collection_name=collection_name,
            keys=keys,
            points=point_ids
        )
        return True
    
    # ==================== Query Operations ====================
    
    def search(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar vectors.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Optional filter (see filter_points for format)
        
        Returns:
            List of dictionaries containing id, score, vector, and payload
        """
        query_filter = None
        if filter_conditions:
            query_filter = self._build_filter(filter_conditions)
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter
        )
        
        return [
            {
                "id": result.id,
                "score": result.score,
                "vector": result.vector,
                "payload": result.payload
            }
            for result in results
        ]
    
    def search_by_filter(
        self,
        collection_name: str,
        filter_conditions: Dict[str, Any],
        limit: int = 100,
        with_vectors: bool = False
    ) -> List[Dict]:
        """
        Search points by filter conditions without vector search.
        
        Args:
            collection_name: Name of the collection
            filter_conditions: Simple dict with field:value pairs (e.g., {"type": "require"})
            limit: Maximum number of results
            with_vectors: Include vectors in response
        
        Returns:
            List of dictionaries containing id, payload, and optionally vector
        """
        # Build filter from simple field:value dict
        must_conditions = []
        for key, value in filter_conditions.items():
            must_conditions.append({"key": key, "match": {"value": value}})
        
        filter_obj = self._build_filter({"must": must_conditions})
        
        # Use scroll with filter to get all matching points
        all_results = []
        offset = None
        
        while True:
            results, offset = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_obj,
                limit=min(limit, 100),  # Batch size
                with_vectors=with_vectors,
                offset=offset
            )
            
            all_results.extend([
                {
                    "id": result.id,
                    "vector": result.vector if with_vectors else None,
                    "payload": result.payload,
                    "score": 1.0  # No score for filter-only search
                }
                for result in results
            ])
            
            # Break if we've got enough or no more results
            if len(all_results) >= limit or offset is None:
                break
        
        return all_results[:limit]

    def retrieve_by_ids(
        self, 
        collection_name: str, 
        ids: List[int],
        with_vectors: bool = True
    ) -> List[Dict]:
        """
        Retrieve points by their IDs.
        
        Args:
            collection_name: Name of the collection
            ids: List of point IDs
            with_vectors: Include vectors in response
        
        Returns:
            List of points
        """
        results = self.client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_vectors=with_vectors
        )
        
        return [
            {
                "id": result.id,
                "vector": result.vector if with_vectors else None,
                "payload": result.payload
            }
            for result in results
        ]
    
    def scroll_all(
        self, 
        collection_name: str, 
        limit: int = 100,
        with_vectors: bool = False
    ) -> List[Dict]:
        """
        Retrieve all points from a collection (paginated).
        
        Args:
            collection_name: Name of the collection
            limit: Number of points per page
            with_vectors: Include vectors in response
        
        Returns:
            List of all points
        """
        all_points = []
        offset = None
        
        while True:
            results, offset = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_vectors=with_vectors
            )
            
            all_points.extend([
                {
                    "id": point.id,
                    "vector": point.vector if with_vectors else None,
                    "payload": point.payload
                }
                for point in results
            ])
            
            if offset is None:
                break
        
        return all_points
    
    def filter_points(
        self, 
        collection_name: str, 
        filter_conditions: Dict[str, Any],
        limit: int = 100
    ) -> List[Dict]:
        """
        Filter points by payload conditions.
        
        Args:
            collection_name: Name of the collection
            filter_conditions: Dictionary of conditions, e.g.:
                {
                    "must": [{"key": "age", "range": {"gte": 18}}],
                    "should": [{"key": "city", "match": {"value": "NYC"}}],
                    "must_not": [{"key": "status", "match": {"value": "banned"}}]
                }
            limit: Maximum number of results
        
        Returns:
            List of filtered points
        """
        query_filter = self._build_filter(filter_conditions)
        
        results, _ = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_vectors=False
        )
        
        return [
            {
                "id": point.id,
                "payload": point.payload
            }
            for point in results
        ]
    
    # ==================== Delete Operations ====================
    
    def delete_by_ids(
        self, 
        collection_name: str, 
        ids: List[int]
    ) -> bool:
        """
        Delete points by their IDs.
        
        Args:
            collection_name: Name of the collection
            ids: List of point IDs to delete
        
        Returns:
            True if successful
        """
        self.client.delete(
            collection_name=collection_name,
            points_selector=ids
        )
        return True
    
    def delete_by_filter(
        self, 
        collection_name: str, 
        filter_conditions: Dict[str, Any]
    ) -> bool:
        """
        Delete points matching filter conditions.
        
        Args:
            collection_name: Name of the collection
            filter_conditions: Dictionary of conditions (same format as filter_points)
        
        Returns:
            True if successful
        """
        query_filter = self._build_filter(filter_conditions)
        
        self.client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(filter=query_filter)
        )
        return True
    
    # ==================== Helper Methods ====================
    
    def _build_filter(self, conditions: Dict[str, Any]) -> Filter:
        """
        Build a Qdrant filter from condition dictionary.
        
        Args:
            conditions: Dictionary with must/should/must_not keys
        
        Returns:
            Filter object
        """
        must = []
        should = []
        must_not = []
        
        for cond in conditions.get("must", []):
            must.append(self._build_field_condition(cond))
        
        for cond in conditions.get("should", []):
            should.append(self._build_field_condition(cond))
        
        for cond in conditions.get("must_not", []):
            must_not.append(self._build_field_condition(cond))
        
        return Filter(must=must or None, should=should or None, must_not=must_not or None)
    
    def _build_field_condition(self, cond: Dict[str, Any]) -> FieldCondition:
        """
        Build a field condition from dictionary.
        
        Args:
            cond: Condition dictionary with 'key' and 'match'/'range'
        
        Returns:
            FieldCondition object
        """
        key = cond["key"]
        
        if "match" in cond:
            return FieldCondition(key=key, match=cond["match"])
        elif "range" in cond:
            return FieldCondition(key=key, range=cond["range"])
        else:
            raise ValueError(f"Invalid condition: {cond}")
    
    def count_points(self, collection_name: str) -> int:
        """
        Count total number of points in a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Number of points
        """
        info = self.client.get_collection(collection_name=collection_name)
        return info.points_count

    def view_collection_content(
        self,
        collection_name: str,
        limit: Optional[int] = None,
        show_info: bool = True
    ) -> List[Dict]:
        """
        View and display content of a collection.

        Args:
            collection_name: Name of the collection
            limit: Maximum number of points to display (None = all)
            with_vectors: Include vector data in output
            show_info: Print collection info to console

        Returns:
            List of points with their data

        Example usage:
            manager = QdrantManager(path="./qdrant_storage")
            points = manager.view_collection_content("lessons_general", limit=10)
        """
        try:
            # Get collection info
            if show_info:
                info = self.get_collection_info(collection_name)
                print(f"\n{'='*80}")
                print(f"Collection: {collection_name}")
                print(f"{'='*80}")
                print(f"Total points: {info['points_count']}")
                print(f"Vector size: {info['config']['params']['vectors']['size']}")
                print(f"Distance: {info['config']['params']['vectors']['distance']}")
                print(f"{'='*80}\n")

            # Get all points (or limited)
            if limit:
                results, _ = self.client.scroll(
                    collection_name=collection_name,
                    limit=limit,
                )
                points = [
                    {
                        "id": point.id,
                        "payload": point.payload
                    }
                    for point in results
                ]
            else:
                points = self.scroll_all(collection_name)

            # Display points
            if show_info:
                print(f"Displaying {len(points)} point(s):\n")
                for i, point in enumerate(points, 1):
                    print(f"[{i}] ID: {point['id']}")
                    for key, value in point['payload'].items():
                        # Truncate long text for display
                        print(f"  - {key}: {value}")
                    print()
            return points

        except Exception as e:
            print(f"Error viewing collection '{collection_name}': {e}")
            raise

    def import_from_json(
        self,
        json_file: str,
        collection_name: str,
        embedding_client,
        vector_size: int = 1024,
        batch_size: int = 32
    ) -> int:
        """
        Import lessons from JSON file to Qdrant collection.

        Args:
            json_file: Path to JSON file containing lessons
            collection_name: Name of the target collection
            embedding_client: Callable that takes text and returns vector (e.g., EmbeddingClient instance)
            vector_size: Dimension of vectors (default: 1024 for bge-large-en-v1.5)
            batch_size: Batch size for embedding generation

        Returns:
            Number of lessons imported

        JSON format expected:
        [
            {"id": 1, "type": "success", "lesson": "lesson text"},
            {"id": 2, "type": "failure", "lesson": "lesson text"},
            ...
        ]
        """
        import json

        # Read JSON file
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                lessons = json.load(f)
        except Exception as e:
            raise Exception(f"Failed to read JSON file: {e}")

        if not isinstance(lessons, list):
            raise ValueError("JSON file must contain a list of lessons")

        # Ensure collection exists
        collections = self.list_collections()
        if collection_name not in collections:
            self.create_collection(
                collection_name=collection_name,
                vector_size=vector_size,
                distance="Cosine"
            )
            print(f"Created collection: {collection_name}")

        # Process lessons in batches
        imported_count = 0
        for i in range(0, len(lessons), batch_size):
            batch = lessons[i:i+batch_size]

            # Extract lesson texts
            lesson_texts = []
            for lesson in batch:
                if "lesson" in lesson and lesson["lesson"]:
                    lesson_texts.append(lesson["lesson"])
                else:
                    lesson_texts.append("")  # Empty placeholder

            # Generate embeddings for batch
            embeddings = embedding_client(lesson_texts)

            # Prepare points for insertion
            points = []
            for lesson, embedding in zip(batch, embeddings):
                if not lesson.get("lesson"):
                    continue

                point = {
                    "id": lesson.get("id", i + batch.index(lesson) + 1),
                    "vector": embedding,
                    "payload": {
                        "lesson": lesson.get("lesson", ""),
                        "type": lesson.get("type", "failure")
                    }
                }
                points.append(point)

            # Insert points
            if points:
                try:
                    self.insert_points(collection_name, points)
                    imported_count += len(points)
                except Exception as e:
                    print(f"Warning: Failed to insert batch {i}: {e}")

        print(f"Total imported: {imported_count} lessons into collection '{collection_name}'")
        return imported_count


def rebuild_collections(dir_path: str = "../qdrant_storage", use_server=True, server_url="http://localhost:6333"):
    """Rebuild all collections from JSON files.

    Args:
        use_server: If True, use Qdrant server mode (recommended for multi-process)
        server_url: Qdrant server URL
    """
    # Initialize embedding client
    embedding_client = EmbeddingClient("http://localhost:8000")

    if use_server:
        # Server mode: delete collections via API (don't delete storage folder)
        manager = QdrantManager(use_server=True, server_url=server_url)

        # Delete all existing collections
        existing_collections = manager.list_collections()
        for coll in existing_collections:
            print(f"Deleting collection: {coll}")
            manager.delete_collection(coll)
    else:
        # Local mode: delete storage folder and recreate
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        manager = QdrantManager(path=dir_path)

    # Import lessons from JSON files
    for json_path in glob.glob("patterns/*.json"):
        collection_name = os.path.basename(json_path).split(".")[0]

        imported = manager.import_from_json(
            json_file=json_path,
            collection_name=collection_name,
            embedding_client=embedding_client,
            vector_size=1024,
            batch_size=32
        )

    manager.client.close()
    print("Collections rebuilt successfully!")


def add_lessons_to_existing(
    json_file: str,
    collection_name: str,
    path: str = "../qdrant_storage",
    use_server: bool = False,
    server_url: str = "http://localhost:6333",
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Add new lessons to an existing collection.
    
    Args:
        json_file: Path to JSON file
        collection_name: Target collection name
        path: Local storage path
        use_server: Whether to use server mode
        server_url: Server URL
        batch_size: Batch size for processing
        
    Returns:
        Dictionary containing import statistics
    """
    # Initialize
    embedding_client = EmbeddingClient("http://localhost:8000")
    if use_server:
        manager = QdrantManager(use_server=True, server_url=server_url)
    else:
        manager = QdrantManager(path=path)
    
    # Check if collection exists
    collections = manager.list_collections()
    if collection_name not in collections:
        manager.create_collection(collection_name, 1024, "Cosine")
    
    # Get next available ID and existing lessons
    info = manager.get_collection_info(collection_name)
    points_count = info['points_count']
    
    # Get existing lessons to check for duplicates
    existing_lessons = set()
    if points_count > 0:
        all_points = manager.scroll_all(collection_name, limit=10000, with_vectors=False)
        start_id = max(point['id'] for point in all_points) + 1
        # Extract existing lesson texts
        for point in all_points:
            lesson_text = point.get('payload', {}).get('lesson', '')
            if lesson_text:
                existing_lessons.add(lesson_text.strip())
    else:
        start_id = 1
    
    print(f"Collection '{collection_name}' currently has {points_count} points")
    print(f"Existing {len(existing_lessons)} different lessons")
    print(f"New data will start from ID {start_id}")
    
    # Read JSON
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            lessons = json.load(f)
    except Exception as e:
        raise Exception(f"Failed to read JSON file: {e}")
    
    if not isinstance(lessons, list):
        raise ValueError("JSON file must contain a list")
    
    # Statistics
    stats = {
        "total_in_json": len(lessons),
        "successfully_added": 0,
        "skipped": 0,
        "duplicate": 0,
        "failed": 0,
        "start_id": start_id,
        "end_id": start_id
    }
    
    current_id = start_id
    
    # Process in batches
    for i in range(0, len(lessons), batch_size):
        batch = lessons[i:i+batch_size]
        
        # Extract lesson texts and check for duplicates
        lesson_texts = []
        valid_lessons = []
        
        for lesson in batch:
            lesson_text = lesson.get("lesson", "")
            if lesson_text and lesson_text.strip():
                lesson_text_stripped = lesson_text.strip()
                # Check if lesson already exists
                if lesson_text_stripped in existing_lessons:
                    stats["duplicate"] += 1
                    continue
                lesson_texts.append(lesson_text)
                valid_lessons.append(lesson)
                # Add to existing set to avoid duplicates within the same batch
                existing_lessons.add(lesson_text_stripped)
            else:
                stats["skipped"] += 1
        
        if not lesson_texts:
            continue
        
        # Generate embeddings
        try:
            embeddings = embedding_client(lesson_texts)
        except Exception as e:
            print(f"Batch {i} failed to generate embeddings: {e}")
            stats["failed"] += len(valid_lessons)
            continue
        
        # Prepare points for insertion
        points = []
        for lesson, embedding in zip(valid_lessons, embeddings):
            point = {
                "id": current_id,
                "vector": embedding,
                "payload": {
                    "lesson": lesson.get("lesson", ""),
                    "type": lesson.get("type", "failure"),
                    "original_id": lesson.get("id")
                }
            }
            points.append(point)
            current_id += 1
        
        # Insert points
        try:
            manager.insert_points(collection_name, points)
            stats["successfully_added"] += len(points)
            print(f"Successfully added batch {i//batch_size + 1}: {len(points)} items")
        except Exception as e:
            print(f"Batch {i} insertion failed: {e}")
            stats["failed"] += len(points)
    
    stats["end_id"] = current_id - 1
    
    # Print statistics
    print("\n" + "="*60)
    print(f"Import completed!")
    print(f"Collection name: {collection_name}")
    print(f"JSON file: {json_file}")
    print(f"Total: {stats['total_in_json']}")
    print(f"Successfully added: {stats['successfully_added']}")
    print(f"Skipped (empty content): {stats['skipped']}")
    print(f"Duplicate (already exists): {stats['duplicate']}")
    print(f"Failed: {stats['failed']}")
    print(f"ID range: {stats['start_id']} - {stats['end_id']}")
    print(f"New total points in collection: {manager.count_points(collection_name)}")
    print("="*60)
    
    manager.client.close()
    return stats


if __name__ == "__main__":
    # ===== Method 1: Rebuild all collections (from scratch) =====
    # rebuild_collections(use_server=True, server_url="http://localhost:6333")
    # rebuild_collections(dir_path="D:/projects/qdrant/hsa_gpt5_gta1_50_roi_pattern", use_server=False)
    
    # # ===== Method 2: Add new data to existing collection =====
    for json_file in glob.glob("patterns/*.json"):
        collection_name = os.path.basename(json_file).split(".")[0]
        add_lessons_to_existing(
            json_file=json_file,
            collection_name=collection_name,
            use_server=True,
            # path="D:/projects/qdrant/hsa_gpt5mini_gta1_50_roi_pattern"
        )
    # add_lessons_to_existing(
    #     json_file="patterns/servicenow.json",
    #     collection_name="servicenow",
    #     use_server=True,
    #     # path="D:/projects/qdrant/hsa_gpt5mini_gta1_50_roi_pattern"
    # )
    
    # ===== View collection content =====
    # manager = QdrantManager(
    #     use_server=False,
    #     path="D:/projects/qdrant/hsa_gpt5mini_gta1_50_roi_pattern"
    # )
    # manager.view_collection_content("excel")
    # manager.client.close()