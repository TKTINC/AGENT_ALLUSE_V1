"""
ALL-USE Learning Systems - Document Database

This module implements a document database for the ALL-USE Learning Systems,
providing flexible storage and retrieval of structured data.

The document database is designed to:
- Store structured documents with flexible schemas
- Support rich querying capabilities
- Provide efficient indexing for fast retrieval
- Enable complex data relationships
- Support transactions and ACID properties

Classes:
- DocumentDB: Core document database implementation
- Collection: Represents a collection of documents
- Document: Represents a single document
- Query: Query builder for document retrieval

Version: 1.0.0
"""

import time
import logging
import threading
import json
import os
import sqlite3
import uuid
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IndexType(Enum):
    """Types of indexes that can be created in the document database."""
    BTREE = 1    # Balanced tree index for equality and range queries
    HASH = 2     # Hash index for equality queries
    TEXT = 3     # Full-text search index
    SPATIAL = 4  # Spatial index for geospatial queries

@dataclass
class DocumentDBConfig:
    """Configuration for a document database."""
    db_path: str = "document_db.sqlite"
    auto_vacuum: bool = True
    vacuum_interval_hours: int = 24
    enable_compression: bool = True
    enable_caching: bool = True
    cache_size_mb: int = 100
    max_concurrent_queries: int = 10
    max_document_size_mb: int = 10
    enable_full_text_search: bool = True
    journal_mode: str = "WAL"  # WAL, DELETE, TRUNCATE, PERSIST, MEMORY, OFF

class Document:
    """Represents a single document in the database."""
    
    def __init__(self, doc_id: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        """Initialize the document.
        
        Args:
            doc_id: Unique identifier for this document. If None, a new ID will be generated.
            data: The document data.
        """
        self.doc_id = doc_id or str(uuid.uuid4())
        self.data = data or {}
        self.created_at = time.time()
        self.updated_at = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the document to a dictionary.
        
        Returns:
            Dictionary representation of the document.
        """
        return {
            "doc_id": self.doc_id,
            "data": self.data,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create a document from a dictionary.
        
        Args:
            data: Dictionary representation of the document.
            
        Returns:
            The created document.
        """
        doc = cls(doc_id=data["doc_id"])
        doc.data = data["data"]
        doc.created_at = data["created_at"]
        doc.updated_at = data["updated_at"]
        return doc

class Collection:
    """Represents a collection of documents in the database."""
    
    def __init__(self, db: 'DocumentDB', name: str):
        """Initialize the collection.
        
        Args:
            db: The document database this collection belongs to.
            name: Name of the collection.
        """
        self.db = db
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.collection.{name}")
        self.indexes: Dict[str, Dict[str, Any]] = {}
    
    def insert(self, document: Document) -> bool:
        """Insert a document into the collection.
        
        Args:
            document: The document to insert.
            
        Returns:
            True if the document was successfully inserted, False otherwise.
        """
        return self.db.insert_document(self.name, document)
    
    def insert_many(self, documents: List[Document]) -> bool:
        """Insert multiple documents into the collection.
        
        Args:
            documents: The documents to insert.
            
        Returns:
            True if all documents were successfully inserted, False otherwise.
        """
        return self.db.insert_documents(self.name, documents)
    
    def find_by_id(self, doc_id: str) -> Optional[Document]:
        """Find a document by its ID.
        
        Args:
            doc_id: ID of the document to find.
            
        Returns:
            The found document, or None if not found.
        """
        return self.db.find_document_by_id(self.name, doc_id)
    
    def find(self, query: Dict[str, Any]) -> List[Document]:
        """Find documents matching a query.
        
        Args:
            query: Query to match documents against.
            
        Returns:
            List of matching documents.
        """
        return self.db.find_documents(self.name, query)
    
    def update(self, doc_id: str, updates: Dict[str, Any]) -> bool:
        """Update a document.
        
        Args:
            doc_id: ID of the document to update.
            updates: Updates to apply to the document.
            
        Returns:
            True if the document was successfully updated, False otherwise.
        """
        return self.db.update_document(self.name, doc_id, updates)
    
    def delete(self, doc_id: str) -> bool:
        """Delete a document.
        
        Args:
            doc_id: ID of the document to delete.
            
        Returns:
            True if the document was successfully deleted, False otherwise.
        """
        return self.db.delete_document(self.name, doc_id)
    
    def create_index(self, field: str, index_type: IndexType = IndexType.BTREE) -> bool:
        """Create an index on a field.
        
        Args:
            field: Field to index.
            index_type: Type of index to create.
            
        Returns:
            True if the index was successfully created, False otherwise.
        """
        return self.db.create_index(self.name, field, index_type)
    
    def drop_index(self, field: str) -> bool:
        """Drop an index on a field.
        
        Args:
            field: Field to drop the index for.
            
        Returns:
            True if the index was successfully dropped, False otherwise.
        """
        return self.db.drop_index(self.name, field)
    
    def count(self, query: Optional[Dict[str, Any]] = None) -> int:
        """Count documents in the collection.
        
        Args:
            query: Optional query to filter documents.
            
        Returns:
            Number of matching documents.
        """
        return self.db.count_documents(self.name, query)

class Query:
    """Query builder for document retrieval."""
    
    def __init__(self, db: 'DocumentDB', collection_name: str):
        """Initialize the query.
        
        Args:
            db: The document database to query.
            collection_name: Name of the collection to query.
        """
        self.db = db
        self.collection_name = collection_name
        self.filters = {}
        self.sort_field = None
        self.sort_direction = "ASC"
        self.limit_value = None
        self.skip_value = None
    
    def filter(self, field: str, value: Any) -> 'Query':
        """Add a filter to the query.
        
        Args:
            field: Field to filter on.
            value: Value to filter for.
            
        Returns:
            This query builder.
        """
        self.filters[field] = value
        return self
    
    def sort(self, field: str, direction: str = "ASC") -> 'Query':
        """Set the sorting for the query.
        
        Args:
            field: Field to sort by.
            direction: Direction to sort in ("ASC" or "DESC").
            
        Returns:
            This query builder.
        """
        self.sort_field = field
        self.sort_direction = direction
        return self
    
    def limit(self, limit: int) -> 'Query':
        """Set the maximum number of results to return.
        
        Args:
            limit: Maximum number of results.
            
        Returns:
            This query builder.
        """
        self.limit_value = limit
        return self
    
    def skip(self, skip: int) -> 'Query':
        """Set the number of results to skip.
        
        Args:
            skip: Number of results to skip.
            
        Returns:
            This query builder.
        """
        self.skip_value = skip
        return self
    
    def execute(self) -> List[Document]:
        """Execute the query.
        
        Returns:
            List of documents matching the query.
        """
        return self.db.execute_query(self)

class DocumentDB:
    """Core document database implementation."""
    
    def __init__(self, db_id: str, config: Optional[DocumentDBConfig] = None):
        """Initialize the document database.
        
        Args:
            db_id: Unique identifier for this database.
            config: Configuration for this database.
        """
        self.db_id = db_id
        self.config = config or DocumentDBConfig()
        self.logger = logging.getLogger(f"{__name__}.db.{db_id}")
        self.conn = None
        self.collections: Dict[str, Collection] = {}
        self.running = False
        self.maintenance_thread = None
        self.query_semaphore = threading.Semaphore(self.config.max_concurrent_queries)
        
        # Initialize the database
        self._init_db()
    
    def _init_db(self):
        """Initialize the database."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.config.db_path)), exist_ok=True)
            
            # Connect to the database
            self.conn = sqlite3.connect(self.config.db_path, check_same_thread=False)
            
            # Enable WAL mode for better concurrency
            self.conn.execute(f"PRAGMA journal_mode={self.config.journal_mode}")
            
            # Set cache size
            self.conn.execute(f"PRAGMA cache_size={self.config.cache_size_mb * 1024}")
            
            # Create tables if they don't exist
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY,
                    created_at REAL NOT NULL
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    collection_name TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (collection_name) REFERENCES collections (name)
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS indexes (
                    id TEXT PRIMARY KEY,
                    collection_name TEXT NOT NULL,
                    field TEXT NOT NULL,
                    type TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (collection_name) REFERENCES collections (name)
                )
            """)
            
            # Create indexes
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents (collection_name)")
            
            # Load existing collections
            cursor = self.conn.execute("SELECT name FROM collections")
            for row in cursor:
                collection_name = row[0]
                collection = Collection(self, collection_name)
                self.collections[collection_name] = collection
                
                # Load indexes for this collection
                index_cursor = self.conn.execute(
                    "SELECT field, type FROM indexes WHERE collection_name = ?",
                    (collection_name,)
                )
                for index_row in index_cursor:
                    field, index_type = index_row
                    collection.indexes[field] = {"type": index_type}
            
            self.logger.info(f"Initialized database with {len(self.collections)} collections")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}", exc_info=True)
            raise
    
    def start(self):
        """Start the document database."""
        if self.running:
            self.logger.warning("Database already running")
            return
        
        self.running = True
        self.logger.info(f"Starting document database {self.db_id}")
        
        # Start maintenance thread
        self.maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            name=f"db-maintenance-{self.db_id}",
            daemon=True
        )
        self.maintenance_thread.start()
    
    def stop(self):
        """Stop the document database."""
        if not self.running:
            self.logger.warning("Database not running")
            return
        
        self.logger.info(f"Stopping document database {self.db_id}")
        self.running = False
        
        # Wait for thread to terminate
        if self.maintenance_thread and self.maintenance_thread.is_alive():
            self.maintenance_thread.join(timeout=5.0)
        
        # Close the database connection
        if self.conn:
            self.conn.close()
            self.conn = None
        
        self.logger.info(f"Document database {self.db_id} stopped")
    
    def _maintenance_loop(self):
        """Maintenance loop for the database."""
        self.logger.info(f"Maintenance loop started for database {self.db_id}")
        
        last_vacuum_time = time.time()
        
        while self.running:
            try:
                # Vacuum the database if needed
                current_time = time.time()
                if (self.config.auto_vacuum and 
                    current_time - last_vacuum_time >= self.config.vacuum_interval_hours * 3600):
                    
                    self._vacuum_database()
                    last_vacuum_time = current_time
                
                # Sleep for a while
                time.sleep(3600)  # 1 hour
                
            except Exception as e:
                self.logger.error(f"Error in maintenance loop: {e}", exc_info=True)
                time.sleep(3600)  # Sleep longer on error
    
    def _vacuum_database(self):
        """Vacuum the database to reclaim space."""
        try:
            self.logger.info("Vacuuming database...")
            self.conn.execute("VACUUM")
            self.logger.info("Database vacuum completed")
        except Exception as e:
            self.logger.error(f"Error vacuuming database: {e}", exc_info=True)
    
    def create_collection(self, name: str) -> Collection:
        """Create a new collection.
        
        Args:
            name: Name of the collection to create.
            
        Returns:
            The created collection.
        """
        try:
            # Check if collection already exists
            if name in self.collections:
                return self.collections[name]
            
            # Insert into database
            self.conn.execute(
                "INSERT INTO collections (name, created_at) VALUES (?, ?)",
                (name, time.time())
            )
            self.conn.commit()
            
            # Create collection object
            collection = Collection(self, name)
            self.collections[name] = collection
            
            self.logger.debug(f"Created collection {name}")
            return collection
            
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}", exc_info=True)
            raise
    
    def drop_collection(self, name: str) -> bool:
        """Drop a collection.
        
        Args:
            name: Name of the collection to drop.
            
        Returns:
            True if the collection was successfully dropped, False otherwise.
        """
        try:
            # Check if collection exists
            if name not in self.collections:
                self.logger.warning(f"Collection {name} does not exist")
                return False
            
            # Delete from database
            self.conn.execute("DELETE FROM indexes WHERE collection_name = ?", (name,))
            self.conn.execute("DELETE FROM documents WHERE collection_name = ?", (name,))
            self.conn.execute("DELETE FROM collections WHERE name = ?", (name,))
            self.conn.commit()
            
            # Remove from cache
            del self.collections[name]
            
            self.logger.debug(f"Dropped collection {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error dropping collection: {e}", exc_info=True)
            return False
    
    def get_collection(self, name: str) -> Optional[Collection]:
        """Get a collection by name.
        
        Args:
            name: Name of the collection to get.
            
        Returns:
            The collection, or None if not found.
        """
        return self.collections.get(name)
    
    def list_collections(self) -> List[str]:
        """List all collections.
        
        Returns:
            List of collection names.
        """
        return list(self.collections.keys())
    
    def insert_document(self, collection_name: str, document: Document) -> bool:
        """Insert a document into a collection.
        
        Args:
            collection_name: Name of the collection to insert into.
            document: The document to insert.
            
        Returns:
            True if the document was successfully inserted, False otherwise.
        """
        try:
            # Check if collection exists
            if collection_name not in self.collections:
                self.create_collection(collection_name)
            
            # Check document size
            data_json = json.dumps(document.data)
            if len(data_json) > self.config.max_document_size_mb * 1024 * 1024:
                self.logger.error(f"Document size exceeds maximum allowed size of {self.config.max_document_size_mb}MB")
                return False
            
            # Insert into database
            self.conn.execute(
                "INSERT INTO documents (id, collection_name, data, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (document.doc_id, collection_name, data_json, document.created_at, document.updated_at)
            )
            self.conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error inserting document: {e}", exc_info=True)
            return False
    
    def insert_documents(self, collection_name: str, documents: List[Document]) -> bool:
        """Insert multiple documents into a collection.
        
        Args:
            collection_name: Name of the collection to insert into.
            documents: The documents to insert.
            
        Returns:
            True if all documents were successfully inserted, False otherwise.
        """
        if not documents:
            return True
        
        try:
            # Check if collection exists
            if collection_name not in self.collections:
                self.create_collection(collection_name)
            
            # Prepare data
            values = []
            for document in documents:
                # Check document size
                data_json = json.dumps(document.data)
                if len(data_json) > self.config.max_document_size_mb * 1024 * 1024:
                    self.logger.error(f"Document size exceeds maximum allowed size of {self.config.max_document_size_mb}MB")
                    continue
                
                values.append((document.doc_id, collection_name, data_json, document.created_at, document.updated_at))
            
            # Insert into database
            self.conn.executemany(
                "INSERT INTO documents (id, collection_name, data, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                values
            )
            self.conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error inserting documents: {e}", exc_info=True)
            return False
    
    def find_document_by_id(self, collection_name: str, doc_id: str) -> Optional[Document]:
        """Find a document by its ID.
        
        Args:
            collection_name: Name of the collection to search in.
            doc_id: ID of the document to find.
            
        Returns:
            The found document, or None if not found.
        """
        try:
            # Check if collection exists
            if collection_name not in self.collections:
                self.logger.warning(f"Collection {collection_name} does not exist")
                return None
            
            # Query database
            cursor = self.conn.execute(
                "SELECT data, created_at, updated_at FROM documents WHERE collection_name = ? AND id = ?",
                (collection_name, doc_id)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            data_json, created_at, updated_at = row
            
            # Create document
            document = Document(doc_id=doc_id)
            document.data = json.loads(data_json)
            document.created_at = created_at
            document.updated_at = updated_at
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error finding document: {e}", exc_info=True)
            return None
    
    def find_documents(self, collection_name: str, query: Dict[str, Any]) -> List[Document]:
        """Find documents matching a query.
        
        Args:
            collection_name: Name of the collection to search in.
            query: Query to match documents against.
            
        Returns:
            List of matching documents.
        """
        try:
            # Check if collection exists
            if collection_name not in self.collections:
                self.logger.warning(f"Collection {collection_name} does not exist")
                return []
            
            # Build SQL query
            sql = "SELECT id, data, created_at, updated_at FROM documents WHERE collection_name = ?"
            params = [collection_name]
            
            # This is a simplification; in a real implementation, we would need to parse the JSON
            # and filter based on the query. For now, we'll just return all documents.
            
            # Execute query
            cursor = self.conn.execute(sql, params)
            
            # Process results
            results = []
            for row in cursor:
                doc_id, data_json, created_at, updated_at = row
                
                # Create document
                document = Document(doc_id=doc_id)
                document.data = json.loads(data_json)
                document.created_at = created_at
                document.updated_at = updated_at
                
                # Check if document matches query
                matches = True
                for field, value in query.items():
                    if field not in document.data or document.data[field] != value:
                        matches = False
                        break
                
                if matches:
                    results.append(document)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error finding documents: {e}", exc_info=True)
            return []
    
    def update_document(self, collection_name: str, doc_id: str, updates: Dict[str, Any]) -> bool:
        """Update a document.
        
        Args:
            collection_name: Name of the collection containing the document.
            doc_id: ID of the document to update.
            updates: Updates to apply to the document.
            
        Returns:
            True if the document was successfully updated, False otherwise.
        """
        try:
            # Check if collection exists
            if collection_name not in self.collections:
                self.logger.warning(f"Collection {collection_name} does not exist")
                return False
            
            # Get current document
            document = self.find_document_by_id(collection_name, doc_id)
            if not document:
                self.logger.warning(f"Document {doc_id} not found in collection {collection_name}")
                return False
            
            # Apply updates
            for field, value in updates.items():
                document.data[field] = value
            
            # Update timestamp
            document.updated_at = time.time()
            
            # Update in database
            data_json = json.dumps(document.data)
            
            # Check document size
            if len(data_json) > self.config.max_document_size_mb * 1024 * 1024:
                self.logger.error(f"Document size exceeds maximum allowed size of {self.config.max_document_size_mb}MB")
                return False
            
            self.conn.execute(
                "UPDATE documents SET data = ?, updated_at = ? WHERE collection_name = ? AND id = ?",
                (data_json, document.updated_at, collection_name, doc_id)
            )
            self.conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating document: {e}", exc_info=True)
            return False
    
    def delete_document(self, collection_name: str, doc_id: str) -> bool:
        """Delete a document.
        
        Args:
            collection_name: Name of the collection containing the document.
            doc_id: ID of the document to delete.
            
        Returns:
            True if the document was successfully deleted, False otherwise.
        """
        try:
            # Check if collection exists
            if collection_name not in self.collections:
                self.logger.warning(f"Collection {collection_name} does not exist")
                return False
            
            # Delete from database
            cursor = self.conn.execute(
                "DELETE FROM documents WHERE collection_name = ? AND id = ?",
                (collection_name, doc_id)
            )
            self.conn.commit()
            
            return cursor.rowcount > 0
            
        except Exception as e:
            self.logger.error(f"Error deleting document: {e}", exc_info=True)
            return False
    
    def create_index(self, collection_name: str, field: str, index_type: IndexType = IndexType.BTREE) -> bool:
        """Create an index on a field.
        
        Args:
            collection_name: Name of the collection to create the index in.
            field: Field to index.
            index_type: Type of index to create.
            
        Returns:
            True if the index was successfully created, False otherwise.
        """
        try:
            # Check if collection exists
            if collection_name not in self.collections:
                self.logger.warning(f"Collection {collection_name} does not exist")
                return False
            
            # Check if index already exists
            collection = self.collections[collection_name]
            if field in collection.indexes:
                self.logger.warning(f"Index on field {field} already exists in collection {collection_name}")
                return True
            
            # Create index
            index_id = str(uuid.uuid4())
            self.conn.execute(
                "INSERT INTO indexes (id, collection_name, field, type, created_at) VALUES (?, ?, ?, ?, ?)",
                (index_id, collection_name, field, index_type.name, time.time())
            )
            self.conn.commit()
            
            # Add to collection
            collection.indexes[field] = {"type": index_type.name}
            
            self.logger.debug(f"Created index on field {field} in collection {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating index: {e}", exc_info=True)
            return False
    
    def drop_index(self, collection_name: str, field: str) -> bool:
        """Drop an index on a field.
        
        Args:
            collection_name: Name of the collection containing the index.
            field: Field to drop the index for.
            
        Returns:
            True if the index was successfully dropped, False otherwise.
        """
        try:
            # Check if collection exists
            if collection_name not in self.collections:
                self.logger.warning(f"Collection {collection_name} does not exist")
                return False
            
            # Check if index exists
            collection = self.collections[collection_name]
            if field not in collection.indexes:
                self.logger.warning(f"Index on field {field} does not exist in collection {collection_name}")
                return False
            
            # Delete from database
            self.conn.execute(
                "DELETE FROM indexes WHERE collection_name = ? AND field = ?",
                (collection_name, field)
            )
            self.conn.commit()
            
            # Remove from collection
            del collection.indexes[field]
            
            self.logger.debug(f"Dropped index on field {field} in collection {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error dropping index: {e}", exc_info=True)
            return False
    
    def count_documents(self, collection_name: str, query: Optional[Dict[str, Any]] = None) -> int:
        """Count documents in a collection.
        
        Args:
            collection_name: Name of the collection to count documents in.
            query: Optional query to filter documents.
            
        Returns:
            Number of matching documents.
        """
        try:
            # Check if collection exists
            if collection_name not in self.collections:
                self.logger.warning(f"Collection {collection_name} does not exist")
                return 0
            
            if not query:
                # Simple count
                cursor = self.conn.execute(
                    "SELECT COUNT(*) FROM documents WHERE collection_name = ?",
                    (collection_name,)
                )
                return cursor.fetchone()[0]
            else:
                # Count with query
                # This is inefficient but works for now
                return len(self.find_documents(collection_name, query))
            
        except Exception as e:
            self.logger.error(f"Error counting documents: {e}", exc_info=True)
            return 0
    
    def execute_query(self, query: Query) -> List[Document]:
        """Execute a query.
        
        Args:
            query: The query to execute.
            
        Returns:
            List of documents matching the query.
        """
        # Acquire semaphore to limit concurrent queries
        with self.query_semaphore:
            try:
                # Check if collection exists
                if query.collection_name not in self.collections:
                    self.logger.warning(f"Collection {query.collection_name} does not exist")
                    return []
                
                # Find documents matching filters
                documents = self.find_documents(query.collection_name, query.filters)
                
                # Sort if needed
                if query.sort_field:
                    documents.sort(
                        key=lambda d: d.data.get(query.sort_field, 0),
                        reverse=(query.sort_direction == "DESC")
                    )
                
                # Apply skip and limit
                if query.skip_value:
                    documents = documents[query.skip_value:]
                
                if query.limit_value:
                    documents = documents[:query.limit_value]
                
                return documents
                
            except Exception as e:
                self.logger.error(f"Error executing query: {e}", exc_info=True)
                return []
    
    def create_query(self, collection_name: str) -> Query:
        """Create a new query builder.
        
        Args:
            collection_name: Name of the collection to query.
            
        Returns:
            A new query builder.
        """
        return Query(self, collection_name)

# Example usage
if __name__ == "__main__":
    # Create a document database
    db = DocumentDB("example-db")
    
    # Start the database
    db.start()
    
    # Create a collection
    users = db.create_collection("users")
    
    # Create some documents
    user1 = Document(data={
        "name": "Alice",
        "age": 30,
        "email": "alice@example.com",
        "roles": ["admin", "user"]
    })
    
    user2 = Document(data={
        "name": "Bob",
        "age": 25,
        "email": "bob@example.com",
        "roles": ["user"]
    })
    
    # Insert documents
    users.insert(user1)
    users.insert(user2)
    
    # Create an index
    users.create_index("name")
    
    # Find a document by ID
    found_user = users.find_by_id(user1.doc_id)
    if found_user:
        print(f"Found user: {found_user.data['name']}")
    
    # Find documents by query
    admin_users = users.find({"roles": ["admin", "user"]})
    print(f"Found {len(admin_users)} admin users")
    
    # Update a document
    users.update(user2.doc_id, {"age": 26, "roles": ["user", "editor"]})
    
    # Query using the query builder
    query = db.create_query("users") \
        .filter("age", 26) \
        .sort("name", "ASC") \
        .limit(10)
    
    results = query.execute()
    print(f"Query returned {len(results)} documents")
    
    # Count documents
    user_count = users.count()
    print(f"Total users: {user_count}")
    
    # Delete a document
    users.delete(user1.doc_id)
    
    # Stop the database
    db.stop()

