"""
Graph Executor

Executes Cypher queries against Neo4j database and returns results.
Handles connection management, error handling, and result formatting.
"""

import os
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, exceptions as neo4j_exceptions
import logging


class GraphExecutor:
    """
    Neo4j query executor.

    Manages connection to Neo4j database and executes Cypher queries.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None
    ):
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j connection URI (default: from NEO4J_URI env var)
            username: Neo4j username (default: from NEO4J_USERNAME env var)
            password: Neo4j password (default: from NEO4J_PASSWORD env var)
            database: Neo4j database name (default: from NEO4J_DATABASE env var)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")

        self.logger = logging.getLogger(__name__)

        # Initialize driver
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Test connection
            self.driver.verify_connectivity()
            self.logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def execute(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Execute Cypher query and return results.

        Args:
            query: Cypher query string
            parameters: Optional query parameters
            timeout: Query timeout in seconds

        Returns:
            List of result records as dictionaries

        Raises:
            neo4j_exceptions.CypherSyntaxError: Invalid Cypher syntax
            neo4j_exceptions.ClientError: Query execution error
            Exception: Other execution errors
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        parameters = parameters or {}

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters, timeout=timeout)
                records = [dict(record) for record in result]

                self.logger.debug(
                    f"Query executed successfully. Returned {len(records)} records."
                )

                return records

        except neo4j_exceptions.CypherSyntaxError as e:
            self.logger.error(f"Cypher syntax error: {e}")
            raise

        except neo4j_exceptions.ClientError as e:
            self.logger.error(f"Client error during query execution: {e}")
            raise

        except Exception as e:
            self.logger.error(f"Unexpected error during query execution: {e}")
            raise

    def execute_with_metadata(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Execute query and return results with metadata.

        Args:
            query: Cypher query string
            parameters: Optional query parameters
            timeout: Query timeout in seconds

        Returns:
            Dictionary with keys:
                - success: bool indicating execution success
                - records: list of result records (if successful)
                - error: error message (if failed)
                - error_type: type of error (if failed)
                - record_count: number of records returned
        """
        try:
            records = self.execute(query, parameters, timeout)

            return {
                "success": True,
                "records": records,
                "record_count": len(records),
                "error": None,
                "error_type": None
            }

        except neo4j_exceptions.CypherSyntaxError as e:
            return {
                "success": False,
                "records": [],
                "record_count": 0,
                "error": str(e),
                "error_type": "SyntaxError"
            }

        except neo4j_exceptions.ClientError as e:
            error_msg = str(e)

            # Categorize error type
            if "property" in error_msg.lower() or "attribute" in error_msg.lower():
                error_type = "PropertyError"
            elif "label" in error_msg.lower() or "node" in error_msg.lower():
                error_type = "LabelError"
            elif "relationship" in error_msg.lower():
                error_type = "RelationshipError"
            else:
                error_type = "ClientError"

            return {
                "success": False,
                "records": [],
                "record_count": 0,
                "error": error_msg,
                "error_type": error_type
            }

        except Exception as e:
            return {
                "success": False,
                "records": [],
                "record_count": 0,
                "error": str(e),
                "error_type": "UnknownError"
            }

    def test_connection(self) -> bool:
        """
        Test database connection.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    def close(self):
        """Close database connection."""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


if __name__ == "__main__":
    # Test GraphExecutor
    logging.basicConfig(level=logging.INFO)

    print("Testing GraphExecutor...")

    try:
        with GraphExecutor() as executor:
            # Test connection
            if executor.test_connection():
                print("Connection test: PASSED")
            else:
                print("Connection test: FAILED")

            # Test simple query
            print("\nTesting simple query...")
            result = executor.execute("RETURN 1 AS number")
            print(f"Result: {result}")

            # Test query with metadata
            print("\nTesting query with metadata...")
            metadata = executor.execute_with_metadata("MATCH (n) RETURN count(n) AS count")
            print(f"Metadata: {metadata}")

            # Test syntax error
            print("\nTesting syntax error...")
            metadata = executor.execute_with_metadata("INVALID QUERY")
            print(f"Error metadata: {metadata}")

    except Exception as e:
        print(f"Test failed: {e}")
