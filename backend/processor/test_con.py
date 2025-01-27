from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph
import neo4j
import os
neo4j_url = "neo4j+s://804c0d42.databases.neo4j.io"
neo4j_username = "neo4j"
neo4j_password = "8GpfujKtWTG5o2oSdOiln8fqh2xqByaO40m2sqZJz34"
os.environ["NEO4J_URL"] = neo4j_url
os.environ["NEO4J_USERNAME"] = neo4j_username
os.environ["NEO4J_PASSWORD"] = neo4j_password

try:
        graph = Neo4jGraph(
                        url=neo4j_url, 
                        username=neo4j_username, 
                        password=neo4j_password
                    )
        print("Connected to Neo4j")
        print(graph.schema)
        print(graph.query("MATCH (n) RETURN n LIMIT 10"))
except Exception as e:
    print(f"Error: {e}")

