from dotenv import load_dotenv, get_key
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

# Not: Db yi olusturmaniz lazim ornek 'chatbot_db' adinda bir db olusturun

load_dotenv()
class PostgreSQLManager:
    def __init__(self):
        self.database_url = get_key(".env","POSTGRES_CONNECTION_STRING") or ""
        self._checkpointer = None
        self._setup_completed = False
        self._pool = None
         
    def __del__(self):
        self.cleanup_connections()
        
    def create_connection_pool(self):
        """Create and return a PostgreSQL connection pool"""
        print("Creating PostgreSQL connection pool...")
        return ConnectionPool(
            conninfo=self.database_url,
            min_size=1,
            max_size=10,
            kwargs={
                "autocommit": True,
                "row_factory": dict_row
            }
        )
    
    def get_checkpointer(self):
        """Get or create PostgreSQL checkpointer"""
        if self._pool is None:
            self._pool = self.create_connection_pool()
        
        if self._checkpointer is None:
            self._checkpointer = PostgresSaver(conn=self._pool)  # type: ignore

        return self._checkpointer
    
    def _setup_database(self):
        """Setup database tables if needed (runs only once)"""        
        try:
            if get_key(".env","POSTGRES_SETUP") == "true":
                self.get_checkpointer().setup()
                print("PostgreSQL tables setup completed")
            else:
                print("PostgreSQL setup skipped if you want to setup tables set POSTGRES_SETUP=true in .env file")
        except Exception as e:
            print(f"Setup error (tables might already exist): {e}")
    
    def cleanup_connections(self):
        """Close connection pool and cleanup resources"""
        if self._pool:
            try:
                print("Closing PostgreSQL connection pool...")
                self._pool.close()
                
            except Exception as e:
                print(f"Error during pool cleanup: {e}")
            finally:
                self._pool = None

        self._checkpointer = None
        self._setup_completed = False
