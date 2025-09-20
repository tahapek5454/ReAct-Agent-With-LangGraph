from dotenv import load_dotenv, get_key
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()
DATABASE_URL = get_key(".env","POSTGRES_CONNECTION_STRING") or ""
_pool = None

def create_connection_pool():
    return ConnectionPool(
        conninfo=DATABASE_URL,
        min_size=1,
        max_size=10,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row
        }
    )
  
def get_checkpointer():
    global _pool
    if _pool is None:
        _pool = create_connection_pool()
    return PostgresSaver(conn=_pool) # type: ignore

def cleanup_connections():
    global _pool
    if _pool:
        _pool.close()
        _pool = None


checkpointer = get_checkpointer()

try:
    if get_key(".env","POSTGRES_SETUP") == "true":
        checkpointer.setup()
        print("PostgreSQL tables setup completed")
    else:
        print("PostgreSQL setup skipped if you want to setup tables set POSTGRES_SETUP=true in .env file")
except Exception as e:
    print(f"Setup error (tables might already exist): {e}")