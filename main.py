from concurrent.futures import thread
from vector_database.ingestion import QdrantVectorDatabaseService
from dotenv import load_dotenv, get_key
from uuid import uuid4
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from postgres_database.postgress_database import PostgreSQLManager
from graph.agents.react_agent import ReactAgentManager
load_dotenv()

sqlManager = PostgreSQLManager()
if get_key(".env","POSTGRES_SETUP") == "true":
    sqlManager._setup_database()
agent = ReactAgentManager(sqlManager.get_checkpointer())
react_agent_graph = agent.create_agent()

def document_upload_flow():
    collection_name = "demo_collection"
    vector_db_service = QdrantVectorDatabaseService()

    if collection_name in vector_db_service.get_collections_names():
        print(f"'{collection_name}' koleksiyonu bulundu.")
    else:
        vector_db_service.create_collection(collection_name)
        print(f"'{collection_name}' koleksiyonu oluşturuldu.")

    file_path = get_key(".env", "DOCUMENTS_PATH") or "YOUR_DEFAULT_PDF_PATH"
    print(f"'{file_path}' dosyası yükleniyor...")
    vector_db_service.process_pdf_and_add_to_collection(file_path, collection_name)
    print("Doküman başarılı bir şekilde yüklendi.")


def agent_flow():
    thread_id = str(uuid4())
    print(f"Agent thread ID: {thread_id}")
    config: RunnableConfig = {
        "configurable":{
            "thread_id": thread_id,
            "user_name": "Taha"
        }
    }
    print("Agent başlatıldı. Çıkmak için 'exit' veya 'quit' yazın.")
    while True:
        input_text = input(">") or "exit"
        
        if input_text.lower() in ["exit", "quit"]:
            print("Çıkılıyor...")
            break
        
        user_input = {
            "messages": [HumanMessage(content=input_text)]
        }

        response = react_agent_graph.invoke(input=user_input, config=config)
        
        print("Asistan:", end=" ")
        print(response["messages"][-1].content)

if __name__ == "__main__":
    print("Doküman yüklemek mi yoksa agent çalıştırmak mı istiyorsunuz? (1/2)")
    print("1: Doküman Yükle")
    print("2: Agent Çalıştır")
    choice = input(">").strip().lower()
    if choice == "1":
        document_upload_flow()
    elif choice == "2":
        agent_flow()

    else:
        print("Geçersiz seçim.")
        
    sqlManager.cleanup_connections()