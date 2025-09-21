from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from datetime import datetime
from vector_database.ingestion import QdrantVectorDatabaseService

@tool
def calculate(expression: str) -> str:
    """
    Matematiksel ifadeleri hesaplar. Güvenli matematik hesaplamaları için kullanılır.
    
    Args:
        expression: Hesaplanacak matematiksel ifade (örn: "2 + 3 * 4")
    
    Returns:
        Hesaplama sonucu
    """
    try:
        print("calculate tool çağrıldı.")
        print("*"*20)
        allowed_chars = set('0123456789+-*/().^ ')
        if not all(c in allowed_chars for c in expression.replace(' ', '')):
            return "Hata: Sadece sayılar ve temel matematiksel operatörler (+, -, *, /, **, (), ^) kullanılabilir"
        
        expression = expression.replace('^', '**')
        
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Hesaplama hatası: {str(e)}"

@tool
def get_current_time() -> str:
    """
    Şu anki tarih ve saati getirir.
    
    Returns:
        Mevcut tarih ve saat bilgisi
    """
    print("get_current_time tool çağrıldı.")
    print("*"*20)
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

@tool(response_format="content_and_artifact")
def search_documents(query: str, collection_name: str, k: int = 5):
    """
    Qdrant vector database'inden belirtilen sorguya göre ilgili dokümanları getirir.
    Bu tool RAG (Retrieval-Augmented Generation) için kullanılır.
    
    Args:
        query: Aranacak sorgu metni
        collection_name: Qdrant'taki koleksiyon adı
        k: Getirmek istenen doküman sayısı (varsayılan: 5)
    
    Returns:
        tuple: (formatted_content, raw_documents)
            - formatted_content: Agent'ın okuyacağı formatlanmış metin
            - raw_documents: Ham Document objeler (artifact olarak)
    """
    try:
        print("search_documents tool çağrıldı.")
        print("*"*20)
        
        vector_db_service = QdrantVectorDatabaseService()
        
        retriever = vector_db_service.get_retriever(collection_name)
        
        if retriever is None:
            error_msg = f"Hata: '{collection_name}' koleksiyonu bulunamadı veya erişilemiyor."
            return error_msg, []
        
        retriever.search_kwargs = {"k": k}
    
        docs = retriever.get_relevant_documents(query)
        
        if not docs:
            no_results_msg = f"'{query}' sorgusu için '{collection_name}' koleksiyonunda ilgili doküman bulunamadı."
            return no_results_msg, []
        
        serialized = "\n\n---\n\n".join(
            (f"Source: {doc.metadata}\n Content: {doc.page_content}")
            for doc in docs
        )

        return serialized, docs
        
    except Exception as e:
        error_msg = f"RAG arama hatası: {str(e)}"
        return error_msg, []

@tool
def list_available_collections() -> str:
    """
    Qdrant vector database'inde mevcut koleksiyonları listeler.
    
    Returns:
        Mevcut koleksiyonların listesi
    """
    try:
        print("list_available_collections tool çağrıldı.")
        print("*"*20)
        vector_db_service = QdrantVectorDatabaseService()
        collectionNames = vector_db_service.get_collections_names()
        
        return f"Mevcut koleksiyonlar: {', '.join(collectionNames)}"
        
    except Exception as e:
        return f"Koleksiyonları listelerken hata: {str(e)}"

tools = [
    calculate,
    get_current_time,
    search_documents,
    list_available_collections,
]

tool_node = ToolNode(tools)