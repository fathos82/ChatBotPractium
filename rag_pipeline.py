import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

os.environ['GOOGLE_API_KEY'] = "AIzaSyBrRpwr6q1ta0vpFkNi95_7JLXf0ivy_9M"

class RagPipeline:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def from_config(self, config):
        pass

    def index_documents(self, document_files, username, doc_type_option):
        documents = self.__load_documents(document_files)
        # 🔥 Salva baseado no usuário
        # Apenas um test
        #
        # if doc_type_option.lower() == 'info':
        #     info_db = self.__save_in_database(documents, 'info')
        #     return self.load_existing_vector_store(username).merge_from(info_db)
        #
        # print("OIEIIEISDU")
        # print("OIEIIEISDU")
        # info_db = self.load_existing_vector_store('info')
        # print("01")
        return self.__save_in_database(documents, username)

    def __load_documents(self, documents):
        docs = []
        for pdf in documents:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf.read())
                loader = PyPDFLoader(tmp.name)

                pages = loader.load()
                docs.extend(pages)

        print(f"Total de páginas carregadas: {len(docs)}")
        return docs

    def __save_in_database(self, documents, username):
        username = username.strip().lower().replace(" ", "_")
        save_path = os.path.join("vector_dbs", username)

        vector_store = None
        faiss_index_path = os.path.join(save_path, "index.faiss")

        # Se já existe FAISS → carregar
        if os.path.exists(faiss_index_path):
            print(f"🔎 Carregando base existente de {username}...")

            vector_store = FAISS.load_local(
                save_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )

            # Adicionar os novos documentos
            vector_store.add_documents(documents)

        else:
            # Se não existe FAISS, precisa criar a pasta (agora sim)
            os.makedirs(save_path, exist_ok=True)

            print(f"📁 Criando nova base FAISS para {username}...")
            vector_store = FAISS.from_documents(
                documents,
                self.embedding_model
            )

        # Salvar sempre no final
        vector_store.save_local(save_path)
        print(f"💾 Banco FAISS salvo em {save_path}")

        return vector_store

    # 🔥 Função extra — carregar FAISS do usuário se já existir
    def load_existing_vector_store(self, username):
        username = username.strip().lower().replace(" ", "_")
        save_path = os.path.join("vector_dbs", username)

        os.makedirs(save_path, exist_ok=True)

        index_path = os.path.join(save_path, "index.faiss")

        # Se o arquivo index.faiss NÃO existe, cria FAISS vazio
        if not os.path.exists(index_path):
            print(f"📁 Criando FAISS vazio para '{username}'...")
            vector_store = FAISS.from_documents([], self.embedding_model)
            vector_store.save_local(save_path)
            return vector_store

        # Se existe, carrega normalmente
        print(f"🔎 Carregando banco existente '{username}'...")
        return FAISS.load_local(
            save_path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )


rag_pipeline = RagPipeline()
