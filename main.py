import streamlit as st
import time

# Tenta importar os pipelines.
# O try/except é apenas para garantir que o código não quebre visualmente se os arquivos não existirem no momento.
try:
    from query_pipeline import QueryPipeline
    from rag_pipeline import rag_pipeline
except ImportError:
    # Se você ainda não criou esses arquivos, o código usará mocks (simulações) onde possível
    QueryPipeline = None
    rag_pipeline = None

# Configuração da Página
st.set_page_config(
    page_title="Chatbot RAG com PDFs",
    page_icon="📚",
    layout="wide"
)


def main():
    # --------------------------------------------------------------------------
    # 0. TELA DE LOGIN (Autenticação Simples)
    # --------------------------------------------------------------------------
    if "username" not in st.session_state:
        st.session_state.username = None

    # Se o usuário não estiver "logado", mostra apenas o formulário de entrada
    if st.session_state.username is None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title("🔐 Acesso ao Sistema")
            with st.form("login_form"):
                st.write("Por favor, identifique-se para iniciar a sessão.")
                name_input = st.text_input("Nome de Usuário")
                submit_login = st.form_submit_button("Entrar no Chat")

                if submit_login:
                    if name_input.strip():
                        st.session_state.username = name_input
                        st.rerun()  # Recarrega a página para mostrar o app principal
                    else:
                        st.warning("Por favor, digite um nome válido.")
        return  # Interrompe a execução do restante da função main() até o login

    # --------------------------------------------------------------------------
    # 1. ESTADO DA SESSÃO (Session State)
    # --------------------------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": f"Olá, **{st.session_state.username}**! Selecione o tipo de documento e carregue seus PDFs na barra lateral."}
        ]

    if "is_processed" not in st.session_state:
        st.session_state.is_processed = False

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None

    if "doc_class" not in st.session_state:
        st.session_state.doc_class = None

    # --------------------------------------------------------------------------
    # 2. BARRA LATERAL (Sidebar)
    # --------------------------------------------------------------------------
    with st.sidebar:
        # Informações do Usuário
        st.write(f"👤 Usuário: **{st.session_state.username}**")
        if st.button("Sair / Trocar Usuário"):
            st.session_state.username = None
            st.session_state.messages = []
            st.session_state.is_processed = False
            st.session_state.vector_store = None
            st.session_state.query_engine = None
            st.rerun()

        st.divider()
        st.header("📂 Configuração")

        # --- SELEÇÃO DE CLASSIFICAÇÃO ---
        st.subheader("1. Classificação")
        doc_type_option = st.selectbox(
            "Selecione o tipo de documento:",
            [
                "Geral",
                "Recomendação de Corretivos",
                "Análise de Solo",
                "Relatório Técnico",
                "Manual de Operação",
                "Outros",
                "Info"
            ],
            index=0
        )

        st.subheader("2. Upload")
        st.write("Faça upload dos seus arquivos PDF aqui:")

        pdf_docs = st.file_uploader(
            "Carregar PDFs",
            accept_multiple_files=True,
            type=['pdf']
        )

        process_button = st.button("Processar Documentos", type="primary")

        if process_button and pdf_docs:
            with st.spinner(f"Processando documentos do tipo '{doc_type_option}'..."):

                # Armazena a classificação
                st.session_state.doc_class = doc_type_option

                # Chama o pipeline de indexação (se existir)
                if rag_pipeline:
                    try:
                        vector_store = rag_pipeline.index_documents(pdf_docs, st.session_state.username, doc_type_option)
                        st.session_state.vector_store = vector_store
                        st.session_state.is_processed = True
                        st.success("Documentos processados com sucesso!")
                    except Exception as e:
                        st.error(f"Erro ao processar documentos: {e}")
                else:
                    # Fallback para teste de interface se o pipeline não estiver importado
                    time.sleep(2)
                    st.session_state.is_processed = True
                    st.session_state.vector_store = "MockVectorStore"
                    st.success("Documentos processados (Simulação)!")

        elif process_button and not pdf_docs:
            st.warning("Por favor, carregue pelo menos um arquivo PDF.")

        st.divider()

        if st.button("Limpar Conversa"):
            st.session_state.messages = []
            # Reinicia com saudação personalizada
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Olá novamente, **{st.session_state.username}**! O histórico foi limpo."
            })
            st.rerun()

    # --------------------------------------------------------------------------
    # 3. ÁREA PRINCIPAL DO CHAT
    # --------------------------------------------------------------------------
    st.title("🤖 Chatbot Inteligente (RAG)")

    # Header informativo
    if st.session_state.is_processed and st.session_state.doc_class:
        st.caption(f"Contexto: **{st.session_state.doc_class}** | Usuário: {st.session_state.username}")
    else:
        st.caption(f"Bem-vindo, {st.session_state.username}. Inicie carregando seus documentos.")

    # Exibir histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Captura da entrada do usuário
    if prompt := st.chat_input("Faça uma pergunta sobre os documentos..."):

        # 1. Exibir mensagem do usuário
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. Gerar resposta do assistente
        with st.chat_message("assistant"):

            response_placeholder = st.empty()
            full_response = ""

            if not st.session_state.is_processed:
                full_response = "⚠️ Por favor, carregue e processe os documentos na barra lateral antes de fazer perguntas."
                response_placeholder.markdown(full_response)
            else:
                with st.spinner("Consultando base de conhecimento..."):
                    try:
                        # Se tivermos o pipeline real importado
                        if QueryPipeline and st.session_state.vector_store:
                            # Cria o pipeline de query baseado no store atual
                            if st.session_state.query_engine is None:
                                st.session_state.query_engine = QueryPipeline.from_vector_db(st.session_state.vector_store)
                            full_response = st.session_state.query_engine.invoke(prompt)
                        else:
                            # Simulação caso esteja testando apenas a UI
                            time.sleep(1.5)
                            full_response = f"Simulação de resposta para **{st.session_state.username}**: O sistema encontrou informações relevantes em '{st.session_state.doc_class}' sobre '{prompt}'."

                    except Exception as e:
                        full_response = f"❌ Erro ao consultar a base: {str(e)}"
                        raise e

                response_placeholder.markdown(full_response)

        # 3. Salvar resposta no histórico
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
