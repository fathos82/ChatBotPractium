import os
from glob import glob
from operator import itemgetter

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableBranch
from langchain_google_genai import ChatGoogleGenerativeAI


os.environ['GOOGLE_API_KEY'] = "AIzaSyBrRpwr6q1ta0vpFkNi95_7JLXf0ivy_9M"


class QueryPipeline:

    def __init__(self, vector_db, llm=None, prompts_path="./res/prompts"):
        self.vector_store = vector_db
        self.llm = llm or ChatGoogleGenerativeAI(model="gemini-flash-latest")
        self.memory = InMemoryChatMessageHistory()
        self._last_topic = ''

        self.prompts = self._load_prompts(prompts_path)
        self._build_prompt_templates()
        self._build_retrievers()
        self._build_chains()
        self._build_router()


        self.chain = self.setup | self.router | self.llm | StrOutputParser()

    # ------------------------------------------------------------
    # LOAD PROMPTS
    # ------------------------------------------------------------
    def _load_prompts(self, path):
        prompts = {}
        for p in glob(f"{path}/*.txt"):
            name = os.path.splitext(os.path.basename(p))[0]
            with open(p, "r") as f:
                prompts[name] = f.read()
        return prompts

    # ------------------------------------------------------------
    # PROMPT TEMPLATES
    # ------------------------------------------------------------
    def _build_prompt_templates(self):

        def get_topic(x):
            self._last_topic = x
            return x

        self.default_prompt = ChatPromptTemplate.from_template(self.prompts["default"])
        self.soil_prompt = ChatPromptTemplate.from_template(self.prompts["analise_solo"])
        self.tissue_prompt = ChatPromptTemplate.from_template(self.prompts["analise_tecido_vegetal"])
        self.corrective_prompt = ChatPromptTemplate.from_template(self.prompts["recomendacao_correcao"])
        self.foliar_prompt = ChatPromptTemplate.from_template(self.prompts["recomendacao_foliares"])
        self.maintenance_prompt = ChatPromptTemplate.from_template(self.prompts["recomendacao_manutencao"])

        self.classifier_prompt = PromptTemplate.from_template("""
        
Classifique a intenção da próxima pergunta considerando também o contexto recente.

Contexto recente (últimas palavras da resposta anterior):
{last_ai_words}


Pergunta do usuário:
{query}
Topico Anterior:
{last_topic}

IMPORTANTE:
Se a última resposta da IA e a nova pergunta do usuário indicarem continuação do mesmo assunto
(exemplos: "sim", "continue", "aprofundar", "detalhe", "aprofundar em X", "fale mais", "compare", "explique"),
então mantenha o mesmo tópico da resposta anterior, mesmo que a nova pergunta seja curta ou ambígua.

Classifique a pergunta em **uma única** das categorias abaixo:

- analise_solo
- analise_tecido_vegetal
- recomendacao_correcao
- recomendacao_manutencao
- recomendacao_foliares

Regras:
1. Se envolver mais de um tópico, retorne `default`
2. Se for ambígua **e não houver indicação clara de continuidade**, retorne `default`
3. Se for claramente continuação da conversa anterior, **mantenha o mesmo tópico**

Retorne SOMENTE o nome da categoria.
Topic:
""")
        def print_template(x):
            print(x)
            return x

        self.classifier_chain = self.classifier_prompt | print_template| self.llm | StrOutputParser() | get_topic

    # ------------------------------------------------------------
    # RETRIEVERS
    # ------------------------------------------------------------
    def _create_retriever(self, filter):
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10, "filter": filter}
        )

    def _build_retrievers(self):
        self.default_ret = self._create_retriever({})
        self.soil_ret = self._create_retriever({"type": "analise_solo"})
        self.tissue_ret = self._create_retriever({"type": "analise_tecido_vegetal"})
        self.corrective_ret = self._create_retriever({"type": "recomendacao_correcao"})
        self.maintenance_ret = self._create_retriever({"type": "recomendacao_manutencao"})
        self.foliar_ret = self._create_retriever({"type": "recomendacao_foliares"})

    @staticmethod
    def join_documents(input):
        input["context"] = "\n\n".join([c.page_content for c in input["context"]])
        return input

    # ------------------------------------------------------------
    # FUNÇÃO AUXILIAR PARA ROTAS
    def get_context_window(self, x):
        max_words=30
        if not self.memory.messages:
            return ""

        last_ai = self.memory.messages[-1].content
        if self.memory.messages[-1].type != "ai":
            return ""

        count = 0
        end = len(last_ai)
        i = end - 1

        while i >= 0 and count < max_words:
            if last_ai[i].isspace():
                count += 1
            i -= 1
        return last_ai[i + 1:].strip()

    # ------------------------------------------------------------
    def build_route_chain(self, retriever, prompt):

        def use_retriever(x):
            return retriever.invoke(x["query"])

        def print_topic(x):
            print("TOPIC:", x["topic"])
            return x

        return (
            print_topic
            | RunnableParallel({
                "query": lambda x: x["query"],
                "history": lambda x: x["history"],
                "context": use_retriever
            })
            | self.join_documents
            | prompt
        )

    # ------------------------------------------------------------
    # BUILD CHAINS DE CADA TIPO
    # ------------------------------------------------------------
    def _build_chains(self):
        self.default_chain = self.build_route_chain(self.default_ret, self.default_prompt)
        self.soil_chain = self.build_route_chain(self.soil_ret, self.soil_prompt)
        self.tissue_chain = self.build_route_chain(self.tissue_ret, self.tissue_prompt)
        self.corrective_chain = self.build_route_chain(self.corrective_ret, self.corrective_prompt)
        self.maintenance_chain = self.build_route_chain(self.maintenance_ret, self.maintenance_prompt)
        self.foliar_chain = self.build_route_chain(self.foliar_ret, self.foliar_prompt)
        from langchain_core.runnables import RunnableMap

        self.setup = (
                RunnableParallel({
                    "query": RunnablePassthrough(),
                    "history": lambda _: "\n".join(m.content for m in self.memory.messages),
                    "last_ai_words": self.get_context_window,
                    "last_topic": lambda x: self._last_topic
                })
                | RunnableMap({
            "query": itemgetter("query"),
            "history": itemgetter("history"),
            "topic": self.classifier_chain,
        })
        )

    # ------------------------------------------------------------
    # ROUTER
    # ------------------------------------------------------------


    def _build_router(self):

        self.router =   RunnableBranch(

            (lambda x: x["topic"] == "analise_solo", self.soil_chain),
            (lambda x: x["topic"] == "analise_tecido_vegetal", self.tissue_chain),
            (lambda x: x["topic"] == "recomendacao_correcao", self.corrective_chain),
            (lambda x: x["topic"] == "recomendacao_manutencao", self.maintenance_chain),
            (lambda x: x["topic"] == "recomendacao_foliares", self.foliar_chain),
            self.default_chain,
        )

    # ------------------------------------------------------------
    # INVOKE FINAL
    # ------------------------------------------------------------
    def invoke(self, query: str):
        response = self.chain.invoke(query)
        self.memory.add_user_message(query)
        self.memory.add_ai_message(response)

        return response

    @classmethod
    def from_vector_db(cls, vector_db, llm=None):
        return cls(vector_db=vector_db, llm=llm)
