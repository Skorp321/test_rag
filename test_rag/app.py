import streamlit as st
import os
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from bs4 import BeautifulSoup


load_dotenv()

class DocumentAgent:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            openai_api_base="https://10f9698e-46b7-4a33-be37-f6495989f01f.modelrun.inference.cloud.ru/v1",
            model="qwen3:32b",
            temperature=0.1
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large"
        )
        self.vectorstore = None
        self.qa_chain = None
        
    def load_html_document(self, html_content):
        """Загружает и обрабатывает HTML документ"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            doc = Document(page_content=text, metadata={"source": "uploaded_html"})
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            documents = text_splitter.split_documents([doc])
            bm25 = BM25Retriever.from_documents(documents)
            bm25.k = 4
            faiss = FAISS.from_documents(documents, self.embeddings)
            retriever = faiss.as_retriever(
                search_type="similarity",
                k=4,
                score_threshold=None,
            )

            self.vectorstore = EnsembleRetriever(
                retrievers=[bm25, retriever],
                weights=[
                    0.5,
                    0.5,
                ],
            )
            
            self._create_qa_chain()
            
            return True, f"Документ успешно загружен и обработан. Создано {len(documents)} фрагментов."
            
        except Exception as e:
            return False, f"Ошибка при обработке документа: {str(e)}"
    
    def _create_qa_chain(self):
        """Создает цепочку вопрос-ответ"""
        template = """Используйте следующий контекст для ответа на вопрос. 
        Если вы не знаете ответа, скажите, что не знаете, не пытайтесь придумать ответ.
        Отвечайте на том же языке, на котором задан вопрос.

        Контекст: {context}

        Вопрос: {question}

        Ответ:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def ask_question(self, question):
        """Задает вопрос агенту"""
        if not self.qa_chain:
            return "Ошибка: Документ не загружен. Пожалуйста, загрузите HTML документ."
        
        try:
            result = self.qa_chain({"query": question})
            answer = result["result"]
            sources = result["source_documents"]
            return answer, sources
        except Exception as e:
            return f"Ошибка при обработке вопроса: {str(e)}", []

def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if 'agent' not in st.session_state:
        try:
            st.session_state.agent = DocumentAgent(openai_api_key)
        except Exception as e:
            st.error(f"Ошибка инициализации агента: {str(e)}")
            st.stop()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'document_loaded' not in st.session_state:
        st.session_state.document_loaded = False

    st.set_page_config(
        page_title="Document QA Agent",
        layout="wide"
    )

    st.markdown("<h2 style='text-align: center; color: white;'>Document QA Agent</h2>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: white;'>Агент для ответов на вопросы по содержанию HTML документов</h6>", unsafe_allow_html=True)

    with st.sidebar:

        st.markdown("<h2 style='text-align: center; color: white;'>Загрузка документа</h2>", unsafe_allow_html=True)

        upload_option = st.radio(
            "Выберите способ загрузки файла:",
            ("Загрузить свой файл", "Использовать файл по умолчанию")
        )
        
        default_file_path = "test_rag/doc.html"
        html_content = None
        
        if upload_option == "Загрузить свой файл":
            uploaded_file = st.file_uploader(
                "Выберите HTML файл",
                type=['html'],
                help="Загрузите HTML документ для анализа"
            )
            if uploaded_file is not None:
                html_content = uploaded_file.read().decode('utf-8')
        else:
            if os.path.exists(default_file_path):
                with open(default_file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                st.success(f"Загружен файл по умолчанию: {default_file_path}")
            else:
                st.error(f"Файл по умолчанию не найден: {default_file_path}")
        
        if html_content and st.button("Обработать документ"):
            with st.spinner("Обработка документа..."):
                if hasattr(st.session_state, 'agent'):
                    success, message = st.session_state.agent.load_html_document(html_content)
                    if success:
                        st.success(message)
                        st.session_state.document_loaded = True
                    else:
                        st.error(message)
                        st.session_state.document_loaded = False
                else:
                    st.error("Агент не инициализирован. Пожалуйста, перезагрузите страницу.")

    st.markdown("<h2 style='text-align: center; color: white;'>Чат с агентом</h2>", unsafe_allow_html=True)

    if not hasattr(st.session_state, 'document_loaded') or not st.session_state.document_loaded:
        st.info("Пожалуйста, загрузите и обработайте HTML документ для начала работы.")
        return

    chat_container = st.container()
    
    with chat_container:
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)

        question = st.chat_input("Задайте вопрос по документу...")
        
        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Думаю..."):
                    answer, sources = st.session_state.agent.ask_question(question)
                    st.write(answer)
                    st.session_state.chat_history.append((question, answer))

                    if sources and len(sources) > 0:
                        with st.expander("📖 Источники"):
                            for i, source in enumerate(sources):
                                st.text_area(
                                    f"Источник {i+1}:",
                                    source.page_content,
                                    height=100
                                )

if __name__ == "__main__":
    main()