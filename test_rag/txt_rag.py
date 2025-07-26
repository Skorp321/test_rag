import streamlit as st
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import pandas as pd
import glob
import zipfile
import xml.etree.ElementTree as ET


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
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={'device': 'cuda'}
        )
        self.vectorstore = None
        self.qa_chain = None
        
    def load_documents_from_folder(self, folder_path):
        """Загружает и обрабатывает документы из папки"""
        try:
            all_documents = []
            
            # Поиск всех поддерживаемых файлов
            supported_extensions = ['*.docx', '*.xlsx', '*.DOCX', '*.XLSX']
            files_found = []
            
            for extension in supported_extensions:
                pattern = os.path.join(folder_path, extension)
                files_found.extend(glob.glob(pattern))
            
            if not files_found:
                return False, f"В папке {folder_path} не найдено поддерживаемых файлов (.docx, .xlsx, .DOCX, .XLSX)"
            
            st.info(f"Найдено файлов: {len(files_found)}")
            
            for file_path in files_found:
                file_extension = os.path.splitext(file_path)[1].lower()
                file_name = os.path.basename(file_path)
                
                try:
                    if file_extension == '.docx':
                        text = self._extract_text_from_docx(file_path)
                    elif file_extension == '.xlsx':
                        text = self._extract_text_from_xlsx(file_path)
                    else:
                        continue
                    
                    if text.strip():
                        doc = Document(
                            page_content=text, 
                            metadata={"source": file_name, "file_path": file_path}
                        )
                        all_documents.append(doc)
                        st.success(f"Обработан файл: {file_name}")
                    else:
                        st.warning(f"Файл {file_name} не содержит текста")
                        
                except Exception as e:
                    st.error(f"Ошибка при обработке файла {file_name}: {str(e)}")
                    continue
            
            if not all_documents:
                return False, "Не удалось извлечь текст ни из одного файла"
            
            # Разбиение документов на фрагменты
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            split_documents = text_splitter.split_documents(all_documents)
            
            # Создание ретриверов
            bm25 = BM25Retriever.from_documents(split_documents)
            bm25.k = 4
            faiss = FAISS.from_documents(split_documents, self.embeddings)
            retriever = faiss.as_retriever(
                search_type="similarity",
                k=4,
                score_threshold=None,
            )

            self.vectorstore = EnsembleRetriever(
                retrievers=[bm25, retriever],
                weights=[0.5, 0.5],
            )
            
            self._create_qa_chain()
            
            return True, f"Документы успешно загружены и обработаны. Создано {len(split_documents)} фрагментов из {len(all_documents)} файлов."
            
        except Exception as e:
            return False, f"Ошибка при обработке документов: {str(e)}"
    
    def _extract_text_from_docx(self, file_path):
        """Извлекает текст из DOCX файла"""
        try:
            # DOCX файл - это ZIP архив с XML файлами
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                # Читаем document.xml, который содержит основной текст
                if 'word/document.xml' in zip_file.namelist():
                    xml_content = zip_file.read('word/document.xml')
                    root = ET.fromstring(xml_content)
                    
                    # Находим все параграфы
                    text_parts = []
                    
                    # Извлекаем текст из параграфов
                    for paragraph in root.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                        para_text = []
                        for text_elem in paragraph.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
                            if text_elem.text:
                                para_text.append(text_elem.text)
                        if para_text:
                            text_parts.append(''.join(para_text))
                    
                    # Извлекаем текст из таблиц
                    for table in root.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tbl'):
                        for row in table.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tr'):
                            row_text = []
                            for cell in row.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc'):
                                cell_text = []
                                for text_elem in cell.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
                                    if text_elem.text:
                                        cell_text.append(text_elem.text)
                                if cell_text:
                                    row_text.append(''.join(cell_text))
                            if row_text:
                                text_parts.append(" | ".join(row_text))
                    
                    return "\n".join(text_parts)
                else:
                    st.error(f"Не удалось найти document.xml в файле {file_path}")
                    return ""
                    
        except Exception as e:
            st.error(f"Ошибка при чтении DOCX файла {file_path}: {str(e)}")
            return ""
    
    def _extract_text_from_xlsx(self, file_path):
        """Извлекает текст из XLSX файла"""
        try:
            # Читаем все листы Excel файла
            excel_file = pd.ExcelFile(file_path)
            text_parts = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Добавляем название листа
                text_parts.append(f"Лист: {sheet_name}")
                
                # Добавляем заголовки
                if not df.empty:
                    headers = df.columns.tolist()
                    text_parts.append("Заголовки: " + " | ".join([str(h) for h in headers]))
                    
                    # Добавляем данные (первые 100 строк для экономии места)
                    for index, row in df.head(100).iterrows():
                        row_data = [str(cell) for cell in row.values if pd.notna(cell) and str(cell).strip()]
                        if row_data:
                            text_parts.append(" | ".join(row_data))
                
                text_parts.append("")  # Пустая строка между листами
            
            return "\n".join(text_parts)
        except Exception as e:
            st.error(f"Ошибка при чтении XLSX файла {file_path}: {str(e)}")
            return ""
    
    def _create_qa_chain(self):
        """Создает цепочку вопрос-ответ"""
        template = """Используйте следующий контекст для ответа на вопрос. 
        Если ты не знаешь ответа, скажи, что не знаешь, не пытайся придумать ответ.
        Отвечай на том же языке, на котором задан вопрос.

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
            return "Ошибка: Документы не загружены. Пожалуйста, загрузите документы из папки."
        
        try:
            result = self.qa_chain.invoke({"query": question})
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
    
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False

    st.set_page_config(
        page_title="Document Folder QA Agent",
        layout="wide"
    )

    st.markdown("<h2 style='text-align: center; color: white;'>Document Folder QA Agent</h2>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: white;'>Агент для ответов на вопросы по содержанию документов из папки</h6>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: white;'>Загрузка документов</h2>", unsafe_allow_html=True)

        # Выбор папки
        folder_option = st.radio(
            "Выберите способ загрузки документов:",
            ("Указать путь к папке", "Использовать папку по умолчанию")
        )
        
        folder_path = None
        
        if folder_option == "Указать путь к папке":
            folder_path = st.text_input(
                "Введите путь к папке с документами:",
                placeholder="/path/to/documents",
                help="Укажите полный путь к папке, содержащей .docx и .xlsx файлы"
            )
        else:
            # Папка по умолчанию - создаем в текущей директории
            default_folder = "documents"
            if not os.path.exists(default_folder):
                os.makedirs(default_folder)
                st.info(f"Создана папка по умолчанию: {default_folder}")
                st.info("Поместите ваши .docx и .xlsx файлы в эту папку")
            folder_path = default_folder
            st.success(f"Используется папка по умолчанию: {folder_path}")
        
        if folder_path and st.button("Обработать документы"):
            if os.path.exists(folder_path):
                with st.spinner("Обработка документов..."):
                    if hasattr(st.session_state, 'agent'):
                        success, message = st.session_state.agent.load_documents_from_folder(folder_path)
                        if success:
                            st.success(message)
                            st.session_state.documents_loaded = True
                        else:
                            st.error(message)
                            st.session_state.documents_loaded = False
                    else:
                        st.error("Агент не инициализирован. Пожалуйста, перезагрузите страницу.")
            else:
                st.error(f"Папка не найдена: {folder_path}")

    st.markdown("<h2 style='text-align: center; color: white;'>Чат с агентом</h2>", unsafe_allow_html=True)

    if not hasattr(st.session_state, 'documents_loaded') or not st.session_state.documents_loaded:
        st.info("Пожалуйста, укажите папку с документами и обработайте их для начала работы.")
        return

    chat_container = st.container()
    
    with chat_container:
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)

        question = st.chat_input("Задайте вопрос по документам...")
        
        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Думаю..."):
                    answer, sources = st.session_state.agent.ask_question(question)
                    answer = answer.split("</think>")[-1].strip()
                    st.write(answer)
                    st.session_state.chat_history.append((question, answer))

                    if sources and len(sources) > 0:
                        with st.expander("📖 Источники"):
                            for i, source in enumerate(sources):
                                source_info = f"Файл: {source.metadata.get('source', 'Неизвестно')}"
                                st.text_area(
                                    f"Источник {i+1} ({source_info}):",
                                    source.page_content,
                                    height=100
                                )

if __name__ == "__main__":
    main() 