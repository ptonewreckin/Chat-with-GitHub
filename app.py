import os
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import GitLoader, TextLoader
import shutil
import subprocess
from dotenv import load_dotenv


# Loader functions
def clone_repository(repo_url, local_path):
	if os.path.isdir(local_path):
		print("Removing existing code repository !")
		shutil.rmtree(local_path)
	subprocess.run(["git", "clone", repo_url, local_path])


def load_docs(root_dir):
	docs = []
	for dirpath, dirnames, filenames in os.walk(root_dir):
		for file in filenames:
			try:
				loader = TextLoader(os.path.join(
					dirpath, file), encoding='utf-8')
				docs.extend(loader.load_and_split())
			except Exception as e:
				print(e)
	return docs


def split_docs(docs):
	text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
	return text_splitter.split_documents(docs)


def main(repo_url, root_dir, persist_directory):
	try:
		clone_repository(repo_url, root_dir)
		docs = load_docs(root_dir)
		docs = split_docs(docs)
		embeddings = OpenAIEmbeddings()
		vector_store.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
		return True
	except Exception as e:
		print(e)
		return False


# Chat functions
def message(text, is_user=False, key=None):
	if is_user:
		st.markdown(f"> {text}", unsafe_allow_html=True)
	else:
		st.markdown(f"{text}", unsafe_allow_html=True)


def get_text():
	input_text = st.text_input("", key="input")
	return input_text


def search_db(query):
	retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
	model = ChatOpenAI(model='gpt-3.5-turbo')
	qa = RetrievalQA.from_llm(model, retriever=retriever)
	return qa.run(query)


st.set_page_config(page_title="Chat with GitHub", page_icon=None, layout='centered', initial_sidebar_state='collapsed')

st.title("Chat with GitHub")

repo_url = st.text_input("Enter a GitHub repository URL:")

if repo_url:
	root_dir = "./code_repo"
	persist_directory = 'db'
	load_dotenv()
	embeddings = OpenAIEmbeddings()
	vector_store = Chroma(persist_directory='db', embedding_function=embeddings)
	success = main(repo_url, root_dir, persist_directory)
	if success:
		st.success("Data successfully embedded.")
	else:
		st.error("Data failed to embed.")

	if 'generated' not in st.session_state:
		st.session_state['generated'] = ['i am ready to help you ser']

	if 'past' not in st.session_state:
		st.session_state['past'] = ['hello']

	user_input = get_text()

	if user_input:
		output = search_db(user_input)
		st.session_state.past.append(user_input)
		st.session_state.generated.append(output)

	if st.session_state['generated']:
		for i in range(len(st.session_state['generated'])):
			message(st.session_state['past'][i],
			        is_user=True, key=str(i) + '_user')
			message(st.session_state["generated"][i], key=str(i))
