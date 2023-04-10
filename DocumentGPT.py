from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.verctorstores import Pinecone
from langchain.embedding.openai import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain import PromtTemplate, LLMChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

import pinecone

# A dictionary of OpenAI and Pinecone API keys (check app.pinecone.io)
from api_keys import api_keys_dict

class DocumentGPT():
    def __init__(
        self, document_link, model_name='gpt-4', temperature=.1,
        role='You are a helpful natural language processing expert.'
        ) -> None:

        self.document_link = document_link

        self.model_name = model_name
        self.temperature = temperature

        self.role = role

        # Load the PDF from the Link
        self.loader = OnlinePDFLoader(self.document_link)
        self.data = loader.load()

        # Split the Large PDF into small chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=0
            )
        self.texts = self.text_splitter.split_documents(self.data)

        # Embedding the textual data and saving in Pinecone in vectorial format
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_keys_dict['openai_api_key']
            )

        pinecone.init(
            api_key=api_keys_dict['pinecode_api_key'],
            environment=api_keys_dict['pinecode_api_env']
            )
        self.index_name = 'pinex'

        self.docsearch = Pinecone.from_texts(
            [t.page_content for t in self.texts], 
            self.embeddings, 
            index_name=self.index_name
            )
        
        # Initialize ChatGPT with the right model and temperature
        self.llm = ChatOpenAI(
            model_name=self.model_name, 
            temperature=self.temperature, 
            openai_api_key=api_keys_dict['openai_api_key']
            )
        
        # Specify the template and the Role that ChatGPT should play (Here an NLP Expert)
        self.template = role + ' Answer the questions in detail in the language the question was asked. {documents}'
        system_message_prompt = SystemMessagePromptTemplate.from_template(self.template)

        self.human_template = '{question}'
        self.human_message_promt = HumanMessagePromptTemplate.from_template(self.human_template)

        self.self.chat_prompt = ChatPromptTemplate.from_messages([self.system_message_prompt, self.human_message_promt])

        self.chain = LLMChain(llm=self.llm, prompt=self.chat_promt)

    def query(self, query):

        docs = self.docsearch.similarity_search(query, include_metadata=True)

        return self.chain.run(documents=docs, question=query)