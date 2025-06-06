from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_models import BedrockChat
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_aws import BedrockEmbeddings

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end. if you know the answer summarize with 
10 words. If you don't know the answer, 
just say give answer as "I don't know" in 3 word

<context>
{context}
</context
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 1}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer

def get_claude_llm():
    llm=BedrockChat(
        credentials_profile_name='default',
        model_id='anthropic.claude-3-sonnet-20240229-v1:0',
        region_name='us-east-1',
        model_kwargs={
        "max_tokens":2048,
        "temperature": 0.1,
        "top_p": 0.9})
    return llm    

def getVectorStore():
    index = pc.Index("langchain-rag-index")
    data_embeddings=BedrockEmbeddings(
    credentials_profile_name= 'default',
    model_id='amazon.titan-embed-text-v1',
    region_name='us-east-1')    
    vector_store = PineconeVectorStore(index=index, embedding=data_embeddings)
    return vector_store    

 
def lambda_handler(event, context):
    llm=get_claude_llm()
    vectorStore = getVectorStore()
    result = get_response_llm(llm,vectorStore,"what are the different type of time-off")
    return result 
