from dotenv import load_dotenv
import os
# For using Google Generative AI with LangChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# For splitting long texts into manageable chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# For storing and searching vectorized text using FAISS
from langchain_community.vectorstores import FAISS

# For defining custom prompts
from langchain_core.prompts import PromptTemplate

# For extracting subtitles/transcripts from YouTube videos
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

#BUILDING CHAIN
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


###### Step 1a - Indexing (Document Ingestion)"""
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY')
video_id = "F4Zu5ZZAG7I"
try:
    # If you don’t care which language, this returns the “best” one
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    transcripts = []
    for chunk in transcript_list:
        transcripts.append(chunk["text"])
        transcript = " ".join(transcripts)


except:
    print("No captions available for this video.")

transcript

###### Step 1B (indexing: text spliteing, Chunk)"""#####

spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# Split the full transcript text into chunks
chunks = spliter.create_documents([transcript])

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

### Step 3 - Augmentation ###
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    task="text-generation",
    api_key=os.getenv("GEMINI_API_KEY")
)
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question= "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs= retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question})

### Step 4 - Generation ###
answer = llm.invoke(final_prompt)
print(answer.content)

### Building a Chain ###

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text   
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser


main_chain.invoke('Can you summarize the video')