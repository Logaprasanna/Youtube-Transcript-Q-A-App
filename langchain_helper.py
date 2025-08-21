from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts  import PromptTemplate
import os

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Directory to store FAISS indexes
INDEX_DIR = "faiss_indexes"
os.makedirs(INDEX_DIR, exist_ok=True)


def create_vec_db_from_url(youtube_url: str) -> FAISS:
    if "shorts" in youtube_url:
        video_id = youtube_url.split("shorts/")[1]
    else:
        video_id = youtube_url.split("v=")[1].split("&")[0]

    index_path = os.path.join(INDEX_DIR, video_id)

    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    yt_api = YouTubeTranscriptApi()
    fetched_transcript = yt_api.fetch(video_id)
    full_text = " ".join([snippet.text for snippet in fetched_transcript])
    doc = Document(
        page_content= full_text,
        metadata = {
            "source": youtube_url,
            "video_id": video_id,
            "language": fetched_transcript.language,
            "is_generated": True
        }
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size =1000, chunk_overlap = 100)
    docs = text_splitter.split_documents([doc])
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(index_path)
    return db


def get_response_from_query(db, query, k=8):
    documents = db.similarity_search(query, k)
    full_docs = " ".join([doc.page_content for doc in documents])
    llm = ChatOpenAI(model="gpt-4o-mini")  
    prompt = PromptTemplate(
        input_variables= ["question", "docs"],
        template = """
        You are a helpful assistant that answers questions about YouTube videos 
        based only on their transcript.

        Context (transcript): {docs}
        Question: {question}

        🔹 Please respond elaborately in **under 8 sentences**.
        🔹 Format your answer in **Markdown** with bullets or short paragraphs.
        🔹 If the answer is not in the transcript, just say: "I don't know."
        """ ,
    )
    chains = prompt | llm  
    response = chains.invoke({"question": query, "docs": full_docs})
    response = response.content.replace("\n", "")

    return response

