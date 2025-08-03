from pathlib import Path
import speech_recognition as sr
from pprint import pprint
import os
from moviepy import VideoFileClip
import yt_dlp
import json

from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode
# from llama_index.llms.gemini import Gemini
from llama_index.multi_modal_llms.gemini import GeminiMultiModal

from dotenv import load_dotenv
load_dotenv()
google_api_key = os.getenv('google_gemini')
print(google_api_key)

def download_video(url, output_path):
    
    # Download a video from a given url and save it to the output path.

    # Parameters:
    # url (str): The url of the video to download.
    # output_path (str): The path to save the video to.

    # Returns:
    # dict: A dictionary containing the metadata of the video.
    ydl_opts = {
        'format': 'best',
        'outtmpl': output_path+"input_vid.%(ext)s",  # <--- path here
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        metadata = {"Author": info.get("uploader"), "Title": info.get("title"), "Views": info.get("view_count")}
    return metadata

def video_to_images(video_path, output_folder):
    """
    Convert a video to a sequence of images and save them to the output folder.

    Parameters:
    video_path (str): The path to the video file.
    output_folder (str): The path to the folder to save the images to.

    """
    clip = VideoFileClip(video_path)
    clip.write_images_sequence(
        os.path.join(output_folder, "frame%04d.png"), fps=0.2)
    
def video_to_audio(video_path, output_audio_path):
    """
    Convert a video to audio and save it to the output path.

    Parameters:
    video_path (str): The path to the video file.
    output_audio_path (str): The path to save the audio to.

    """
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(output_audio_path)

def audio_to_text(audio_path):
    """
    Convert audio to text using the SpeechRecognition library.

    Parameters:
    audio_path (str): The path to the audio file.

    Returns:
    test (str): The text recognized from the audio.

    """
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_path)

    with audio as source:
        # Record the audio data
        audio_data = recognizer.record(source)

        try:
            # Recognize the speech
            text = recognizer.recognize_whisper(audio_data)
        except sr.UnknownValueError:
            print("Speech recognition could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from service; {e}")

    return text

def data_loading(url):
    output_video_path = "./video_data/"
    output_folder = "./mixed_data/"
    output_audio_path = "./mixed_data/output_audio.wav"

    filepath = output_video_path + "input_vid.mp4"

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path(output_video_path).mkdir(parents=True, exist_ok=True)
    try:
        metadata_vid = download_video(url, output_video_path)
        video_to_images(filepath, output_folder)
        video_to_audio(filepath, output_audio_path)
        text_data = audio_to_text(output_audio_path)

        with open(output_folder + "output_text.txt", "w") as file:
            file.write(text_data)
        print("Text data saved to file")
        file.close()
        os.remove(output_audio_path)
        print("Audio file removed")

    except Exception as e:
        raise e
    return metadata_vid, output_folder


def data_store(output_folder):

    text_embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # image_model, image_preprocess = load_clip_model("openai/clip-vit-base-patch32")
    image_embed_model = ClipEmbedding(model_name="ViT-B/32")

    text_store = LanceDBVectorStore(uri="lancedb", table_name="text_collection")
    image_store = LanceDBVectorStore(uri="lancedb", table_name="image_collection")

    storage_context = StorageContext.from_defaults(
        vector_store=text_store,
        image_store=image_store)

    # Create the MultiModal index
    documents = SimpleDirectoryReader(output_folder).load_data()
    print('hii')
    index = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=text_embed_model,
        image_embed_model=image_embed_model,
    )
    print('hii')
    return index


def retrive_answer(index, query_str, metadata_vid,output_folder):
    retriever_engine = index.as_retriever(
        similarity_top_k=5, image_similarity_top_k=3
    )

    retrieval_results = retriever_engine.retrieve(query_str)

    retrieved_image = []
    retrieved_text = []

    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        else:
            display_source_node(res_node, source_length=200)
            retrieved_text.append(res_node.text)

    image_documents = SimpleDirectoryReader(
        input_dir=output_folder, input_files=retrieved_image
    ).load_data()

    context_str = "".join(retrieved_text)

    return image_documents, context_str


def final_result(metadata_vid, context_str, query_str, image_documents):

    metadata_str = json.dumps(metadata_vid)

    qa_tmpl_str = (
        "Given the provided information, including relevant images and retrieved context from the video, \
    accurately and precisely answer the query without any additional prior knowledge.\n"
        "Please ensure honesty and responsibility, refraining from any racist or sexist remarks.\n"
        "---------------------\n"
        "Context: {context_str}\n"
        "Metadata for video: {metadata_str} \n"
        "---------------------\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    gemini_llm = GeminiMultiModal(model_name="models/gemini-1.5-flash",api_key=google_api_key,max_new_tokens=1500)

    response_1 = gemini_llm.complete(
        prompt=qa_tmpl_str.format(
            context_str=context_str, query_str=query_str, metadata_str=metadata_str
        ),
        image_documents=image_documents,
    )

    return response_1