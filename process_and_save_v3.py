import os
import concurrent.futures
from rag_llama.rag.multi_context import MultiContextManager

data_input_folder = 'data_input'
vector_store_folder = 'vector_store'

if not os.path.exists(vector_store_folder):
    os.makedirs(vector_store_folder)


def split_text_by_words(text, max_words=5000):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def process_file(file_path, mcm, domain_id="default"):
    filename = os.path.basename(file_path)
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        text_content = file.read()

    chunks = split_text_by_words(text_content, max_words=5000)
    if chunks:
        mcm.add_document_to_domain(
            domain_id=domain_id,
            doc_id=filename,
            text_chunks=chunks,
            metadata={"source_file": file_path}
        )

    os.remove(file_path)
    print(f"Processed into domain '{domain_id}' with Context Network links and deleted {filename}")


def process_and_save_to_vector_store(embedding_model="plutonioumguy/bge-m3", max_workers=None, domain_id="default"):
    mcm = MultiContextManager(default_domain=domain_id)
    txt_files = [os.path.join(data_input_folder, f) for f in os.listdir(data_input_folder) if f.endswith('.txt')]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, file_path, mcm, domain_id) for file_path in txt_files]
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    process_and_save_to_vector_store(max_workers=200)
