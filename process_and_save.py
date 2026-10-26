import os
from rag_llama.rag.multi_context import MultiContextManager

data_input_folder = 'data_input'
vector_store_folder = 'vector_store'

if not os.path.exists(vector_store_folder):
    os.makedirs(vector_store_folder)


def process_and_save_to_vector_store(
    embedding_model="plutonioumguy/bge-m3",
    domain_id="default",
    multi_context_manager=None
):
    mcm = multi_context_manager or MultiContextManager(default_domain=domain_id)

    for filename in os.listdir(data_input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_input_folder, filename)
            if not os.path.isfile(file_path):
                continue
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text_content = file.read()

            chunks = [c.strip() for c in text_content.split('\n\n') if c.strip()]
            if chunks:
                mcm.add_document_to_domain(
                    domain_id=domain_id,
                    doc_id=filename,
                    text_chunks=chunks,
                    metadata={"source_file": file_path, "embedding_model": embedding_model}
                )

            os.remove(file_path)
            print(f"Processed into domain '{domain_id}' with Context Network links and deleted {filename}")


if __name__ == "__main__":
    process_and_save_to_vector_store()
