import os

data_input_folder = 'data_input'

if not os.path.exists(data_input_folder):
    os.makedirs(data_input_folder)


def convert_epub_to_text(file_path):
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup

        book = epub.read_epub(file_path)
        text = ''
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text += soup.get_text() + '\n'
        return text
    except Exception as e:
        print(f"Error converting EPUB {file_path}: {e}")
        return ""


def convert_pdf_to_text(file_path):
    try:
        import PyPDF2

        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + '\n'
        return text
    except Exception as e:
        print(f"Error converting PDF {file_path}: {e}")
        return ""


def convert_files_to_text(input_folder):
    if not os.path.exists(input_folder):
        return

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        if not os.path.isfile(file_path):
            continue

        text_content = ""
        try:
            if filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    text_content = file.read()
            elif filename.endswith('.epub'):
                text_content = convert_epub_to_text(file_path)
            elif filename.endswith('.pdf'):
                text_content = convert_pdf_to_text(file_path)
            else:
                print(f"Unsupported file format: {filename}")
                continue
        except Exception as e:
            print(f"{filename} error: {e}")
            continue

        output_filename = filename.replace('.pdf', '.txt').replace('.epub', '.txt')
        output_file = os.path.join(data_input_folder, output_filename)
        with open(output_file, 'w', encoding='utf-8') as out_file:
            out_file.write(text_content)
            print(f"Converted {filename} to {output_file}")


if __name__ == "__main__":
    input_folder = "input_files"
    convert_files_to_text(input_folder)
