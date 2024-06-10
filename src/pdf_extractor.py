def get_text_from_pdf(path):
    from pypdf import PdfReader 
    reader = PdfReader(path) 
    page = reader.pages[0] 
    text = ''
    for page in reader.pages: 
        text += page.extract_text() 
    return text