from pathlib import Path
import re
import numpy as np
import ollama
import pandas as pd
from sentence_transformers import SentenceTransformer
import PyPDF2


def read_file(file_path: Path) -> str:
    """
    Read file content from .txt or .pdf
    :param file_path:
    :return:
    """
    if file_path.suffix.lower() == ".txt":
        return file_path.read_text(encoding="utf-8")
    elif file_path.suffix.lower() == ".pdf":
        text=""
        with file_path.open("rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


def clean_text(text: str) -> str:
    """
    Remove sections like 'Bibliography' or 'References' if present.
    :param text:
    :return:
    """
    match = re.search(r"(Bibliography|References)", text, re.IGNORECASE)
    return text[:match.start()] if match else text


def chunk_text(text: str, max_chunk_length: int = 1500) -> list:
    """
    Split text into smaller chunks for RAG
    :param text:
    :param max_chunk_length:
    :return:
    """
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 1 > max_chunk_length:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
        else:
            current_chunk += para + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def embed_chunks(chunks: list, embedder) -> np.ndarray:
    """
    Compute embedding for each chunk
    :param chunks:
    :param embedder:
    :return:
    """
    return np.array([embedder.encode(chunk) for chunk in chunks])


def retrieve_relevant_chunks(query: str, chunks: list, chunk_embeddings: np.ndarray,
                             embedder, top_k: int = 3) -> list:
    """
    Retrieve top_k chunks that are most similar to the query
    :param query:
    :param chunks:
    :param chunk_embeddings:
    :param embedder:
    :param top_k:
    :return:
    """
    query_embedding = embedder.encode(query)
    norms = np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    similarities = np.dot(chunk_embeddings, query_embedding) / (norms + 1e-10)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]


def rag_summarise(query: str, chunks, embeddings, embedder) -> str:
    """
    Given a document and a query, retrieve top relevant chunks and use them to prompt the LLM
    :param query:
    :param chunks:
    :param embeddings
    :param embedder:
    :return:
    """
    relevant_chunks = retrieve_relevant_chunks(query, chunks, embeddings, embedder, top_k=3)
    context = "\n".join(relevant_chunks)

    prompt = (f"Question: {query}\n\nContext:\n{context}\n\n"
              "Answer concisely based on the context")
    response = ollama.generate(model="gemma3:4b", prompt=prompt)
    return response.get("response", "").strip()


def process_file(file_path: Path,
                 embedder,
                 query_author: str,
                 query_research_obj: str,
                 query_concepts: str,
                 query_key_findings: str,
                 query_limitations: str,
                 query_gap: str,
                 query_relevance: str) -> tuple[str,str,str,str,str,str,str] or None:
    """
    Process a file using RAG: read the file, summarise it,
    save the summary as a .txt file, and return (filename, summary).
    :param file_path:
    :param embedder:
    :param query_author:
    :param query_research_obj:
    :param query_concepts:
    :param query_key_findings:
    :param query_limitations:
    :param query_gap:
    :param query_relevance:
    :return:
    """
    try:
        text = read_file(file_path)
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")
        return None

    try:
        cleaned_text = clean_text(text)
        chunks = chunk_text(cleaned_text)
        print(f"Document split into {len(chunks)} chunks.")
        embeddings = embed_chunks(chunks, embedder)

        print(f"Searching for the Author of {file_path.name}")
        author = rag_summarise(query_author, chunks, embeddings, embedder)
        print(f"Searching for the Research Objective of {file_path.name}")
        research_obj = rag_summarise(query_research_obj, chunks, embeddings, embedder)
        print(f"Searching for the Key Concepts of {file_path.name}")
        concepts = rag_summarise(query_concepts, chunks, embeddings, embedder)
        print(f"Searching for the Key Findings of {file_path.name}")
        key_findings = rag_summarise(query_key_findings, chunks, embeddings, embedder)
        print(f"Searching for the Limitations of {file_path.name}")
        limitations = rag_summarise(query_limitations, chunks, embeddings, embedder)
        print(f"Finding out the Research Gaps of {file_path.name}")
        gap = rag_summarise(query_gap, chunks, embeddings, embedder)
        print(f"Searching for the Industry Relevance of {file_path.name}")
        relevance = rag_summarise(query_relevance, chunks, embeddings, embedder)
        return author, research_obj, concepts, key_findings, limitations, gap, relevance
    except Exception as e:
        print(f"Error summarising {file_path.name}: {e}")
        return None


def main():
    input_folder = Path("input")
    output_folder = Path("output_rag")
    output_folder.mkdir(exist_ok=True)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    query_author = "Give the output as just the author and the year. nothing else. First author's fullname + publication year.No citations.No markdown notes.No commentary.No references section.Answer must be less than 30 words.Return Plain text."
    query_research_obj = "Give the output as just the research objective of the research paper. nothing else.No citations.No markdown notes.No commentary.No references section.Answer must be less than 30 words.Return Plain text."
    query_concepts = "Give the output just as Main Materials / Concepts that is Key materials, methods, technologies, or concepts only.No citations.No markdown notes.No commentary.No references section.Answer must be less than 30 words.Return Plain text."
    query_key_findings = "Give the output as just Key Findings that is the Most important result(s) only. nothing else.No citations.No markdown notes.No commentary.No references section.Report findings as direct factual statements.Write findings in an impersonal factual style.Focus only on results and conclusions.Answer must be less than 50 words.Do NOT mention author names, publication years, Study names, Citations.State only the result, conclusion or observation in the form of factual statement.Return Plain text."
    query_limitations = "Give the output as just the Limitations that is the Main limitation of the research paper. nothing else.No citations.No markdown notes.No commentary.No references section.Answer must be less than 40 words.Return Plain text."
    query_gap = "Give the output as just the Research Gap that is, the Unexplored aspect or unresolved issue of the research paper.No citations.No markdown notes.No commentary.No references section.Answer must be less than 30 words.Output only the extracted text itself.Do not prepend field names, labels, headers, or category names.Return Plain text."
    query_relevance = "Give the output as just the Industry Relevance that is, the Practical application or industrial significance. nothing else.No citations.No markdown notes.No commentary.No references section.Answer must be less than 30 words.Return Plain text."
    files = list(input_folder.glob("*.txt")) + list(input_folder.glob("*.pdf")) + list(input_folder.glob("*.PDF"))

    if not files:
        print("No supported files found in the input folder")
        return

    results = []
    for file in files:
        print(f"\nProcessing file {file.name} with RAG")
        result = process_file(file,
                              embedder,
                              query_author,
                              query_research_obj,
                              query_concepts,
                              query_key_findings,
                              query_limitations,
                              query_gap,
                              query_relevance)
        if result:
            results.append(result)

    if results:
        df = pd.DataFrame(results, columns=["Author & Year",
                                            "Research Objective",
                                            "Main Materials & Concepts",
                                            "Key Findings",
                                            "Limitations",
                                            "Research Gap",
                                            "Industry Relevance"])
        excel_path = output_folder / "summaries.xlsx"
        df.index = range(1, len(df)+1)
        df.to_excel(excel_path, index=True)
        print(f"\nAll summaries saved to {excel_path}")


if __name__ == "__main__":
    main()