from pathlib import Path
from ragbolt.core.policy import Chunk, Corpus
import json, hashlib, uuid


SUPPORTED_EXTENSIONS = {".txt", ".md", ".json"}


def _split_long_sentence(sentence: str, chunk_size: int) -> list[str]:
    words = sentence.split()
    if len(words) <= chunk_size:
        return [sentence]
    parts: list[str] = []
    for i in range(0, len(words), chunk_size):
        parts.append(" ".join(words[i : i + chunk_size]))
    return parts


def _split_text_chunks(text: str, chunk_size: int) -> list[str]:
    chunks: list[str] = []
    paragraphs = text.split("\n\n")
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        words = paragraph.split()
        if len(words) <= chunk_size:
            if len(paragraph) >= 10:
                chunks.append(paragraph)
            continue
        sentences = [s.strip() for s in paragraph.split(". ") if s.strip()]
        current_words: list[str] = []
        for sentence in sentences:
            for part in _split_long_sentence(sentence, chunk_size):
                part_words = part.split()
                if len(current_words) + len(part_words) > chunk_size and current_words:
                    candidate = " ".join(current_words).strip()
                    if len(candidate) >= 10:
                        chunks.append(candidate)
                    current_words = []
                current_words.extend(part_words)
        if current_words:
            candidate = " ".join(current_words).strip()
            if len(candidate) >= 10:
                chunks.append(candidate)
    return chunks


def ingest_file(path: Path, chunk_size: int = 512) -> list[Chunk]:
    """
    Read a single file and split into chunks.

    .txt and .md files:
      - Read as UTF-8 text
      - Split into paragraphs on double newline
      - If paragraph > chunk_size words: split further on sentences (". ")
      - Each chunk gets chunk_id = f"{path.stem}_{i:04d}"
      - source = path.name

    .json files:
      - Must be a JSON array of objects
      - Each object must have at least "text" and "chunk_id" keys
      - "source" defaults to path.name if missing
      - Validated via Chunk model
      - Raise ValueError if not a list or missing required keys

    Returns list[Chunk]. Raises ValueError on malformed input.
    Raises FileNotFoundError if path missing.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if ext in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
        raw_chunks = _split_text_chunks(text, chunk_size)
        chunks: list[Chunk] = []
        for i, chunk_text in enumerate(raw_chunks, start=1):
            try:
                chunks.append(
                    Chunk(
                        chunk_id=f"{path.stem}_{i:04d}",
                        text=chunk_text,
                        source=path.name,
                    )
                )
            except Exception as e:
                raise ValueError(f"Invalid chunk in {path.name}: {e}") from e
        return chunks

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in {path.name}: {e}") from e
    if not isinstance(payload, list):
        raise ValueError("JSON input must be an array of objects")

    chunks = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("JSON array elements must be objects")
        if "text" not in item or "chunk_id" not in item:
            raise ValueError("JSON objects must include 'text' and 'chunk_id'")
        text = str(item["text"])
        if len(text.strip()) < 10:
            continue
        try:
            chunks.append(
                Chunk(
                    chunk_id=str(item["chunk_id"]),
                    text=text,
                    source=str(item.get("source", path.name)),
                    metadata=item.get("metadata", {}),
                )
            )
        except Exception as e:
            raise ValueError(f"Invalid chunk in {path.name}: {e}") from e
    return chunks


def _unique_chunk_id(base: str, file_stem: str, seen: set[str]) -> str:
    if base not in seen:
        return base
    candidate = f"{base}_{file_stem}"
    if candidate not in seen:
        return candidate
    nonce = 1
    while True:
        digest = hashlib.sha1(f"{candidate}:{nonce}".encode("utf-8")).hexdigest()[:8]
        final_id = f"{candidate}_{digest}_{uuid.uuid4().hex[:4]}"
        if final_id not in seen:
            return final_id
        nonce += 1


def ingest_directory(
    directory: Path,
    chunk_size: int = 512,
    recursive: bool = False,
) -> Corpus:
    """
    Ingest all supported files in directory.
    If recursive=True, walk subdirectories.
    corpus_id = directory.stem
    Raise ValueError if no supported files found.
    Raise ValueError if duplicate chunk_ids across files.
    Return Corpus.
    """
    if not directory.exists():
        raise FileNotFoundError(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    walker = directory.rglob("*") if recursive else directory.iterdir()
    files = sorted(
        [p for p in walker if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )
    if not files:
        raise ValueError("No supported files found")

    all_chunks: list[Chunk] = []
    seen_ids: set[str] = set()
    for file_path in files:
        file_chunks = ingest_file(file_path, chunk_size=chunk_size)
        for chunk in file_chunks:
            new_id = _unique_chunk_id(chunk.chunk_id, file_path.stem, seen_ids)
            if new_id != chunk.chunk_id:
                chunk = Chunk(
                    chunk_id=new_id,
                    text=chunk.text,
                    source=chunk.source,
                    metadata=chunk.metadata,
                )
            if chunk.chunk_id in seen_ids:
                raise ValueError(f"duplicate chunk_id: {chunk.chunk_id}")
            seen_ids.add(chunk.chunk_id)
            all_chunks.append(chunk)

    return Corpus(chunks=all_chunks, corpus_id=directory.stem)


def write_corpus(corpus: Corpus, output: Path) -> None:
    """
    Write corpus as JSON array to output path.
    Creates parent dirs if needed.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [chunk.model_dump() for chunk in corpus.chunks]
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
