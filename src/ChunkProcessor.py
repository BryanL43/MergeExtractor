from langchain.text_splitter import RecursiveCharacterTextSplitter
from spacy.language import Language
import re
from fuzzywuzzy import fuzz
import spacy

class ChunkProcessor:
    def __init__(self, nlp: Language):
        self.nlp = nlp;
    
    @staticmethod
    def extract_chunks_with_dates(
            chunks: list[str], 
            nlp: Language
        ) -> list[tuple[int, str]]:

        chunks_with_dates = [];
        for i, chunk in enumerate(chunks):
            doc = nlp(chunk);
            date_entities = [ent for ent in doc.ents if ent.label_ == "DATE"];
            
            filtered_dates = [];
            for ent in date_entities:
                text = ent.text.lower();

                # Exclude entries with hyphens (often IDs or codes)
                if '-' in text:
                    continue;

                # For purely numeric entries, check if they exceed possible date ranges
                if text.replace('/', '').replace(' ', '').isdigit():
                    components = re.split(r'[/\s]+', text);

                    # Check if any component exceeds possible date values (4-digit integers)
                    if any(len(component) > 4 for component in components):
                        continue;

                    # If it's a single number, make sure it's in a reasonable year range
                    if len(components) == 1 and text.isdigit():
                        num = int(text);
                        if num < 1900 or num > 2030:
                            continue;

                filtered_dates.append(ent.text);

            # Context-based date detection for years
            year_mentions = re.findall(r'\b((?:19|20)\d{2})\b', chunk);
            for year in year_mentions:
                if year not in [date.lower() for date in filtered_dates]:
                    filtered_dates.append(year);
            
            # Remove duplicates while preserving order
            seen = set();
            filtered_dates = [x for x in filtered_dates if not (x in seen or seen.add(x))];
            
            if filtered_dates:
                chunks_with_dates.append((i, chunk));
        
        return chunks_with_dates;

    @staticmethod
    def process_chunk(chunk: str, start_phrases: list[str], nlp: Language) -> tuple[str, int]:
        doc = nlp(chunk);
        sentences = [sent.text.strip() for sent in doc.sents];

        foundStartPhrase = None
        for sentence in sentences:
            sentenceStripped = sentence.strip();
            lines = [line.strip() for line in sentenceStripped.split("\n")];

            # Check for literal match within sentence
            match = next(
                (sc for sc in start_phrases if sc.lower() in sentenceStripped.lower()),
                None
            )
            if match:
                foundStartPhrase = match;
                break;

            # If no match in the sentence, check each line with additional fuzzy match
            for line in lines:
                if len(line) == 0:
                    continue;
                
                match = next(
                    (sc for sc in start_phrases if sc.lower() in line.lower() or fuzz.ratio(line.lower(), sc.lower()) > 80),
                    None
                )
                if match:
                    foundStartPhrase = line;
                    break;
        
            if foundStartPhrase:
                break;

        return foundStartPhrase;
    
    @staticmethod
    def has_empty_neighbors(i: int, lines: list[str]) -> bool:
        """Check if the line has empty neighbors."""
        has_empty_before = (i == 0) or not lines[i - 1].strip();
        has_empty_after = (i == len(lines) - 1) or not lines[i + 1].strip();
        return has_empty_before or has_empty_after;

    @staticmethod
    def get_approx_chunk_indices(
        chunks_with_dates: list[tuple[int, str]], 
        start_phrases: list[str], 
        nlp: Language
    ):
        
        approx_indices = [];
        for idx, chunk in chunks_with_dates:
            found_start_phrase = ChunkProcessor.process_chunk(chunk, start_phrases, nlp);
            if not found_start_phrase:
                continue;

            lines = chunk.split("\n");
            for i, line in enumerate(lines):
                line = line.strip();
                if len(line) == 0:
                    continue;
                
                if found_start_phrase.lower() not in line.lower():
                    continue;

                if "reason" in line.lower() or "industry" in line.lower():
                    continue;

                if ChunkProcessor.has_empty_neighbors(i, lines):
                    approx_indices.append(idx);
                    break;
    
        return approx_indices;


    @staticmethod
    def locateBackgroundChunk(
        text: str, 
        start_phrases: list[str],
        nlp_model: Language = None,
        chunk_size: int = 2048, 
        chunk_overlap: int = 400, 
    ) -> list[int] | None:
        nlp = spacy.load(nlp_model) if isinstance(nlp_model, str) else nlp_model or ChunkProcessor.nlp;
        if nlp is None:
            raise RuntimeError("ERROR: You did not pass a nlp model into ChunkProcessor.locateBackgroundChunk");

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        );
        chunks = text_splitter.split_text(text);

        chunks_with_dates = ChunkProcessor.extract_chunks_with_dates(chunks, nlp);
        if len(chunks_with_dates) == 0:
            return None;

        approx_indices = ChunkProcessor.get_approx_chunk_indices(chunks_with_dates, start_phrases, nlp);
        approx_indices = sorted(approx_indices);

        return approx_indices;