from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor, ALL_COMPLETED, wait, as_completed
from spacy.language import Language
import re
from fuzzywuzzy import fuzz
import spacy
import torch
from openai import OpenAI
import json
import torch
import sys

from Logger import Logger

QUERY_EMBEDDING_FILE = "./config/query_embedding.json";

class ChunkProcessor:
    nlp: Language = None;

    def __init__(self, nlp: Language, client: OpenAI, thread_pool: ThreadPoolExecutor):
        ChunkProcessor.nlp = nlp;
        self.client = client;
        self.thread_pool = thread_pool;
    
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

            # Check for literal match within sentence if we are using complete phrases rather than just "Background"
            if not (len(start_phrases) == 1 and start_phrases[0].lower() == "background"):
                match = next(
                    (sc for sc in start_phrases if sc.lower() in sentenceStripped.lower()),
                    None
                );
                if match:
                    foundStartPhrase = match;
                    break;

            # If no match in the sentence, check each line with additional fuzzy match
            for line in lines:
                if len(line) == 0:
                    continue;
                
                # Special case for only "Background" phrase to check for exact match 
                if len(start_phrases) == 1 and start_phrases[0].lower() == "background":
                    if line.strip().lower() == "background":  # Exact match check
                        foundStartPhrase = line;
                        break;
                else:
                    # Perform fuzzy matching for other start phrases
                    match = next(
                        (sc for sc in start_phrases if sc.lower() in line.lower() or fuzz.ratio(line.lower(), sc.lower()) > 80),
                        None
                    );
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

                # Filter out false positive section titles
                if any(word in line.lower() for word in ["reason", "industry", "identity", "filing", "corporate"]):
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
    ) -> list[tuple[int, str]] | None:
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
        approx_chunks = [(i, chunks[i]) for i in approx_indices];

        return chunks, approx_chunks;

    def __get_embedding(self, text: str) -> torch.Tensor:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        );
        return torch.tensor(response.data[0].embedding, dtype=torch.float32);

    def __get_beginning_chunk(self, finalChunks, startPhrases):
        for i, (index, chunk, _) in enumerate(finalChunks, start=1):
            doc = ChunkProcessor.nlp(chunk);
            sentences = [sent.text for sent in doc.sents];

            idx = -1;
            for j, sentence in enumerate(sentences):
                sentenceStripped = sentence.strip();

                # Check if the sentence starts with one of the phrases
                match = next(
                    (sc for sc in startPhrases if sc.lower() in sentenceStripped.lower()),
                    None
                );

                if match:
                    idx = j;
                    break;

            if idx != -1:
                return (index, " ".join(sentences[idx:]));

        return None;

    def __normalize_chunks(self, text: str) -> str:
        """ Remove overlapping lines """
        unique_lines = set();
        normalized_text = [];
        
        for line in text.split("\n"):
            line_s = line.strip();
            
            if line_s:
                if line_s not in unique_lines:
                    unique_lines.add(line_s);
                    normalized_text.append(line);
            else:
                normalized_text.append("");
        
        return "\n".join(normalized_text);

    def __find_parent_definition(self, chunks: list[str]) -> list[tuple[int, str]]:
        """
            Find the chunk that contains the definition of the abbreviation "Parent."
            It is usually in the format of: Company name ("Parent")
        """
        results = [];
        for i, chunk in enumerate(chunks):
            # Check if "Parent" is in the chunk
            if re.search(r'\bParent\b', chunk, re.IGNORECASE) and re.search(r'\(\s*"Parent"\s*\)', chunk):
                results.append((i, chunk));
        
        return results;

    def __extract_parent_paragraph(self, text) -> str | None:
        """Extract the paragraph that contains the word 'Parent'."""
        paragraphs = re.split(r'\n\s*\n|-{5,}', text);

        for paragraph in paragraphs:
            if re.search(r'parent', paragraph, re.IGNORECASE):
                return paragraph.strip();

        return None;

    def __extract_all_but_last_word(self, company_name: str) -> str:
        """
            Given a company name, drop the last word and merge domain-like terms.

            Parameters
            ----------
            company_name : str
                The company name to be processed.

            Returns
            -------
            str
                The processed company name.
        """
        clean_name = re.sub(r"\(.*?\)", "", company_name);  # Remove parentheses content
        words = re.split(r"[\s\_]+", clean_name.strip());  # Split by space or underscore

        # Domain-like terms to merge
        merge_words = {"net", "com", "org", "co"};

        # Merge domain-like words
        for i in range(len(words) - 1):
            if words[i].lower() in merge_words:
                words[i] = words[i] + "." + words[i + 1];
                words.pop(i + 1);
                break;

        if len(words) > 1:
            if words[-2] == "&":
                words = words[:-2]; # Remove both "&" and the last word
            else:
                words = words[:-1];  # Remove only the last word

        return " ".join(words);

    def getSectionPassage(
            self, 
            chunks: list[str], 
            approx_chunks: list[tuple[int, str]], 
            start_phrases: list[str],
            company_names: list[str]
        ):
        # Load query embeddings
        with open(QUERY_EMBEDDING_FILE, "r") as f:
            query_embedding = torch.tensor(json.load(f), dtype=torch.float32);
        
        # Create multiple threads to embed each chunk in parallel
        futures = {
            self.thread_pool.submit(self.__get_embedding, chunk):
            chunk for _, chunk in approx_chunks
        };

        # Ensure all futures are completed before proceeding
        wait([future for future in futures], return_when=ALL_COMPLETED);

        # Compile the embeddings from the futures
        chunk_embeddings = [];
        for future in as_completed(futures):
            try:
                embeddings = future.result();
                chunk_embeddings.append(embeddings);
            except Exception as e:
                Logger.logMessage(f"[{Logger.get_current_timestamp()}] [-] Error retrieving embeddings: {e}");
                sys.exit(1);

        # Stack embeddings into a tensor
        chunk_embeddings = torch.stack(chunk_embeddings);

        # Compute cosine similarity
        similarities = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), chunk_embeddings);

        # Sort indices by similarity in descending order
        indices = torch.argsort(similarities, descending=True).tolist();
        
        final_chunks = [
            (approx_chunks[i][0], approx_chunks[i][1], similarities[i].item()) 
            for i in indices
        ];

        # Locate the chunk that is the beginning of the "Background of the Merger" section
        beginning_chunk = self.__get_beginning_chunk(final_chunks, [phrase for phrase in start_phrases if phrase != "Background"]);
        if not beginning_chunk:
            beginning_chunk = self.__get_beginning_chunk(final_chunks, ["Background"]); # Fallback with low confidence

        if not beginning_chunk:
            return None;

        # Acquire the background section text (with some margin of other sections)
        index, truncated_chunk = beginning_chunk;
        extracted_section = "\n".join(chunks[index + 1:index + 10]);
        extracted_section = truncated_chunk + "\n" + extracted_section;

        passage = self.__normalize_chunks(extracted_section);
        parent_word_count = len(re.findall(r'\bparent\b', passage, re.IGNORECASE));
        if parent_word_count < 5:
            return "The following provides details on events leading up to the merger deal:\n" + passage;
    
        # Abbreviation is at the top of document so narrow the search
        parent_def_chunks = [chunk for _, chunk in self.__find_parent_definition(chunks[2:30])];
        parent_paragraph = self.__extract_parent_paragraph(parent_def_chunks[0]);
        if parent_paragraph is None:
            raise Exception("FATAL: Parent paragraph not found");

        # Validate company name are present in parent paragraph
        parent_paragraph_clean = re.sub(r'\s+', ' ', parent_paragraph.lower().strip());
        company_names_cut = [self.__extract_all_but_last_word(name).lower() for name in company_names];
        if not (company_names_cut[0] in parent_paragraph_clean or company_names_cut[1] in parent_paragraph_clean):
            raise Exception("FATAL: Company names were not found in the parent definition paragraph");

        finalPassage = (
            "The Parent abbreviation is defined here:\n\n" 
            + parent_paragraph + "\n\n"
            + "The following provides details on events leading up to the merger deal:\n\n" 
            + passage
        );

        return finalPassage; 