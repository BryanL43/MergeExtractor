from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor, ALL_COMPLETED, wait
from spacy.language import Language
import re
from rapidfuzz import fuzz
import spacy
import torch
from openai import OpenAI
import json
import torch
import torch.nn as nn
import sys
from collections import Counter
from sentence_transformers import CrossEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from Logger import Logger

QUERY_EMBEDDING_FILE = "./config/query_embedding.json";
RERANK_QUERY_FILE = "./config/rerank_query.txt";
BATCH_SIZE = 128; # Computing power specific. Tune for your device.

class ChunkProcessor:
    nlp: Language = None;
    # Strict model else accuracy for abbreviation is impacted; isolated due to significant performance impact
    _nlp_ent_model = spacy.load("en_core_web_lg");

    def __init__(self, nlp: Language, reranker_model: CrossEncoder, client: OpenAI, thread_pool: ThreadPoolExecutor):
        ChunkProcessor.nlp = nlp;
        self.reranker_model = reranker_model;
        self.client = client;
        self.thread_pool = thread_pool;

    @staticmethod
    def extract_chunks_with_dates(chunks: list[str], nlp: Language) -> list[tuple[int, str]]:
        # Subroutine speeds up runtime slightly
        def get_chunks_with_date(batch_chunks):
            docs = list(nlp.pipe(batch_chunks));
            results = [];

            for i, doc in enumerate(docs):
                chunk = batch_chunks[i];  # Get the corresponding chunk from the batch
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
                    results.append((i, chunk));

            return results;

        chunks_with_dates = [];

        # Using ThreadPoolExecutor to process chunks in parallel
        with ThreadPoolExecutor() as executor:
            futures = [];
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE];  # Create a batch of chunks
                futures.append(executor.submit(get_chunks_with_date, batch));  # Submit the batch for processing

            # Collect the results as they complete
            for future in as_completed(futures):
                result = future.result();
                if result:
                    chunks_with_dates.extend(result);

        return chunks_with_dates;

    @staticmethod
    def locate_chunk_header(chunk: str, start_phrases: list[str], nlp: Language) -> str:
        doc = nlp(chunk);
        start_phrases_lower = [p.lower() for p in start_phrases];
        background_only = len(start_phrases) == 1 and start_phrases_lower[0] == "background";

        # Fragment the chunk into sentences and then check their lines for potential start phrases headers
        for sent in doc.sents:
            sentence_text = sent.text.strip();
            lines = [line.strip() for line in sentence_text.splitlines() if line.strip()];

            # Check for literal match within sentence if we are using complete phrases rather than just "Background"
            if not background_only:
                for phrase in start_phrases_lower:
                    if phrase in sentence_text.lower() and "background" in sentence_text.lower():
                        return phrase;

            # If no match in the sentence, check each line with additional fuzzy match.
            # Additionally checks for direct "background" word match.
            for line in lines:
                if len(line) == 0:
                    continue;
                
                line_lower = line.lower();
                # Special case for only "Background" phrase to check for exact match 
                if background_only:
                    if line_lower == "background":
                        return line;
                else: # Perform fuzzy matching for other start phrases
                    for phrase in start_phrases_lower:
                        if phrase in line_lower or fuzz.ratio(line_lower, phrase) > 80:
                            if "background" in line_lower:
                                return line;

        return None;
    
    @staticmethod
    def has_section_title(chunk: str, phrase: str) -> bool:
        """Check if the section contains the header as a title"""
        paragraphs = [];
        buffer = [];

        # Split the text into paragraphs
        for line in chunk.splitlines():
            line = line.strip();

            # Detected a empty line; flush the buffer & stash paragraph
            if line == "":
                if buffer:
                    paragraphs.append(buffer);
                    buffer = [];
            else: # Non-empty line
                buffer.append(line);

        # Flush the last paragraph
        if buffer:
            paragraphs.append(buffer);

        # Find the paragraph containing the phrase.
        # If it has a lenght of 2 or less line then it's likely a section title.
        for para_lines in paragraphs:
            joined = "\n".join(para_lines);
            if phrase.lower() in joined.lower() and len(para_lines) <= 2:
                return True;
            
        return False;

    @staticmethod
    def is_not_toc(chunk: str, phrase: str) -> bool:
        lines = [line.strip() for line in chunk.splitlines()];
        start_index = next((i for i, line in enumerate(lines) if phrase.lower() in line.lower()), 0);

        toc_like_count = 0;
        paragraph_like_count = 0;
        i = start_index;

        # Iterate through the lines to count "TOC-like" and "paragraph-like" patterns
        while i < len(lines) - 1:
            # Heuristic 1: Count "text followed by empty line" pairs (TOC-like pattern)
            if lines[i] and not lines[i + 1]:
                toc_like_count += 1;
                i += 2;
            # Heuristic 2: Count "two or more consecutive non-empty lines" (paragraph-like pattern)
            elif lines[i] and lines[i + 1]:
                paragraph_like_count += 1;
                i += 2;
            # Default increment
            else:
                i += 1;

        # Decision: flag as TOC if TOC-like count >= 3 & paragraph-like count < 3
        return not (toc_like_count >= 3 and paragraph_like_count < 3);

    @staticmethod
    def _process_single_chunk(idx, chunk, start_phrases, nlp):
        found_start_phrase = ChunkProcessor.locate_chunk_header(chunk, start_phrases, nlp);
        if not found_start_phrase:
            return None;

        if not ChunkProcessor.has_section_title(chunk, found_start_phrase):
            return None;
        
        if not ChunkProcessor.is_not_toc(chunk, found_start_phrase):
            return None;

        lines = chunk.splitlines();
        lower_phrase = found_start_phrase.lower();

        for i, line in enumerate(lines):
            line = line.strip();
            if not line or lower_phrase not in line.lower():
                continue;

            # Filter out false positive section titles
            if any(term in line.lower() for term in ["reason", "industry", "identity", "filing", "corporate"]):
                continue;

            # Ensure passage is not too short (noise sections)
            passage = "\n".join(lines[i:]);
            if len(passage) > 200:
                return (idx, passage);
                break;

    @staticmethod
    def get_approx_chunks(
        chunks_with_dates: list[tuple[int, str]], 
        start_phrases: list[str], 
        nlp: Language
    ):
        
        approx_chunks = [];
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    lambda idx=idx, chunk=chunk: ChunkProcessor._process_single_chunk(idx, chunk, start_phrases, nlp),
                )
                for idx, chunk in chunks_with_dates
            ];

            for future in as_completed(futures):
                result = future.result();
                if result:
                    approx_chunks.append(result);

        return approx_chunks;

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

        approx_chunks = ChunkProcessor.get_approx_chunks(chunks_with_dates, start_phrases, nlp);
        if len(approx_chunks) == 0: # Edge case where there is no date within the "Background" chunk; toss all the chunks
            approx_chunks = ChunkProcessor.get_approx_chunks(list(enumerate(chunks)), start_phrases, nlp);

        approx_chunks = sorted(approx_chunks);
        approx_chunks = [(idx, chunk) for idx, chunk in approx_chunks];

        return chunks, approx_chunks;

    def __get_embedding(self, text: str) -> torch.Tensor:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        );
        return torch.tensor(response.data[0].embedding, dtype=torch.float32);

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

    def _find_definition_paragraph(self, chunks: list[str], org: str) -> (str | None):
        # Pattern to match the ORG in quotes within parentheses
        pattern = re.compile(r'\([^)]*?"{}"[^)]*?\)'.format(re.escape(org)), re.IGNORECASE);
        
        for chunk in chunks:
            # Split chunk into paragraphs
            paragraphs = re.split(r'\n\s*\n', chunk);
            for para in paragraphs:
                if pattern.search(para):
                    return para.strip();
                
        return None;

    def _compute_cosine_similarity(self, approx_chunks: list[tuple[int, str]]) -> list[tuple[int, float, str]]:
        # Load locally saved query embedding
        with open(QUERY_EMBEDDING_FILE, "r") as f:
            query_embedding = torch.tensor(json.load(f), dtype=torch.float32);

        # Create multiple threads to embed each chunk in parallel
        futures = [];
        for _, chunk in approx_chunks:
            futures.append(self.thread_pool.submit(self.__get_embedding, chunk));

        # Ensure all futures are completed before proceeding
        wait(futures, return_when=ALL_COMPLETED);

        # Compile the embeddings from the futures
        chunk_embeddings = [];
        for future in futures:
            try:
                embeddings = future.result();
                chunk_embeddings.append(embeddings);
            except Exception as e:
                Logger.logMessage(f"[-] Error retrieving embeddings: {e}");
                sys.exit(1);
        
        # Stack embeddings into a tensor
        chunk_embeddings = torch.stack(chunk_embeddings);

        # Normalize query embeddings
        query_embedding_normalized = query_embedding / torch.norm(query_embedding);

        # Normalize chunk embeddings
        chunk_embeddings_normalized = chunk_embeddings / torch.norm(chunk_embeddings, dim=1, keepdim=True);

        # Compute cosine similarity
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding_normalized.unsqueeze(0), 
            chunk_embeddings_normalized
        );

        return [(approx_chunks[i][0], similarities[i].item(), approx_chunks[i][1]) for i in range(len(similarities))];

    def _rerank_with_hybrid_score(
        self,
        final_chunks_sorted: list[tuple[int, float, str]]
    ) -> list[tuple[int, float, float, float, str]]:
        # Load locally saved rerank query
        with open(RERANK_QUERY_FILE, "r", encoding="utf-8") as f:
            rerank_query = f.read();
        
        pairs = [(rerank_query, chunk) for _, _, chunk in final_chunks_sorted];
        rerank_scores = self.reranker_model.predict(pairs, activation_fn=nn.Sigmoid()); # Sigmoid to map prob in the range of [0, 1]
    
        COSINE_WEIGHT = 0.4;
        RERANK_WEIGHT = 0.6;

        # Compute the hybrid score based on desire weights
        hybrid_chunks = [];
        for (index, cos_score, chunk), rerank_score in zip(final_chunks_sorted, rerank_scores):
            hybrid_score = (COSINE_WEIGHT * cos_score) + (RERANK_WEIGHT * rerank_score);
            hybrid_chunks.append((index, hybrid_score, cos_score, rerank_score, chunk));

        # Sort by hybrid score descending
        return sorted(hybrid_chunks, key=lambda x: x[1], reverse=True);

    def _generate_abbreviation_definitions(self, passage: str, chunks: list[str], company_names: list[str]) -> str:
        doc = ChunkProcessor._nlp_ent_model(passage);
        # Tracks the frequency of organization entities
        org_counter = Counter(ent.text for ent in doc.ents if ent.label_ == "ORG");

        # Identify all valid paragraphs that defines the top 5 most frequent potential abbreviations
        # These potential abbreviation may or may not be the expected result but it improves accuracy
        abbreviation_map = {};
        for org, _ in org_counter.most_common(5):
            definition_paragraph = self._find_definition_paragraph(chunks, org);
            if definition_paragraph:
                abbreviation_map.setdefault(definition_paragraph, []).append(org);
        
        # No definition paragraphs found, most likely due to acronym company names.
        # Directly return with the expanded company names in the header.
        header = f"The following provides details about the events leading up to the merger deal between {company_names[0]} & {company_names[1]}:\n";
        if len(abbreviation_map) == 0:
            return header + passage;
        
        output = "Here are some potentially useful abbreviation definitions that could help with analyzing the 'Background' section:\n";
        seen = set();

        # Format the final output abbreviation definition string to append to passage
        for definition, orgs in abbreviation_map.items():
            if definition in seen:
                continue;
            seen.add(definition);

            # Format the ORG list correctly
            if len(orgs) == 1:
                orgs_str = f"'{orgs[0]}'";
            elif len(orgs) == 2:
                orgs_str = f"'{orgs[0]}' and '{orgs[1]}'";
            else:
                orgs_str = "', '".join(orgs[:-1]);
                orgs_str = f"'{orgs_str}', and '{orgs[-1]}'";

            output += f"\nPassage that defines the abbreviation {orgs_str}:\n{definition}\n";

        return output + "\n" + header + "\n" + passage;

    def getSectionPassage(
        self, 
        chunks: list[str], 
        approx_chunks: list[tuple[int, str]], 
        company_names: list[str]
    ):
        if not approx_chunks:
            return None;

        # Case 1: Only one approximate chunk in list, so use the chunk directly
        if len(approx_chunks) == 1:
            index, beginning_chunk = approx_chunks[0];
        else: # Case 2: Multiple approximate chunks in list; use cosine similarity & reranker to rank chunks
            final_chunks = self._compute_cosine_similarity(approx_chunks);
            hybrid_chunks = self._rerank_with_hybrid_score(final_chunks);

            if not hybrid_chunks:
                return None;

            index, _, _, _, beginning_chunk = hybrid_chunks[0];
        
        # Acquire the background section text (with some margin of other sections)
        extracted_section = beginning_chunk + "\n" + "\n".join(chunks[index + 1:index + 12]);
        passage = self.__normalize_chunks(extracted_section);

        # Validate whether both company names are present in the passage: meaning no abbreviations required
        passage_clean = re.sub(r'\s+', ' ', passage.lower().strip());

        # Extract the simplified company name tokens (first word)
        company_tokens = [name.lower().split()[0].split('.')[0] for name in company_names];
        
        # Case 1: Direct presence (preserve hyphen)
        if all(token in passage_clean for token in company_tokens):
            return f"The following provides details about the events leading up to the merger deal between {company_names[0]} & {company_names[1]}:\n" + passage;

        # Case 2: Try replacing hyphens with spaces in company names, then check
        modified_tokens = [token.replace('-', ' ') for token in company_tokens];
        if all(token in passage_clean for token in modified_tokens):
            return f"The following provides details about the events leading up to the merger deal between {company_names[0]} & {company_names[1]}:\n" + passage;

        # If abbreviation defs are needed
        return self._generate_abbreviation_definitions(passage, chunks, company_names);
