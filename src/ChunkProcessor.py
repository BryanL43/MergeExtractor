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

from Logger import Logger

QUERY_EMBEDDING_FILE = "./config/query_embedding.json";
RERANK_QUERY_FILE = "./config/rerank_query.txt";

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

        found_start_phrase = None
        for sentence in sentences:
            sentence_stripped = sentence.strip();
            lines = [line.strip() for line in sentence_stripped.split("\n")];

            # Check for literal match within sentence if we are using complete phrases rather than just "Background"
            if not (len(start_phrases) == 1 and start_phrases[0].lower() == "background"):
                match = next(
                    (sc for sc in start_phrases if sc.lower() in sentence_stripped.lower()),
                    None
                );
                if match and "background" in sentence_stripped.lower():
                    found_start_phrase = match;
                    break;

            # If no match in the sentence, check each line with additional fuzzy match.
            # Additionally checks for direct "background" word match.
            for line in lines:
                if len(line) == 0:
                    continue;
                
                # Special case for only "Background" phrase to check for exact match 
                if len(start_phrases) == 1 and start_phrases[0].lower() == "background":
                    if line.strip().lower() == "background":
                        found_start_phrase = line;
                        break;
                else:
                    # Perform fuzzy matching for other start phrases
                    match = next(
                        (sc for sc in start_phrases if sc.lower() in line.lower() or fuzz.ratio(line.lower(), sc.lower()) > 80),
                        None
                    );
                    if match and "background" in line.lower():
                        found_start_phrase = line;
                        break;
        
            if found_start_phrase:
                break;

        return found_start_phrase;
    
    @staticmethod
    def has_section_title(chunk: str, phrase: str) -> bool:
        """Check if the section contains the header as a title"""
        lines = chunk.splitlines();
        paragraphs = [];
        buffer = [];

        # Split the text into paragraphs
        for idx, line in enumerate(lines):
            if line.strip() == "":
                # Detected a empty line; flush the buffer & stash paragraph
                if buffer:
                    paragraphs.append(buffer);
                    buffer = [];
            else: # Non-empty line
                buffer.append(line);

        # Flush the last paragraph
        if buffer:
            paragraphs.append(buffer);

        # with open("WFJWAHFWBAFJ.txt", "a", encoding="utf-8") as file:
        #     for para_lines, idxs in paragraphs:
        #         file.write("\n");
        #         file.write("--" * 50);
        #         file.write("\n");
        #         file.write("\n".join(para_lines));

        # Find the paragraph containing the phrase.
        # If it has a lenght of 2 or less line then it's likely a section title.
        for para_lines in paragraphs:
            joined = "\n".join(para_lines);
            if phrase.lower() in joined.lower() and len(joined.split("\n")) <= 2:
                return True;
            
        return False;

    @staticmethod
    def get_approx_chunks(
        chunks_with_dates: list[tuple[int, str]], 
        start_phrases: list[str], 
        nlp: Language
    ):
        
        approx_chunks = [];
        for idx, chunk in chunks_with_dates:
            found_start_phrase = ChunkProcessor.process_chunk(chunk, start_phrases, nlp);
            if not found_start_phrase:
                continue;

            has_title = ChunkProcessor.has_section_title(chunk, found_start_phrase);
            if not has_title:
                continue;

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
                    approx_chunks.append((idx, passage));
                    break;

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

    def __find_definition_paragraph(self, chunks: list[str], org: str) -> (str | None):
        # Pattern to match the ORG in quotes within parentheses
        pattern = re.compile(r'\([^)]*?"{}"[^)]*?\)'.format(re.escape(org)), re.IGNORECASE);
        
        for chunk in chunks:
            # Split chunk into paragraphs
            paragraphs = re.split(r'\n\s*\n', chunk)
            for para in paragraphs:
                if pattern.search(para):
                    return para.strip();
                
        return None;

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

        # Normalize query embedding
        query_embedding_normalized = query_embedding / torch.norm(query_embedding);

        # Normalize chunk embeddings
        chunk_embeddings_normalized = chunk_embeddings / torch.norm(chunk_embeddings, dim=1, keepdim=True);

        # Compute cosine similarity
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding_normalized.unsqueeze(0), 
            chunk_embeddings_normalized
        );

        final_chunks = [
            (approx_chunks[i][0], similarities[i].item(), approx_chunks[i][1]) 
            for i in range(len(similarities))
        ];

        final_chunks_sorted = sorted(final_chunks, key=lambda x: x[2], reverse=True);

        # Rerank the chunks to more accurately fit the "Background" section
        # rerank_query = (
        #     "Detailed chronological account of merger negotiations including: "
        #     "specific dates, financial advisors involved, board meeting details, "
        #     "and progression of deal terms"
        # );

        with open(RERANK_QUERY_FILE, "r", encoding="utf-8") as f:
            rerank_query = f.read();

        pairs = [(rerank_query, chunk) for _, _, chunk in final_chunks_sorted];
        rerank_scores = self.reranker_model.predict(pairs, activation_fn=nn.Sigmoid()); # Sigmoid to map prob in the range of [0, 1]

        COSINE_WEIGHT = 0.4;
        RERANK_WEIGHT = 0.6;

        # Combine scores and sort
        hybrid_chunks = []
        for (index, cos_score, chunk), rerank_score in zip(final_chunks_sorted, rerank_scores):
            hybrid_score = (COSINE_WEIGHT * cos_score) + (RERANK_WEIGHT * rerank_score)
            hybrid_chunks.append((
                index,
                hybrid_score,
                cos_score,
                rerank_score,
                chunk
            ));

        # Sort by hybrid score descending
        hybrid_chunks_sorted = sorted(hybrid_chunks, key=lambda x: x[1], reverse=True);

        # Print results with hybrid scores
        for entry in hybrid_chunks_sorted:
            index, hybrid_score, cos_score, rerank_score, chunk = entry
            print("--" * 50);
            print(f"Chunk {index} | Hybrid: {hybrid_score:.3f} | Cosine: {cos_score:.3f} | Rerank: {rerank_score:.3f}");
            print(chunk);

        # Select top result
        if hybrid_chunks_sorted:
            best_entry = hybrid_chunks_sorted[0];
            index, hybrid_score, cos_score, rerank_score, beginning_chunk = best_entry;
            print("==" * 50);
            print(f"Top hybrid-scored chunk: {index}");
            print(f"Hybrid: {hybrid_score:.3f} | Cosine: {cos_score:.3f} | Rerank: {rerank_score:.3f}");
            print(beginning_chunk);
        else:
            beginning_chunk = None;

        # reranked_chunks = sorted(zip(final_chunks_sorted, rerank_scores), key=lambda x: x[1], reverse=True);

        # for (index, cos_score, chunk), rerank_score in reranked_chunks:
        #     print("--" * 50);
        #     print(f"Chunk {index} | Cosine: {cos_score:.3f} | Rerank: {rerank_score:.3f}");
        #     print(chunk);

        # (index, cos_score, beginning_chunk), rerank_score = reranked_chunks[0];
        # print("==" * 50);
        # print("Top reranked chunk:");
        # print(f"Chunk {index} | Cosine: {cos_score:.3f} | Rerank: {rerank_score:.3f}");
        # print(beginning_chunk);

        if not beginning_chunk:
            return None;

        # Acquire the background section text (with some margin of other sections)
        # index, truncated_chunk = beginning_chunk;
        extracted_section = "\n".join(chunks[index + 1:index + 12]);
        extracted_section = beginning_chunk + "\n" + extracted_section;

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

        doc = ChunkProcessor._nlp_ent_model(passage);
        org_counter = Counter(); # Tracks the frequency of organization words

        # Label all potential Organiation entities
        for ent in doc.ents:
            if ent.label_ == "ORG":
                org_counter[ent.text] += 1;
        
        seen_chunks = set();
        abbreviation_map = {}; # Store abbreviation definitions with associated ORG

        abbreviation_definitions = "Here are some potentially useful abbreviation definitions that could help with analyzing the 'Background' section:\n";

        # Identify all valid paragraphs that defines the top 5 most frequent potential abbreviations
        # These potential abbreviation may or may not be the expected result but it improves accuracy
        for org, _ in org_counter.most_common(5):
            definition_paragraph = self.__find_definition_paragraph(chunks, org);
            if definition_paragraph:
                abbreviation_map.setdefault(definition_paragraph, []).append(org);
        
        # No definition paragraphs found, most likely due to acronym company names.
        # Directly return with the expanded company names in the header.
        if len(abbreviation_map) == 0:
            return f"The following provides details about the events leading up to the merger deal between {company_names[0]} & {company_names[1]}:\n" + passage;
        
        # Format the final output abbreviation definition string to append to passage
        for definition_paragraph, orgs in abbreviation_map.items():
            if definition_paragraph not in seen_chunks:
                seen_chunks.add(definition_paragraph);  # Mark chunk as seen

                # Format the ORG list correctly
                if len(orgs) > 2:
                    orgs_text = "', '".join(orgs[:-1]);
                    orgs_text = f"'{orgs_text}', and '{orgs[-1]}'";
                elif len(orgs) == 2:
                    orgs_text = f"'{orgs[0]}' and '{orgs[1]}'";
                else:
                    orgs_text = f"'{orgs[0]}'";

                abbreviation_definitions += f"\nPassage that defines the abbreviation {orgs_text}:\n";
                abbreviation_definitions += definition_paragraph + "\n";
        
        abbreviation_definitions += "\n";

        finalPassage = (
            abbreviation_definitions
            + f"The following provides details about the events leading up to the merger deal between {company_names[0]} & {company_names[1]}:\n\n" 
            + passage
        );

        return finalPassage; 