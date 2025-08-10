from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from tqdm import tqdm
import traceback
import gc
import time
import requests
import re
from lxml import html as lhtml
import unicodedata as ud
import spacy
from collections import Counter

from src.utils.Logger import Logger
from src.dependencies.DatabaseHandler import DatabaseHandler
from src.dependencies.config import (
    COMPANY_A_LIST,
    COMPANY_B_LIST
)

BLOCKS = ('p','div','li','h1','h2','h3','h4','h5','h6','blockquote');

_CP1252_MOJIBAKE_MAP = str.maketrans({
    "\u0091": "\u2018",  # ‚ -> ‘  (left single smart)
    "\u0092": "\u2019",  # ’ -> ’  (right single smart)
    "\u0093": "\u201C",  # “ -> “  (left double smart)
    "\u0094": "\u201D",  # ” -> ”  (right double smart)
    "\u0096": "\u2013",  # – -> –  (en dash)
    "\u0097": "\u2014",  # — -> —  (em dash)
    "\u0085": "\u2026",  # … -> …
});

_EXTRA_QUOTES = {
    "\u00AB", "\u00BB",   # « »
    "\u2039", "\u203A",   # ‹ ›
    "\u201A", "\u201B",   # ‚ ‛
    "\u201C", "\u201D",   # “ ”
    "\u201E", "\u201F",   # „ ‟
    "\u275B", "\u275C", "\u275D", "\u275E",  # ❛ ❜ ❝ ❞
};

class Patcher:
    # Strict model else accuracy for abbreviation is impacted; non-param due to significant performance impact
    _nlp_ent_model = spacy.load("en_core_web_lg");

    def _leaf_block_xpath(self):
        cond = " or ".join(f"self::{t}" for t in BLOCKS);
        not_desc = " and ".join(f"not(descendant::{t})" for t in BLOCKS);
        return f"//*[{cond}][{not_desc}]";

    def _fix_quotes_and_spaces(self, s: str) -> str:
        s = s.translate(_CP1252_MOJIBAKE_MAP); # Fix cp1252 mojibake -> proper Unicode punctuation
        s = s.replace("\u00A0", " "); # Normalize NBSP to space
        return s;

    def preprocess_text(self, content: str) -> str:
        if isinstance(content, bytes):
            # Prefer utf-8; if it fails, fall back to cp1252
            try:
                content = content.decode("utf-8");
            except UnicodeDecodeError:
                content = content.decode("cp1252", errors="replace");

        doc = lhtml.fromstring(content);

        paras = [];
        for el in doc.xpath(self._leaf_block_xpath()):
            t = el.text_content(); # entities decoded -> Unicode
            t = self._fix_quotes_and_spaces(t);

            # Flatten ragged line wrapping inside a paragraph
            lines = [re.sub(r'[ \t]+', ' ', ln).strip() for ln in t.splitlines()];
            t = ' '.join(ln for ln in lines if ln);

            # Drop pure page-number blocks
            if not t or re.fullmatch(r'\d+', t):
                continue;

            paras.append(t);

        text = '\n\n'.join(paras);
        text = re.sub(r'\n{3,}', '\n\n', text).strip(); # Collapse accidental 3+ newlines
        return text;

    def normalize_quotes(self, text: str) -> str:
        out = [];
        for ch in text:
            cat = ud.category(ch); # e.g., 'Pi', 'Pf', 'Po'
            name = ud.name(ch, "");

            is_quote = (
                ch in _EXTRA_QUOTES or
                cat in ("Pi", "Pf") and ("QUOTE" in name or "QUOTATION" in name or "GUILLEMET" in name)
            );

            if not is_quote:
                out.append(ch);
                continue;

            # Singles -> "'", everything else -> '"'
            if "SINGLE" in name or ch in {"\u2039", "\u203A", "\u201A", "\u201B"}:
                out.append("'");
            else:
                out.append('"');
        return "".join(out);

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

    def _generate_abbreviation_definitions(self, passage: str, chunks: list[str], company_names: list[str]) -> str:
        doc = Patcher._nlp_ent_model(passage);
        # Tracks the frequency of organization entities
        org_counter = Counter(ent.text for ent in doc.ents if ent.label_ == "ORG");

        # Identify all valid paragraphs that defines the top 10 most frequent potential abbreviations
        # These potential abbreviation may or may not be the expected result but it improves accuracy
        abbreviation_map = {};
        for org, _ in org_counter.most_common(10):
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

        return output + "\n" + header + passage;

    def process_single_doc(self, main_index: int, company_A: str, company_B: str):
        # Determine batch collection
        batch_start = (main_index // 100) * 100;
        batch_end = batch_start + 99;
        collection_name = f"batch_{batch_start}_{batch_end}";

        with DatabaseHandler() as db:
            dataset_collection = db.dataset_db[collection_name];
            extracted_collection = db.extracted_sections_db[collection_name];

            # Locate extracted section
            doc = extracted_collection.find_one({"main_index": main_index});
            if not doc:
                print(f"Skipping index {main_index}: Document does not exist...");
                return;

            # Store the content from the database
            url = dataset_collection.find_one({"main_index": main_index})["url"];
            text = doc["content"];

        lines = text.split("\n");

        # Validate clean content
        if lines[0].startswith("Here are some potentially useful abbreviation definitions"):
            print(f"Skipping index {main_index}: Extracted section is clean...");
            return;

        # Check for the presence of the company names
        company_names = [company_A, company_B];
        company_tokens = {name.lower().split()[0].split('.')[0] for name in company_names};

        # Case 1: Direct presence (preserve hyphen)
        if all(token in "\n".join(lines[1:]).lower() for token in company_tokens):
            print(f"Skipping index {main_index}: Extracted section is clean...");
            return;

        # Case 2: Try replacing hyphens with spaces in company names, then check
        modified_tokens = [token.replace('-', ' ') for token in company_tokens];
        if all(token in "\n".join(lines[1:]).lower() for token in modified_tokens):
            print(f"Skipping index {main_index}: Extracted section is clean...");
            return;

        # Create a request that mimics browser activity
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        };

        response = requests.get(url, headers=headers);
        preprocessed_text = self.preprocess_text(response.text);
        normalize_quotes_text = self.normalize_quotes(preprocessed_text);

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=400
        );
        chunks = text_splitter.split_text(normalize_quotes_text);

        final_text = self._generate_abbreviation_definitions("\n".join(lines[1:]).strip(), chunks, company_names);

        # Update the extracted section in the database
        with DatabaseHandler() as db:
            extracted_collection = db.extracted_sections_db[collection_name];
            extracted_collection.update_one(
                {"main_index": main_index},
                {"$set": {"content": final_text}}
            );

        Logger.logMessage(f"[i] Patched abbreviation to index {main_index}");

    def runPatcher(
        self,
        start_index: int = None,
        end_index: int = None,
        index: int = None,
        batch_size: int = None
    ):
        # Ensure valid batch_size parameter
        if batch_size is None and index is None:
            raise ValueError("batch_size must be defined.");

        # If index is provided, override startIndex and endIndex
        if index is not None:
            self.__start_index = index;
            self.__end_index = index + 1;
            self.__batch_size = 1; # Index provided means only 1 batch size is necessary
        else:
            # Ensure startIndex and endIndex are set properly
            if start_index is None or end_index is None:
                raise ValueError("start_index and end_index must be provided if index is not provided.");

            self.__start_index = start_index;
            self.__end_index = end_index + 1;
            self.__batch_size = batch_size;

        indices_to_process = list(range(self.__start_index, self.__end_index));
        total_tasks = len(indices_to_process);

        with tqdm(
            total=total_tasks,
            desc = "\033[36mPatching\033[0m",
            unit="items",
            ncols=80,
            bar_format="\033[92m{desc}: {percentage:3.0f}%|\033[92m{bar}\033[0m| {n_fmt}/{total_fmt} [elapsed: {elapsed}]\n"
        ) as pbar:
            # Process jobs in batches of batch_size
            for i in range(0, total_tasks, self.__batch_size):
                batch_jobs = indices_to_process[i:i + self.__batch_size];

                # Check if the batch has only one job, then don't use multiprocessing
                if len(batch_jobs) == 1:
                    # Process the single job without multiprocessing
                    try:
                        self.process_single_doc(
                            batch_jobs[0],
                            COMPANY_A_LIST[batch_jobs[0]],
                            COMPANY_B_LIST[batch_jobs[0]]
                        );

                        pbar.update(1);
                    except Exception as e:
                        print(f"Error patching single job: {e}");
                        Logger.logMessage(f"[-] Process failed with error: {traceback.format_exc()}");
                else:
                    # Launches process pool for the current batch
                    with ProcessPoolExecutor(mp_context=get_context("spawn"), max_workers=min(self.__batch_size, os.cpu_count())) as process_pool:
                        futures = {
                            process_pool.submit(
                                self.process_single_doc,
                                job,
                                COMPANY_A_LIST[job],
                                COMPANY_B_LIST[job]
                            ): job
                            for job in batch_jobs
                        };

                        # Catch exceptions but nothing is returned
                        for future in as_completed(futures):
                            try:
                                future.result();
                                pbar.update(1);
                            except Exception as e:
                                print(f"Error patching future: {e}");
                                Logger.logMessage(f"[-] Process failed with error: {traceback.format_exc()}");

                # Cooldown and resource flush after every batch
                if len(batch_jobs) > 1:
                    print(f"Completed batch {i // self.__batch_size + 1}, waiting for cooldown...");
                    gc.collect();  # CPU flush
                    time.sleep(2);
                    print("Cooldown complete, proceeding to next batch...");


if __name__ == "__main__":
    # Patcher().runPatcher(index=1700);
    Patcher().runPatcher(start_index=1000, end_index=1699, batch_size=10);
