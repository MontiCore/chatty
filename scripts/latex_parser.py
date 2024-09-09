import copy
from pylatexenc.latex2text import LatexNodes2Text, MacroTextSpec, get_default_latex_context_db, EnvironmentTextSpec, SpecialsTextSpec
from pylatexenc.macrospec import LatexContextDb, MacroSpec, MacroStandardArgsParser
from pylatexenc import latexwalker
import tiktoken
import os
import json
import re
import bibtexparser
import logging
import latexcodec


def bibtex_to_apa(refs):
    # https://www.mendeley.com/guides/apa-citation-guide/
    # https://apastyle.apa.org/style-grammar-guidelines/references/elements-list-entry
    for k, v in refs.items():
        fields = v.fields_dict
        for k_f, v_f in fields.items():
            fields[k_f].value = v_f.value.encode("latex", errors="ignore").decode("latex")
        apa_string = ""
        if "author" in fields:
            apa_string += f"{fields['author'].value.replace('{', '').replace('}', '')}. "
        if "year" in fields:
            date = str(fields['year'].value.replace('{', '').replace('}', ''))
            if "month" in fields:
                date = f"{date}, {fields['month'].value.replace('{', '').replace('}', '')}"
            apa_string += f"({date}). "
        else:
            apa_string += f"(n.d). "
        if "booktitle" in fields:
            apa_string += f"{fields['booktitle'].value.replace('{', '').replace('}', '')}: "
        apa_string += f"{fields['title'].value.replace('{', '').replace('}', '')}. "
        if "publisher" in fields:
            apa_string += f"{fields['publisher'].value.replace('{', '').replace('}', '')}. "
        refs[k] = apa_string.strip()
    return refs


def load_projects(latex_dir, publications):
    projects = []
    for publication in publications:
        project_path = os.path.join(latex_dir, publication["source"])
        if not os.path.isdir(project_path):
            raise NotADirectoryError(f"{project_path} is expected to be a directory containing .tex files")
        project_dict = {"title": publication["title"], "path": project_path, "chapters": {},
                       "authors": publication["authors"], "date": publication["date"],
                       "type": publication["type"], "source": publication["source"]}

        # Load bibtex references
        bib_files = []
        for root, dirs, files in os.walk(os.path.join(latex_dir, publication["source"].split("/")[0])):
            for file in files:
                if file.endswith(".bib"):
                    bib_files.append(os.path.normpath(os.path.join(root, file)))

        references = {}
        for b in bib_files:
            with open(b, "r", encoding="utf-8") as f:
                content = f.read()
                library = bibtexparser.parse_string(content)
                print(f"Failed blocks: {library.failed_blocks}")
                references.update(library.entries_dict)
        references = bibtex_to_apa(references)
        project_dict["references"] = references

        for file in os.listdir(project_path):
            if file.endswith(".tex") and not file.rsplit(".", 1)[0] in publication["excluded"]:
                file_path = os.path.join(project_path, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                project_dict["chapters"][file.rsplit(".", 1)[0]] = {"path": file_path, "content": content}
        if len(project_dict["chapters"]) == 0:
            raise FileNotFoundError(f"No tex files found in {project_path}")
        projects.append(project_dict)
    return projects


def split_chunk(chunk, pattern):
    labels = pattern.findall(chunk)
    curr_label = ""
    splits = []
    for c in pattern.split(chunk):
        if c in labels:
            curr_label = c
        else:
            splits.append({"text": c, "label": curr_label})
    return splits


def chunk_text(text, references, max_tokens=512, chunk_count=0):
    chapter_pattern = re.compile(r'<CHAPTER>(.*?)</CHAPTER>', re.DOTALL)
    section_pattern = re.compile(r'<SECTION>(.*?)</SECTION>', re.DOTALL)
    subsection_pattern = re.compile(r'<SUBSECTION>(.*?)</SUBSECTION>', re.DOTALL)
    subsubsection_pattern = re.compile(r'<SUBSUBSECTION>(.*?)</SUBSUBSECTION>', re.DOTALL)
    citation_pattern = re.compile(r'<CITE>(.*?)</CITE>', re.DOTALL)

    chapters = split_chunk(text, chapter_pattern)
    initial_chunks = []
    for c in chapters:
        chapter = c["label"]
        sections = split_chunk(c["text"], section_pattern)
        for s in sections:
            section = s["label"]
            subsections = split_chunk(s["text"], subsection_pattern)
            for sub in subsections:
                subsection = sub["label"]
                subsubsections = split_chunk(sub["text"], subsubsection_pattern)
                for subsub in subsubsections:
                    if len(subsub["text"]) > 10:
                        initial_chunks.append({"chapter": chapter, "section": section, "subsection": subsection,
                                               "subsubsection": subsub["label"], "body": subsub["text"]})

    # Split section chunks based on max words on paragraph if possible, otherwise sentences.
    chunks = []
    encoder = tiktoken.get_encoding("cl100k_base")  # For OpenAI gpt 3.5/4, change depending on model used

    def add_chunk(chunk, body, tokens, index):
        new_chunk = copy.deepcopy(chunk)
        new_chunk["body"] = body
        new_chunk["tokens"] = tokens
        new_chunk["index"] = index
        chunks.append(new_chunk)
        return "", 0, index+1

    # Split on paragraphs if section is too long
    for c in initial_chunks:
        enc_c = encoder.encode(c["body"])
        if len(enc_c) <= max_tokens:
            c["tokens"] = len(enc_c)
            c["index"] = chunk_count
            chunk_count += 1
            chunks.append(c)
            continue
        paragraphs = c["body"].strip().split("\n\n")
        current_chunk = ""
        current_chunk_len = 0
        for p in paragraphs:
            p = p.strip()
            enc_p = encoder.encode(p)
            if len(enc_p) + current_chunk_len <= max_tokens:
                if current_chunk_len != 0:
                    current_chunk += "\n\n"
                current_chunk += p
                current_chunk_len += len(enc_p)
            elif current_chunk_len == 0:
                sentences = p.split(". ")
                for s in sentences:
                    enc_s = encoder.encode(s)
                    if len(enc_s) + current_chunk_len <= max_tokens:
                        current_chunk += s
                        current_chunk_len += len(enc_s)
                    elif current_chunk_len == 0:
                        logging.error(f"ERROR: Sentence longer than chunk")
                        input("Continue?")
                        continue
                    else:
                        current_chunk, current_chunk_len, chunk_count = add_chunk(c, current_chunk, current_chunk_len, chunk_count)
                    current_chunk, current_chunk_len, chunk_count = add_chunk(c, current_chunk, current_chunk_len,
                                                                              chunk_count)
            else:
                current_chunk, current_chunk_len, chunk_count = add_chunk(c, current_chunk, current_chunk_len,
                                                                          chunk_count)
        if current_chunk_len > 0:
            current_chunk, current_chunk_len, chunk_count = add_chunk(c, current_chunk, current_chunk_len, chunk_count)

    # Resolve citations
    for i, c in enumerate(chunks):
        body = c["body"]
        chunk_cit = citation_pattern.findall(body)
        cit_list = []
        for cit in chunk_cit:
            cit_list.extend(cit.split(","))
        cit_dict = {}
        for cit in cit_list:
            cit = cit.strip()
            if cit in references:
                cit_dict[cit] = references[cit]
            else:
                logging.warning(f"Warning: no citation found for {cit}\n"
                                f"Chunk cit: {chunk_cit}")

        body = body.replace("<CITE>", "[").replace("</CITE>", "]")
        c["body"] = body
        c["citations"] = [f"{k}: {v}" for k, v in cit_dict.items()]

    return chunks


def parse_latex(chapter):
    macros = []
    environments = []

    # Macros to remove entirely with arguments
    remove_macros = ["inputminted", "minipage", "caption",
                     "protocol", "status", "findex", "punkt", "pic", "authors", "drafttext", "existsScode", "scode", "tcode", "laterdo", "markeMCG"]
    remove_environments = ["tip", "table", "tabular", "minipage", "figure", "lstlisting", "marke", "lstset", "lstinputlisting", "tcodeinline"]
    for r in remove_macros:
        macros.append(MacroTextSpec(r, simplify_repl="", discard=True))
    for e in remove_environments:
        environments.append(EnvironmentTextSpec(e, simplify_repl="", discard=True))


    # Custom macro handling
    macros.append(MacroTextSpec("texttt", simplify_repl="%s"))
    macros.append(MacroTextSpec("cite", simplify_repl="<CITE>%(4)s</CITE>"))
    macros.append(MacroTextSpec("section", simplify_repl="<SECTION>%(3)s</SECTION>"))  # Name is stored as third argument
    macros.append(MacroTextSpec("subsection", simplify_repl="<SUBSECTION>%(3)s</SUBSECTION>"))
    macros.append(MacroTextSpec("chapter", simplify_repl="<CHAPTER>%(3)s</CHAPTER>"))
    macros.append(MacroTextSpec("subsubsection", simplify_repl="<SUBSUBSECTION>%(3)s</SUBSUBSECTION>"))

    # TODO: coco makro
    # TODO: move to json file

    # Macro definitions (set number of arguments for custom macros)
    macro_definitions = []
    macro_definitions.append(MacroSpec("status", "{{"))
    macro_definitions.append(MacroSpec("protocol", "{{"))
    macro_definitions.append(MacroSpec("findex", "{"))
    macro_definitions.append(MacroSpec("punkt", "{"))
    macro_definitions.append(MacroSpec("authors", "{"))
    macro_definitions.append(MacroSpec("pic", "{{{{"))
    macro_definitions.append(MacroSpec("existsScode", "{"))
    macro_definitions.append(MacroSpec("scode", "{"))
    macro_definitions.append(MacroSpec("tcode", "{"))
    macro_definitions.append(MacroSpec("markeJava", "{"))
    macro_definitions.append(MacroSpec("laterdo", "{{"))
    macro_definitions.append(MacroSpec("marke", "{"))
    macro_definitions.append(MacroSpec("lstset", "{"))
    macro_definitions.append(MacroSpec("lstinputlisting", "[{"))
    macro_definitions.append(MacroSpec("markeMCG", "{"))
    macro_definitions.append(MacroSpec("drafttext", "{"))


    walker_db = latexwalker.get_default_latex_context_db()
    walker_db.add_context_category("custom", prepend=True, macros=macro_definitions)

    # Custom handling of macros
    default_db = get_default_latex_context_db()
    default_db.add_context_category(category=None, macros=macros, environments=environments, prepend=True)

    lw = latexwalker.LatexWalker(chapter, latex_context=walker_db)
    nodelist, _, _ = lw.get_latex_nodes()
    converter = LatexNodes2Text(latex_context=default_db, fill_text=True)

    converted = converter.nodelist_to_text(nodelist)
    # Remove large empty sections
    sections = converted.strip().split("\n\n")
    cleaned_sections = [section.strip().replace("\n", " ") for section in sections if section.strip()]  # Optionally do: .replace("\n", " ")
    converted = '\n\n'.join(cleaned_sections)
    return converted


def process_project(latex_dir, project):
    chunks = []
    chunk_count = 0
    for k, v in project["chapters"].items():
        text = parse_latex(v["content"])
        chapter_chunks = chunk_text(text, project["references"], max_tokens=512, chunk_count=chunk_count)
        chunk_count += len(chapter_chunks)
        chunks.extend(chapter_chunks)
    for c in chunks:
        c["document_title"] = project["title"]
        c["authors"] = project["authors"]
        c["publish_date"] = project["date"]
        c["document_group"] = project["type"]
        c["body"] = c["body"].strip()
    if not os.path.isdir(f"{latex_dir}/processed"):
        os.mkdir(f"{latex_dir}/processed")
    with open(f"{latex_dir}/processed/{os.path.split(project['source'])[0]}.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    latex_dir = "../data/latex"
    with open(f"latex_meta.json", "r", encoding="utf-8") as f:
        publications = json.load(f)
    projects = load_projects(latex_dir, publications)
    for p in projects:
        process_project(latex_dir, p)


