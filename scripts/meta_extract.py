import requests
import re
import os

SE_BIB_URL = "https://www.se-rwth.de/assets/bibliography/all-software-engineering-rwth-references.bib"

brackets_open = ["{", "[", "("]
brackets_close = ["}", "]", ")"]


def extract_element(string, i):
    '''
    Extracts string until the close of the opening brace
    '''
    if string[i] != "{":
        raise Exception('string[i] should be a brace, but it is not.')
    stack = 1
    extraction_list = []
    i += 1
    while stack > 0:
        c = string[i]
        if c in brackets_close:
            stack -= 1
        elif c in brackets_open:
            stack += 1
        if stack == 0:
            break
        extraction_list.append(c)
        i += 1
    return ''.join(extraction_list)


def extract_title_from_se_web_bib(bib_str):
    bib_title = None
    try:
        ma = re.search(r"title = ", bib_str)
        if ma != None:
            i = ma.span()[1]
            bib_title = extract_element(bib_str, i)
    except Exception as e:
        if str(e) == 'string[i] should be a brace, but it is not.':
            ma = re.search(r"title = ", bib_str)
            if ma != None:
                i = ma.span()[1]
                extraction_list = []
                while bib_str[i] != ",":
                    extraction_list.append(bib_str[i])
                    i += 1
                bib_title = ''.join(extraction_list)
        else:
            raise e
    return bib_title


def match_titles(web_title, tex_title):
    web_title = list(web_title.lower())
    tex_title = list(tex_title.lower())
    last_index = -1
    for c in web_title:
        if re.search(r"[a-zA-Z]", c) != None:
            try:
                index = tex_title.index(c)
                if index >= last_index:
                    del tex_title[index]
                    last_index = index
                else:
                    return False
            except ValueError:
                return False
    return True

def extract_authors_from_se_web_bib(bib_str):
    bib_authors = None
    ma = re.search(r"author = ", bib_str)
    if ma != None:
        i = ma.span()[1]
        bib_authors = extract_element(bib_str, i)
        return bib_authors


def extract_year_from_se_web_bib(bib_str):
    bib_year = None
    try:
        ma = re.search(r"year = ", bib_str)
        if ma != None:
            i = ma.span()[1]
            bib_year = extract_element(bib_str, i)
    except Exception as e:
        if str(e) == 'string[i] should be a brace, but it is not.':
            ma = re.search(r"year = ", bib_str)
            if ma != None:
                i = ma.span()[1]
                extraction_list = []
                while re.search(r"[0-9]", bib_str[i]) != None:
                    extraction_list.append(bib_str[i])
                    i += 1
                bib_year = ''.join(extraction_list)
        else:
            raise e
    return bib_year


def extract_source_from_se_web_bib(bib_str, bib_type):
    bib_source = None
    if bib_type == "book":
        if ma != None:
            j = ma.span()[1]
            bib_source = extract_element(bib_str, j)
    else:
        ma = re.search(r"journal = ", bib_str)
        if ma != None:
            j = ma.span()[1]
            bib_source = extract_element(bib_str, j)
        ma = re.search(r"bookTitle = ", bib_str)
        if ma != None:
            j = ma.span()[1]
            bib_source = extract_element(bib_str, j)
        ma = re.search(r"booktitle = ", bib_str)
        if ma != None:
            j = ma.span()[1]
            bib_source = extract_element(bib_str, j)
    return bib_source


def get_bib_from_se_web(title, url=SE_BIB_URL):
    response = requests.get(url)
    bib_types = re.findall(r"\@[a-zA-Z]*\{", response.text)
    bibs = re.split(r"\@[a-zA-Z]*\{", response.text)
    del bibs[0]
    for i in range(len(bibs)):
        ty = list(bib_types)
        del ty[0]
        del ty[-1]
        ty = ''.join(ty)
        ty = ty.lower()
        if ty == "comment":
            continue
        elif ty == "book":
            bib_type = "book"
        else:
            bib_type = "paper"

        bib_title = extract_title_from_se_web_bib(bibs[i])
        bib_authors = extract_authors_from_se_web_bib(bibs[i])
        bib_year = extract_year_from_se_web_bib(bibs[i])
        bib_source = extract_source_from_se_web_bib(bibs[i], bib_type)
        
        if bib_title != None:
            if match_titles(bib_title, title):
                return {
                    "title": bib_title,
                    "type": bib_type,
                    "authors": bib_authors,
                    "date": bib_year,
                    "source": bib_source
                }
    return "No match found."


def extract_title_from_tex(tex_str):
    title = None
    ma = re.search( r"\\title", tex_str)
    if ma != None:
        i = ma.span()[1]
        while tex_str[i] != "{":
            i += 1
        title = extract_element(tex_str, i)
    return title


def extract_authors_from_tex(tex_str):
    author = None
    ma = re.search( r"\\author", tex_str)
    if ma != None:
        i = ma.span()[1]
        while tex_str[i] != "{":
            i += 1
        author = extract_element(tex_str, i)
        # patterns_to_remove = [r"\\", r"footnote{.*}", r"\$.*\$", r"\n"]
        # for p in patterns_to_remove:
        #     author = re.sub(p, "", author)
    return author


def extract_year_from_tex(tex_str):
    year = None
    ma = re.search( r"\\year", tex_str)
    if ma != None:
        i = ma.span()[1]
        while tex_str[i] != "{":
            i += 1
        year = extract_element(tex_str, i)
    return year


def extract_meta_data_from_tex(path):
    with open(path, 'r') as f:
        tex_str = f.read()
    title = extract_title_from_tex(tex_str)
    authors = extract_authors_from_tex(tex_str)
    year = extract_year_from_tex(tex_str)
    return {
        "title": title,
        "type": None,
        "authors": authors,
        "date": year,
        "source": None
    }


def extract_meta_data_given_tex_file_path(path):
    '''
    Extracts the meta data given tex file path
    '''
    with open(path, 'r') as f:
        tex_str = f.read()
    title = extract_title_from_tex(tex_str)
    bib = get_bib_from_se_web(title)
    if bib == "No match found.":
        print("No match found on SE BibTex website.")
        print("Extracting from the tex file...")
        return extract_meta_data_from_tex(path)
    else:
        return bib


def extract_meta_data(path):
    dir_list = os.listdir(path)
    if "main.tex" in dir_list:
        tex_path = path + "/main.tex"
        return extract_meta_data_given_tex_file_path(tex_path)
    elif "paper.tex" in dir_list:
        tex_path = path + "/paper.tex"
        return extract_meta_data_given_tex_file_path(tex_path)
    else:
        raise Exception('Did not found the tex file.')
    
# print(get_bib_from_se_web("Reuse and Customization for Code Generators: \\Synergy by Transformations and Templates"))
# print(extract_meta_data_from_tex("Paper_Archive/19.03.25.Modelsward.TrafoTemplate/paper.tex"))
# print(extract_meta_data_given_tex_file_path("Paper_Archive/19.05.13.JB.EMISA.SE/paper.tex"))

### Some positive examples

# print(extract_meta_data("Paper_Archive/18.05.25.ID.EMISA.WebDEx"))
# print(extract_meta_data("Paper_Archive/19.03.25.Modelsward.TrafoTemplate"))
# print(extract_meta_data("Paper_Archive/19.05.06.IU.EMISA19.MontiGEM"))
# print(extract_meta_data("Paper_Archive/19.05.13.JB.EMISA.SE"))
# print(extract_meta_data("Paper_Archive/19.06.05.IP.CAiSE19.PrivacyPreservingDesign"))