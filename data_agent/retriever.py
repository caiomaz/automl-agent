import os
import requests
import re
import zipfile
import tarfile
from pathlib import Path
from urllib.parse import urlparse, unquote

from openai import OpenAI
from utils import get_kaggle, search_web, print_message
from utils.workspace import ensure_workspace, dataset_path_for_url, DATASETS_DIR
from configs import AVAILABLE_LLMs
from validators import url
from glob import glob

# Default User-Agent for HTTP requests
USER_AGENT = os.getenv("USER_AGENT") or "AutoML-Agent/1.0 (https://github.com/caiomaz/automl-agent)"

# DATASETS_DIR is re-exported from utils.workspace for backward compatibility


def retrieve_datasets(user_requirements, data_path, client, model):
    """Retrieve up-to-date and state-of-the-art knowledge from the websearch and relevant hubs for data manipulation and analysis."""

    # Return detailed sources and implementation for subsequent code generation
    # Internal: User Upload, User Link, and Data Hub
    # External: OpenML, UCI ML Dataset Archive, Hugging Face, Torch DataHub, Tensorflow DataHub, and Kaggle Dataset
    datasets = []

    for data in user_requirements["dataset"]:
        data["task"] = user_requirements["problem"]["downstream_task"]
        data["llm_client"] = client
        data["llm_model"] = model
        if data.get("source", "user-upload") in ["user-upload", "upload"]:
            # get upload files from storage
            loader_key = True
            datasets.append(
                {
                    "name": data["name"],
                    "loader_key": data_path,
                    "source": "user-upload",
                }
            )
            continue
        elif data.get("source", "user-upload") == "user-link":
            # download file from the given link
            data["url"] = re.search(
                r"(?P<url>https?://[^\s]+)", str(user_requirements["dataset"])
            ).group("url")
            loader_key = retrieve_download(**data)
            datasets.append(
                {
                    "name": data["name"],
                    "loader_key": loader_key,
                    "source": "user-link",
                }
            )
            continue
        elif data.get("source", "user-upload") == "direct-search":
            # search using name in data loaders
            loader_key = retrieve_huggingface(**data)
            if loader_key:
                datasets.append(
                    {
                        "name": data["name"],
                        "loader_key": loader_key,
                        "source": "huggingface-hub",
                    }
                )
                continue

            loader_key = retrieve_kaggle(**data)
            if loader_key and loader_key[0]:
                local_path, files = loader_key
                datasets.append(
                    {
                        "name": data["name"],
                        "loader_key": local_path,
                        "files": files,
                        "source": "local (downloaded from kaggle)",
                    }
                )
                continue

            loader_key, hub_name = retrieve_pytorch(**data)
            if loader_key:
                datasets.append(
                    {
                        "name": data["name"],
                        "loader_key": loader_key,
                        "source": hub_name,
                    }
                )
                continue

            loader_key = retrieve_tensorflow(**data)
            if loader_key:
                datasets.append(
                    {
                        "name": data["name"],
                        "loader_key": loader_key,
                        "source": "tensorflow-datasets",
                    }
                )
                continue

            loader_key = retrieve_uci(**data)
            if loader_key:
                datasets.append(
                    {
                        "name": data["name"],
                        "source": "ucimlrepo",
                    }
                )
                continue

            loader_key = retrieve_openml(**data)
            if loader_key:
                datasets.append({"name": data["name"], "source": "openml"})
                continue
              
        else:
            loader_key = retrieve_infer(**data)
            if loader_key:
                datasets.append(
                    {
                        "name": data["name"],
                        "loader_key": loader_key,
                        "source": "infer-search",
                    }
                )
                continue
    return datasets


def _is_applicable(data_task, user_task):
    if isinstance(data_task, list):
        for i in range(len(data_task)):
            data_task[i] = (
                data_task[i].replace("-", " ").replace("_", " ").lower().strip()
            )
            user_task = (
                user_task.replace("-", " ").replace("_", " ").lower().strip()
                if isinstance(user_task, str)
                else [
                    task.replace("-", " ").replace("_", " ").lower().strip()
                    for task in user_task
                ]
            )
            if data_task[i] in user_task:
                return True
    elif isinstance(data_task, str):
        if data_task in user_task:
            return True
    return False


def retrieve_infer(**kwargs):
    from langchain_community.document_loaders import PDFMinerLoader
    from langchain_community.document_transformers import Html2TextTransformer
    from langchain_core.documents import Document
    from utils.embeddings import chunk_and_retrieve

    query_prompt = f"""Give me a search query without special symbols to search for a dataset described by "{kwargs['description']}". Give me only the search query without explanation."""
    client = kwargs["llm_client"]

    messages = [
        {
            "role": "system",
            "content": "You are a data curator who has a lot of experience in data collection.",
        },
        {"role": "user", "content": query_prompt},
    ]
    while True:
        try:
            response = client.chat.completions.create(
                model=kwargs["llm_model"], messages=messages, temperature=0.3
            )
            break
        except Exception as e:
            print_message("system", e)
            continue

    search_query = response.choices[0].message.content.strip().replace('"', "")
    kaggle_api = get_kaggle()
    datasets = (kaggle_api.dataset_list(search=search_query, sort_by="votes") or [])[:10]
    for dataset in datasets:
        tags = [tag.name for tag in (dataset.tags or [])]
        if _is_applicable(tags, kwargs["modality"]):
            local_path, _files = _download_kaggle_dataset(kaggle_api, dataset.ref)
            return local_path or dataset.ref
    else:
        search_results = search_web(search_query)
        DOMAIN_BLOCKLIST = [
            "youtube.com",
            "twitter.com",
            "x.com",
            "hindawi.com",
            "ejournal.ittelkom-pwt.ac.id",
        ]

        search_results = [
            result
            for result in search_results
            if not any(domain in result["link"] for domain in DOMAIN_BLOCKLIST)
        ][:10]
        urls = [link["link"] for link in search_results if url(link["link"])]

        html_docs = []
        non_pdf_urls = [link for link in urls if ".pdf" not in link]

        # Only attempt Playwright if the chromium binary is present on disk.
        # This avoids a hard crash inside the multiprocessing worker when the
        # binary was never installed or its system deps are missing.
        def _chromium_binary_ready() -> bool:
            try:
                cache = Path.home() / ".cache" / "ms-playwright"
                return any(cache.rglob("chrome-headless-shell"))
            except Exception:
                return False

        playwright_tried = False
        if non_pdf_urls and _chromium_binary_ready():
            try:
                from langchain_community.document_loaders import AsyncChromiumLoader
                loader = AsyncChromiumLoader(non_pdf_urls)
                raw_html = loader.load()
                html2text = Html2TextTransformer()
                html_docs = html2text.transform_documents(raw_html)
                playwright_tried = True
            except BaseException as playwright_err:
                print_message("system", f"Playwright failed ({type(playwright_err).__name__}: {playwright_err}), using requests fallback.")

        if not playwright_tried:
            _headers = {"User-Agent": USER_AGENT}
            for _url in non_pdf_urls:
                try:
                    _resp = requests.get(_url, headers=_headers, timeout=10)
                    _resp.raise_for_status()
                    html_docs.append(Document(page_content=_resp.text, metadata={"source": _url}))
                except Exception as req_err:
                    print_message("system", f"Could not fetch {_url}: {req_err}")
            if html_docs:
                try:
                    _ht = Html2TextTransformer()
                    html_docs = _ht.transform_documents(html_docs)
                except Exception:
                    # html2text not installed — keep raw HTML documents as-is
                    pass
        for link in urls:
            if (
                "arxiv.org/pdf" in link
                or "/pdf?id=" in link
                or "&name=pdf" in link
            ):
                try:
                    html_docs += PDFMinerLoader(link).load()
                except Exception as e:
                    print('cannot load', link, 'with error:', e)                   
                
        # generate context from HTML pages for summary
        context = "".join(
            [
                d.page_content
                for d in chunk_and_retrieve(
                    ref_text=kwargs["description"],
                    documents=html_docs,
                    top_k=10,
                    ranker="bm25",
                )
            ]
        )
        # genearte summmary
        summary_prompt = f"""I searched the web using the query: {search_query}. Here is the result:
        =====================
        {context}
        =====================
        
        According to the given result, where can I retrieve the dataset described by the following reference data description? Please give me one location or possible URL for download.
        # Reference Data Description
        {kwargs['description']}
        """

        while True:
            try:
                response = client.chat.completions.create(
                    model=kwargs["llm_model"],
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a data curator who has a lot of experience in data collection.",
                        },
                        {"role": "user", "content": summary_prompt},
                    ],
                    temperature=0.3,
                )
                break
            except Exception as e:
                print_message("system", e)
                continue
        
        if re.search(r"(?P<url>https?://[^\s]+)", response.choices[0].message.content.strip()):
            return re.search(r"(?P<url>https?://[^\s]+)", response.choices[0].message.content.strip()).group("url")
        else:
            return response.choices[0].message.content.strip()

def retrieve_huggingface(**kwargs):
    if kwargs["name"] and kwargs["name"] != "":
        from huggingface_hub import HfApi

        hf_api = HfApi()
        dataset = list(
            hf_api.list_datasets(
                search=kwargs["name"],
                sort="downloads",
                full=True,
            )
        )
        if len(dataset) > 0:
            dataset = dataset[0]
            if dataset.card_data:
                if _is_applicable(dataset.card_data.task_categories, kwargs["task"]) or _is_applicable(dataset.card_data.task_ids, kwargs["task"]):
                    return dataset
                else:
                    return None
            else:
                return None
        else:
            return None


def retrieve_tensorflow(**kwargs):
    if kwargs["name"] and kwargs["name"] != "":
        try:
            import tensorflow_datasets as tfds
        except ImportError:
            return None

        name = kwargs["name"].strip().lower()
        is_exist = name in [
            ds.replace("_", " ").replace("-", " ").strip()
            for ds in tfds.list_builders()
        ]
        if is_exist:
            dataset = tfds.list_builders().index(name)
            return dataset
    return None


def retrieve_pytorch(**kwargs):
    if kwargs["name"] and kwargs["name"] != "":
        utility_classes = [
            "DatasetFolder",
            "ImageFolder",
            "VisionDataset",
            "wrap_dataset_for_transforms_v2",
        ]
        try:
            if "image" in kwargs["modality"] or "video" in kwargs["modality"]:
                from torchvision import datasets
                hub_name = "torchvision"
            elif "text" in kwargs["modality"]:
                from torchtext import datasets
                hub_name = "torchtext"
            elif "audio" in kwargs["modality"]:
                from torchaudio import datasets
                hub_name = "torchaudio"
            elif "graph" in kwargs["modality"]:
                from torch_geometric import datasets
                hub_name = "torch_geometric"
            else:
                # not support multimodal, time series, and tabular modalities
                return None, None
        except ImportError:
            return None, None

        avail_datasets = [
            fn.lower().replace("-", " ").replace("_", " ")
            for fn in datasets.__all__
            if fn not in utility_classes
        ]

        query = kwargs["name"].lower().replace("-", " ").replace("_", " ")
        if query in avail_datasets:
            return avail_datasets[avail_datasets.index(query)], hub_name
    return None, None


def _download_kaggle_dataset(kaggle_api, ref):
    """Download a Kaggle dataset to DATASETS_DIR and return (local_path, file_list).

    Returns ``(None, [])`` on failure so callers can fall through gracefully.
    """
    slug = ref.replace("/", "_")
    dest = DATASETS_DIR / slug
    dest.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded
    if any(dest.iterdir()):
        files = [f.name for f in dest.iterdir() if f.is_file()]
        print_message("data", f"Using cached Kaggle dataset at {dest}")
        return str(dest), files

    try:
        kaggle_api.dataset_download_files(ref, path=str(dest), unzip=True, quiet=True)
        files = [f.name for f in dest.iterdir() if f.is_file()]
        print_message("data", f"Downloaded Kaggle dataset '{ref}' → {dest}  ({', '.join(files)})")
        return str(dest), files
    except Exception as e:
        print_message("system", f"Kaggle download failed for '{ref}': {e}")
        return None, []


def retrieve_kaggle(**kwargs):
    kaggle_api = get_kaggle()
    if kwargs["name"] and kwargs["name"] != "":
        datasets = (kaggle_api.dataset_list(search=kwargs["name"], sort_by="votes") or [])[:10]
        for dataset in datasets:
            tags = [tag.name for tag in (dataset.tags or [])]
            if _is_applicable(tags, kwargs["modality"]) or _is_applicable(
                tags, kwargs["task"]
            ):
                local_path, files = _download_kaggle_dataset(kaggle_api, dataset.ref)
                if local_path:
                    return local_path, files
                return dataset.ref, []
        return None, []


def retrieve_uci(**kwargs):
    if kwargs["name"] and kwargs["name"] != "":
        from ucimlrepo import fetch_ucirepo

        try:
            dataset = fetch_ucirepo(name=kwargs["name"])
            if dataset != None:
                return dataset
        except:
            return None


def retrieve_openml(**kwargs):
    from openml.datasets import list_datasets

    if kwargs["name"] and kwargs["name"] != "":
        datalist = list_datasets(output_format="dataframe")
        found_dataset = datalist[datalist["name"].str.contains(kwargs["name"].lower())]
        found_dataset = found_dataset[found_dataset["status"] == "active"]
        return len(found_dataset) > 0


def retrieve_download(url: str, name: str = "", workspace=None, **kwargs):
    """Download a dataset from *url* into the workspace and return the local path.

    Parameters
    ----------
    url:
        Source URL (direct link, Kaggle page URL, HuggingFace, etc.).
    name:
        Human-readable label used to build the destination folder name.
    workspace:
        Override the default ``WORKSPACE_DIR`` (useful in tests).  The
        datasets sub-folder is always ``<workspace>/datasets/``.
    **kwargs:
        Extra keys from the pipeline (``task``, ``llm_client``, …) are
        accepted and silently ignored so the function is callable with
        ``**data`` dicts from ``retrieve_datasets``.

    Returns
    -------
    str | None
        Absolute string path of the local dataset directory, or ``None``
        on failure.
    """
    from utils.workspace import WORKSPACE_DIR as _DEFAULT_WS
    ws = Path(workspace) if workspace is not None else _DEFAULT_WS
    ensure_workspace(ws)

    # Unique, stable destination directory — same URL → same directory (cache)
    dest = dataset_path_for_url(url, name=name, datasets_dir=ws / "datasets")

    # Cache hit: directory exists and is non-empty
    if dest.exists() and any(dest.iterdir()):
        print_message("data", f"Using cached dataset at {dest}")
        return str(dest)

    # Download
    try:
        res = requests.get(url, stream=True, timeout=120, headers={"User-Agent": USER_AGENT})
        res.raise_for_status()
    except Exception as e:
        print_message("system", f"Failed to download {url}: {e}")
        return None

    dest.mkdir(parents=True, exist_ok=True)

    # Determine filename from Content-Disposition or URL path
    cd = res.headers.get("Content-Disposition", "")
    if "filename=" in cd:
        fname = re.findall(r'filename="?([^"]+)"?', cd)[0]
    else:
        fname = Path(unquote(urlparse(url).path)).name or "data"

    dest_file = dest / fname
    with open(dest_file, "wb") as f:
        for chunk in res.iter_content(chunk_size=8192):
            f.write(chunk)
    print_message("data", f"Downloaded {fname} → {dest}")

    # Auto-extract archives
    try:
        if zipfile.is_zipfile(dest_file):
            with zipfile.ZipFile(dest_file, "r") as zf:
                zf.extractall(dest)
            print_message("data", f"Extracted zip in {dest}")
        elif tarfile.is_tarfile(str(dest_file)):
            with tarfile.open(dest_file, "r:*") as tf:
                tf.extractall(dest)
            print_message("data", f"Extracted tar in {dest}")
    except Exception as e:
        print_message("system", f"Archive extraction failed: {e}")

    return str(dest)
