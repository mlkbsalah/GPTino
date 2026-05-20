from datasets import load_dataset
import tiktoken
import os
import numpy as np
import multiprocessing
from tqdm import tqdm

tiktoken_cache_dir = "/workdir/bensalama/GPTino/tiktoken/"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

OUTPUT_DIR = "/workdir/bensalama/GPTino/fineweb-edu-tokenized/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

shard_size = 100_000_000

def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc))
    tokens_np = np.array(tokens)
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_tokens_file(filename, tokens_np):
    np.save(filename, tokens_np)

if __name__=="__main__":
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        name="sample-10BT", 
        split="train",
        streaming=True,
        ).select_columns(["text"])

    n_processes = os.cpu_count() - 1
    # import sys; sys.exit(0)

    with multiprocessing.Pool(n_processes) as pool:
        shard_idx = 0
        tokens_count = 0
        all_tokens = np.empty((shard_size,), dtype=np.uint16)
        progress_bar = None

        for tokens in pool.imap(tokenize, ds["text"], chunksize=16):

            while len(tokens) + tokens_count > shard_size:
                remainder = shard_size - tokens_count
                all_tokens[tokens_count:shard_size] = tokens[:remainder]
                progress_bar.update(remainder)
                split = "val" if shard_idx==0 else "train"
                filename = os.path.join(OUTPUT_DIR, f"edufineweb_{split}_{shard_idx:06d}")
                write_tokens_file(filename, all_tokens)

                shard_idx += 1
                tokens = tokens[remainder:]
                tokens_count = 0
                progress_bar = None
            
            all_tokens[tokens_count:tokens_count + len(tokens)] = tokens
            tokens_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, desc=f"Shard {shard_idx:06d}")
            progress_bar.update(len(tokens))
        
        if tokens_count > 0:
            split = "val" if shard_idx == 0 else "train"
            filename = os.path.join(OUTPUT_DIR, f"edufineweb_{split}_{shard_idx:06d}")
            write_tokens_file(filename, all_tokens[:tokens_count])
