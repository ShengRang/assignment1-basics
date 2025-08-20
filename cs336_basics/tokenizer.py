import os
import re
from typing import BinaryIO, List, Dict, Tuple, NamedTuple, Optional, Set
from concurrent.futures import ProcessPoolExecutor, as_completed

import regex
from sortedcontainers import SortedList

# class Chunk(NamedTuple):
#     is_special: bool
#     start: int
#     length: int


# def find_chunk_boundaries(
#     file: BinaryIO,
#     desired_num_chunks: int,
#     special_tokens: List[bytes],
# ) -> List[Chunk]:
#     max_overlap = max(map(len, special_tokens))
#     file.seek(0, os.SEEK_END)
#     file_size = file.tell()
#     file.seek(0)
#
#     chunk_size = (file_size + desired_num_chunks - 1) // desired_num_chunks
#     mini_chunk_size = 1
#     while mini_chunk_size < 4096 and mini_chunk_size * 2 < chunk_size:
#         mini_chunk_size *= 2
#     for start in range(0, desired_num_chunks, chunk_size):
#         end = min(file_size, start + chunk_size)
#         mini_chunk = file.read(mini_chunk_size + max_overlap)
#         i = 0
#         while i < len(mini_chunk):
#             found = None
#             for st in special_tokens:
#                 if mini_chunk[i:i + max_overlap].startswith(st):
#                     found = st
#                     break
#             if found:
#                 i += len(found)
#             else:
#                 i += 1
#         if len(mini_chunk) < mini_chunk_size + max_overlap:
#             break
# todo 这个实现太复杂了，还是用提供的那个实现吧


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: List[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    # assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found = False
            for st in split_special_tokens:
                found_at = mini_chunk.find(st)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    found = True
                    break
            if found:
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def read_and_count(input_path: str, start: int, length: int, special_tokens: List[str]) -> Dict[str, int]:
    with open(input_path, "rb") as fp:
        fp.seek(start)
        chunk = fp.read(length).decode("utf-8", errors="ignore")
        special_token_re_pat = re.compile(f"{'|'.join(map(re.escape, special_tokens))}")
        split = special_token_re_pat.split(chunk)
        pre_tokenizer_pat = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        ans = {}
        for st in split:
            for mat in pre_tokenizer_pat.finditer(st):
                s = mat.group(0)
                ans[s] = ans.get(s, 0) + 1
        return ans


def merge_pair(t: Tuple[bytes, ...], idx: int) -> Tuple[bytes, ...]:
    n = len(t)
    if idx < 0:
        idx += n
    if idx < 0 or idx + 1 >= n:
        raise IndexError("idx out of range or no element at idx+1")
    return t[:idx] + (t[idx] + t[idx+1],) + t[idx+2:]


def build_tokenizer(
    word_freq: Dict[str, int],
    special_tokens: List[str],
    vocab_limit: int,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    vocab: Dict[int, bytes] = {}
    rev_vocab: Dict[bytes, int] = {}
    merges: List[Tuple[bytes, bytes]] = []
    tot = 256
    for i in range(256):
        b = bytes([i])
        vocab[i] = b
        rev_vocab[b] = i

    def insert(wb: bytes):
        if wb in rev_vocab:
            return
        nonlocal tot
        rev_vocab[wb] = tot
        vocab[tot] = wb
        tot += 1

    for special_token in special_tokens:
        insert(special_token.encode("utf-8"))

    bytes_freq: Dict[bytes, int] = {k.encode("utf-8"): v for k, v in word_freq.items()}
    bytes_min_max: Dict[bytes, Tuple[List[int], List[int]]] = {}

    for k, v in bytes_freq.items():
        m = len(k)
        bytes_min_max[k] = [*range(m)], [*range(m)]

    cnt_mp: Dict[Tuple[bytes, bytes], int] = {}
    pr_pos_mp: Dict[Tuple[bytes, bytes], Set[Tuple[bytes, int]]] = {}
    for k, v in bytes_freq.items():
        m = len(k)
        for i in range(m - 1):
            pr = (k[i:i+1], k[i + 1:i + 2])
            cnt_mp[pr] = cnt_mp.get(pr, 0) + v
            pr_pos_mp[pr] = pr_pos_mp.get(pr, set()) | {(k, i)}

    tree = SortedList()
    for k, v in cnt_mp.items():
        tree.add((v, k))

    def incr_pr_cnt(pr: Tuple[bytes, bytes], v: int, wd: bytes, idx: int):
        nonlocal tree
        nonlocal cnt_mp
        nonlocal pr_pos_mp
        old = cnt_mp.get(pr, 0)
        tree.discard((old, pr))
        cnt_mp[pr] = old + v
        tree.add((v + old, pr))
        pr_pos_mp[pr] = pr_pos_mp.get(pr, set()) | {(wd, idx)}

    def del_pr_cnt(pr: Tuple[bytes, bytes], v: int, wd: bytes, idx: int):
        nonlocal tree
        nonlocal cnt_mp
        nonlocal pr_pos_mp
        old = cnt_mp.get(pr, 0)
        tree.discard((old, pr))
        cnt_mp[pr] = old - v
        if old - v > 0:
            tree.add((old - v, pr))
        pr_pos_mp[pr] = pr_pos_mp.get(pr, set()) - {(wd, idx)}

    print(f'cnt_mp = {cnt_mp}')
    print(f'pr_pos_mp = {pr_pos_mp}')

    def find_next(word: bytes, idx: int) -> Tuple[bytes, int] | Tuple[None, None]:
        m = len(word)
        mi, ma = bytes_min_max[word]
        r = ma[idx] + 1
        if r < m:
            rr = ma[r]
            return word[r:rr+1], r
        return None, None

    def find_prev(word: bytes, idx: int) -> Tuple[bytes, int] | Tuple[None, None]:
        m = len(word)
        mi, ma = bytes_min_max[word]
        if idx > 0:
            l = mi[idx-1]
            return word[l:idx], l
        return None, None

    while tot < vocab_limit and len(tree) > 0:
        cnt, pr = tree.pop(-1)
        if cnt != cnt_mp.get(pr, 0):
            continue
        if cnt == 0:
            continue
        merges.append(pr)
        merged_bytes = pr[0] + pr[1]
        rev_vocab[merged_bytes] = tot
        vocab[tot] = merged_bytes
        tot += 1
        # pos = pr_pos_mp[pr]
        pos = set(pr_pos_mp.get(pr, set()))
        for wd, idx in pos:
            mcnt = bytes_freq[wd]
            ridx = bytes_min_max[wd][1][idx]
            if ridx + 1 < len(wd):
                ridx = bytes_min_max[wd][1][ridx+1]
            for i in range(idx, ridx+1):
                bytes_min_max[wd][1][i] = ridx
                bytes_min_max[wd][0][i] = idx
            (pre, pid), (nxt, nid) = find_prev(wd, idx), find_next(wd, idx)
            if pre:
                del_pr_cnt((pre, pr[0]), mcnt, wd, pid)
                incr_pr_cnt((pre, merged_bytes), mcnt, wd, pid)
            if nxt:
                del_pr_cnt((pr[1], nxt), mcnt, wd, nid)
                incr_pr_cnt((merged_bytes, nxt), mcnt, wd, idx)
    return vocab, merges


def bpe_tokenizer_training(
        input_path: str,
        vocab_size: int,
        special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[tuple[bytes, bytes]]]:
    pass


def main():
    input_path = '/home/yanx/learning/data/mini_txt.txt'
    num_worker = os.cpu_count()
    special_tokens = ["<|endoftext|>"]
    with open(input_path, "rb") as fp:
        boundaries = find_chunk_boundaries(fp, desired_num_chunks=num_worker, split_special_tokens=[b"<|endoftext|>"])
        word_freq = {}
        with ProcessPoolExecutor(max_workers=num_worker) as ex:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                print(f'{end - start}')
            futures = [ex.submit(read_and_count, *(input_path, start, end - start, special_tokens)) for start, end in zip(boundaries[:-1], boundaries[1:])]
            for fut in as_completed(futures):
                mp = fut.result()
                for k, v in mp.items():
                    word_freq[k] = word_freq.get(k, 0) + v
        print(len(word_freq))
        print(next(iter(word_freq.items())))
        vocab, merged = build_tokenizer(word_freq, special_tokens, 1000)
        print(len(vocab))
        print(f'merged: {len(merged)}')


if __name__ == "__main__":
    main()