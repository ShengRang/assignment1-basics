import os
import re
from typing import BinaryIO, List, Dict, Tuple, NamedTuple, Optional, Set, Union
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


from typing import Dict, Tuple, List, Union
from collections import Counter
import itertools

def _to_bytes(x: Union[str, bytes]) -> bytes:
    return x if isinstance(x, (bytes, bytearray)) else str(x).encode('utf-8')

def _split_by_specials(word: bytes, specials: List[bytes]) -> List[bytes]:
    """
    Split a bytes string into segments where special tokens are isolated.
    Example: b"abc<e>def" with specials=[b"<e>"] -> [b"abc", b"<e>", b"def"]
    """
    if not specials:
        return [word] if word else []
    segments = []
    i = 0
    n = len(word)
    # to avoid repeatedly scanning, find next earliest special occurrence
    while i < n:
        best_pos = None
        best_tok = None
        for st in specials:
            pos = word.find(st, i)
            if pos != -1 and (best_pos is None or pos < best_pos):
                best_pos = pos
                best_tok = st
        if best_pos is None:
            segments.append(word[i:])  # remainder
            break
        if best_pos > i:
            segments.append(word[i:best_pos])
        segments.append(best_tok)
        i = best_pos + len(best_tok)
    return segments

def brute_force_bpe_with_specials(
    word_freq: Dict[Union[str, bytes], int],
    vocab_size: int,
    special_tokens: List[Union[str, bytes]] = None,
    verbose: bool = True
):
    """
    Brute-force BPE trainer that respects special tokens (they are atomic and never merged).
    Returns (vocab_map: dict token(bytes)->id, merges: list[(left,right)], tokenized: dict word_bytes->list[tokens]).
    """
    special_tokens = [ _to_bytes(s) for s in (special_tokens or []) ]
    special_set = set(special_tokens)

    # normalize words to bytes and aggregate frequencies
    words = {}
    for w, cnt in word_freq.items():
        wb = _to_bytes(w)
        if not wb:
            continue
        words[wb] = words.get(wb, 0) + int(cnt)

    # initial tokenization: split by special tokens, non-special segments -> list of single-bytes
    tokenized = {}
    for wb in words.keys():
        segs = _split_by_specials(wb, special_tokens)
        toks = []
        for seg in segs:
            if seg in special_set:
                toks.append(seg)           # keep special token as a single atomic token
            else:
                # split into bytes
                toks.extend([seg[i:i+1] for i in range(len(seg))])
        tokenized[wb] = toks

    # initial vocab: all tokens present (including special tokens)
    vocab = {}
    next_id = 0
    for toks in tokenized.values():
        for t in toks:
            if t not in vocab:
                vocab[t] = next_id
                next_id += 1

    merges: List[Tuple[bytes, bytes]] = []

    def count_pairs():
        """Count adjacent pairs across all tokenized words, but skip pairs that include special tokens."""
        c = Counter()
        for wb, cnt in words.items():
            toks = tokenized[wb]
            for i in range(len(toks) - 1):
                a, b = toks[i], toks[i+1]
                # skip any pair that involves a special token (protect specials)
                if a in special_set or b in special_set:
                    continue
                c[(a, b)] += cnt
        return c

    # training loop
    while len(vocab) < vocab_size:
        pair_counts = count_pairs()
        if not pair_counts:
            if verbose:
                print("No mergeable pairs left.")
            break
        pair, freq = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
        left, right = pair
        if freq == 0:
            if verbose:
                print("Top pair has zero frequency; stopping.")
            break

        merged = left + right
        merges.append((left, right))
        if verbose:
            print(f"Merge #{len(merges)}: ({left!r}, {right!r}) freq={freq}")

        # add merged token to vocab if new
        if merged not in vocab:
            vocab[merged] = next_id
            next_id += 1

        # apply merge across all tokenized words (naive left-to-right)
        for wb in list(tokenized.keys()):
            toks = tokenized[wb]
            new_toks = []
            i = 0
            while i < len(toks):
                if i + 1 < len(toks) and toks[i] == left and toks[i+1] == right:
                    new_toks.append(merged)
                    i += 2
                else:
                    new_toks.append(toks[i])
                    i += 1
            tokenized[wb] = new_toks

    return vocab, merges, tokenized



def build_tokenizer(
    word_freq: Dict[str, int],
    special_tokens: List[str],
    vocab_limit: int,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    vocab: Dict[int, bytes] = {}
    rev_vocab: Dict[bytes, int] = {}
    merges: List[Tuple[bytes, bytes]] = []
    tot = 256
    # 初始化 0..255 为单字节
    for i in range(256):
        b = bytes([i])
        vocab[i] = b
        rev_vocab[b] = i

    def insert(wb: bytes):
        nonlocal tot
        if wb in rev_vocab:
            return
        rev_vocab[wb] = tot
        vocab[tot] = wb
        tot += 1

    for special_token in special_tokens:
        insert(special_token.encode("utf-8"))

    # bytes 级别频次与辅助结构
    bytes_freq: Dict[bytes, int] = {k.encode("utf-8"): v for k, v in word_freq.items()}
    bytes_min_max: Dict[bytes, Tuple[List[int], List[int]]] = {}
    for k, v in bytes_freq.items():
        m = len(k)
        # mi, ma 初始为每位自身的区间
        bytes_min_max[k] = [*range(m)], [*range(m)]

    # pair -> count, pair -> set((word_bytes, idx))
    cnt_mp: Dict[Tuple[bytes, bytes], int] = {}
    pr_pos_mp: Dict[Tuple[bytes, bytes], Set[Tuple[bytes, int]]] = {}
    for k, v in bytes_freq.items():
        m = len(k)
        for i in range(m - 1):
            pr = (k[i:i+1], k[i+1:i+2])
            cnt_mp[pr] = cnt_mp.get(pr, 0) + v
            s = pr_pos_mp.get(pr)
            if s is None:
                s = set()
                pr_pos_mp[pr] = s
            s.add((k, i))

    tree = SortedList()
    for k, v in cnt_mp.items():
        tree.add((v, k))

    # 原子地增加/删除 pair 的计数及位置记录
    def incr_pr_cnt(pr: Tuple[bytes, bytes], v: int, wd: bytes, idx: int):
        nonlocal tree, cnt_mp, pr_pos_mp
        old = cnt_mp.get(pr, 0)
        # 删除旧的 (old, pr)（若存在）
        try:
            tree.discard((old, pr))
        except Exception:
            pass
        new = old + v
        cnt_mp[pr] = new
        tree.add((new, pr))
        s = pr_pos_mp.get(pr)
        if s is None:
            s = set()
            pr_pos_mp[pr] = s
        s.add((wd, idx))

    def del_pr_cnt(pr: Tuple[bytes, bytes], v: int, wd: bytes, idx: int):
        nonlocal tree, cnt_mp, pr_pos_mp
        old = cnt_mp.get(pr, 0)
        try:
            tree.discard((old, pr))
        except Exception:
            pass
        new = old - v
        if new > 0:
            cnt_mp[pr] = new
            tree.add((new, pr))
        else:
            # 清理 cnt_mp
            cnt_mp.pop(pr, None)
        s = pr_pos_mp.get(pr)
        if s:
            s.discard((wd, idx))
            if not s:
                pr_pos_mp.pop(pr, None)

    # helper: find right/left neighbour token and its starting index (based on bytes_min_max)
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

    # 主循环：弹出最高频 pair，但弹出后基于 pr_pos_mp 校验真实位置/计数，修剪 stale，再合并
    while tot < vocab_limit and len(tree) > 0:
        cnt, pr = tree.pop(-1)

        # 若 cnt 与缓存不一致，跳过（lazy）
        if cnt != cnt_mp.get(pr, 0):
            continue
        if cnt == 0:
            # 清理可能残留的空位置
            pr_pos_mp.pop(pr, None)
            cnt_mp.pop(pr, None)
            continue

        # 从位置集合中校验并保留真实匹配的位置（修剪 stale）
        pos = pr_pos_mp.get(pr, set())
        valid_pos: Set[Tuple[bytes, int]] = set()
        real_cnt = 0
        left_len = len(pr[0])
        right_len = len(pr[1])
        for wd, idx in pos:
            # 验证边界合法并且切片匹配
            if idx >= 0 and idx + left_len + right_len <= len(wd):
                if wd[idx: idx + left_len] == pr[0] and wd[idx + left_len: idx + left_len + right_len] == pr[1]:
                    valid_pos.add((wd, idx))
                    real_cnt += bytes_freq.get(wd, 0)

        # 如果真实计数与 cnt 不符，更新缓存并重新入队（或清理）
        if real_cnt != cnt:
            # 更新缓存与数据结构
            if real_cnt > 0:
                cnt_mp[pr] = real_cnt
                tree.add((real_cnt, pr))
                pr_pos_mp[pr] = valid_pos
            else:
                cnt_mp.pop(pr, None)
                pr_pos_mp.pop(pr, None)
            continue

        # 真正合并 pr（使用 valid_pos）
        merges.append(pr)
        merged_bytes = pr[0] + pr[1]
        # print(f'merge [{pr[0]}] [{pr[1]}] -> {merged_bytes}')
        rev_vocab[merged_bytes] = tot
        vocab[tot] = merged_bytes
        tot += 1

        # 遍历快照位置（避免迭代时被修改）
        for wd, idx in set(valid_pos):
            mcnt = bytes_freq.get(wd, 0)

            # 计算当前 token 区间右边界 ridx（保持你原来的更新方式）
            ridx = bytes_min_max[wd][1][idx]
            if ridx + 1 < len(wd):
                ridx = bytes_min_max[wd][1][ridx+1]
            # 更新 bytes_min_max 把 idx..ridx 归到 merged token 的区间
            for i in range(idx, ridx + 1):
                bytes_min_max[wd][1][i] = ridx
                bytes_min_max[wd][0][i] = idx

            # 找到左/右相邻 token（以及他们在 wd 的起始 idx）
            prev_res = find_prev(wd, idx)
            next_res = find_next(wd, idx)
            pre, pid = (None, None)
            nxt, nid = (None, None)
            if prev_res != (None, None):
                pre, pid = prev_res
            if next_res != (None, None):
                nxt, nid = next_res

            # left pair was (pre, pr[0]) at pid
            if pre:
                del_pr_cnt((pre, pr[0]), mcnt, wd, pid)
                incr_pr_cnt((pre, merged_bytes), mcnt, wd, pid)

            # right pair was (pr[1], nxt) at nid
            if nxt:
                # 注意：删除时使用 nxt 的起始位置 nid
                del_pr_cnt((pr[1], nxt), mcnt, wd, nid)
                # 新的右 pair (merged_bytes, nxt) 在合并后从 idx 开始
                incr_pr_cnt((merged_bytes, nxt), mcnt, wd, idx)

        # 合并完成后，清理被合并 pair 的位置记录（若已无位置）
        pr_pos_mp.pop(pr, None)
        cnt_mp.pop(pr, None)

    return vocab, merges


def bpe_tokenizer_training(
        input_path: str,
        vocab_size: int,
        special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[tuple[bytes, bytes]]]:
    num_worker = os.cpu_count()
    with open(input_path, "rb") as fp:
        boundaries = find_chunk_boundaries(fp, desired_num_chunks=num_worker, split_special_tokens=[x.encode('utf-8') for x in special_tokens])
        word_freq = {}
        with ProcessPoolExecutor(max_workers=num_worker) as ex:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                print(f'{end - start}')
            futures = [ex.submit(read_and_count, *(input_path, start, end - start, special_tokens)) for start, end in
                       zip(boundaries[:-1], boundaries[1:])]
            for fut in as_completed(futures):
                mp = fut.result()
                for k, v in mp.items():
                    word_freq[k] = word_freq.get(k, 0) + v
    vocab, merged = build_tokenizer(word_freq, special_tokens, vocab_size)
    return vocab, merged


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
        vocab0, merged0, _ = brute_force_bpe_with_specials(word_freq, 1000, special_tokens)
        print(len(vocab))
        print(f'merged: {len(merged)}')
        for i in range(500):
            if merged[i] != merged0[i]:
                print(f'merged: {merged[i]} != merged0: {merged0[i]}')


if __name__ == "__main__":
    main()