"""

Pipeline:
  1. Distributional word clusters (PPMI + SVD + KMeans) from the competition text only.
  2. Supertagger (LinearSVC) predicting deprel+head-direction per token, 3 stacked stages,
     jackknifed on train (stage 3 is parse-informed).
  3. Arc-hybrid transition parsers (L2R and R2L), averaged perceptron, dynamic oracle.
  4. Graph-based parsers guided by (stacked on) the transition parsers:
     first-order (Chu-Liu-Edmonds) and second-order sibling (Eisner, McDonald & Pereira 2006),
     trained with averaged cost-augmented MIRA.
  5. Ensemble re-parsing: weighted arc votes + maximum spanning tree.
  6. Dependency labels: LinearSVC over rich arc features.

Usage:
  python main.py --data /kaggle/input/<competition-folder> --out submission.csv
  python main.py --data <folder> --dev # 90/10 held-out estimate of the leaderboard metric
"""
import os, re, sys, time, pickle, zlib, random, argparse
import numpy as np, pandas as pd, scipy.sparse as sp
from collections import Counter
from joblib import Parallel, delayed
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
import warnings; warnings.filterwarnings('ignore')


# common

D_BITS = 23
D = 1 << D_BITS
MASK = np.uint64(D - 1)
PR = np.uint64(0x100000001B3)
GOLD = np.uint64(0x9E3779B97F4A7C15)


def hcomb(tid, arrs):
    """Hash template id + list of int arrays (broadcastable) into [0, D)."""
    h = np.uint64((tid * 0x9E3779B97F4A7C15 + 12345) & 0xFFFFFFFFFFFFFFFF)
    with np.errstate(over='ignore'):
        for a in arrs:
            h = (h ^ a.astype(np.uint64)) * PR
            h = h ^ (h >> np.uint64(31))
        h = h * GOLD
        h = h ^ (h >> np.uint64(29))
    return (h & MASK).astype(np.int64)


def load(path, has_target):
    df = pd.read_csv(path, keep_default_na=False, dtype={'token': str})
    df = df.sort_values(['sentence_id', 'token_id'])
    sents = []
    for sid, g in df.groupby('sentence_id', sort=False):
        s = {'sid': sid, 'ids': g.ID.values, 'tok': list(g.token.values)}
        if has_target:
            hr = g.target.str.split('|', n=1)
            s['head'] = np.array([int(x[0]) for x in hr])
            s['rel'] = [x[1] for x in hr]
        sents.append(s)
    return sents


def shape(w):
    s = re.sub('[A-Z]', 'X', w)
    s = re.sub('[a-z]', 'x', s)
    s = re.sub('[0-9]', 'd', s)
    return re.sub(r'(.)\1+', r'\1\1', s)


PUNCT_RE = re.compile(r'^[^\w]+$')


def norm(w):
    lw = w.lower()
    if re.fullmatch(r'[\d.,:/-]+', lw) and any(c.isdigit() for c in lw):
        return '<NUM>'
    if lw.startswith('http') or lw.startswith('www.'):
        return '<URL>'
    if '@' in lw and '.' in lw:
        return '<EMAIL>'
    return lw


def build_clusters(all_sents, ks=(32, 128, 512), seed=0):
    """Distributional word clusters from raw (unlabelled) train+test text: PPMI + SVD + KMeans."""
    cnt = Counter(norm(w) for s in all_sents for w in s['tok'])
    vocab = [w for w, c in cnt.items()]
    widx = {w: i for i, w in enumerate(vocab)}
    ctx = [w for w, c in cnt.most_common(3000)]
    cidx = {w: i for i, w in enumerate(ctx)}
    C = len(ctx)
    rows, cols = [], []
    for s in all_sents:
        ws = ['<S>'] + [norm(w) for w in s['tok']] + ['</S>']
        for i in range(1, len(ws) - 1):
            r = widx[ws[i]]
            for off, slot in ((-1, 0), (1, 1), (-2, 2), (2, 3)):
                j = i + off
                if 0 <= j < len(ws):
                    c = cidx.get(ws[j])
                    if c is None:
                        c = C if ws[j] in ('<S>', '</S>') else C + 1
                    rows.append(r); cols.append(slot * (C + 2) + c)
    M = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(vocab), 4 * (C + 2))).tocsr()
    M.sum_duplicates()
    tot = M.sum()
    rs = np.asarray(M.sum(1)).ravel(); cs = np.asarray(M.sum(0)).ravel() ** 0.75
    cs = cs / cs.sum()
    M = M.tocoo()
    pmi = np.log(M.data * 1.0 / (rs[M.row] / tot * cs[M.col] * tot) + 1e-12)
    keep = pmi > 0
    P = sp.csr_matrix((pmi[keep], (M.row[keep], M.col[keep])), shape=M.shape)
    X = TruncatedSVD(100, random_state=seed).fit_transform(P)
    X = normalize(X)
    out = {}
    for k in ks:
        km = MiniBatchKMeans(k, random_state=seed, n_init=3, batch_size=4096).fit(X)
        out[k] = {w: int(l) for w, l in zip(vocab, km.labels_)}
    return out

class Vocab:
    def __init__(self):
        self.d = {}

    def __call__(self, x):
        v = self.d.get(x)
        if v is None:
            v = self.d[x] = len(self.d) + 10
        return v


V = Vocab()


def token_attrs(s, wcount, clusters, min_count=2):
    toks = s['tok']
    A = {}
    nw = [norm(w) for w in toks]
    A['raw'] = toks
    A['w'] = [w if wcount.get(w, 0) >= min_count else 'UNK_' + shape(t) for w, t in zip(nw, toks)]
    A['nw'] = nw
    A['shape'] = [shape(t) for t in toks]
    A['s3'] = [w[-3:] for w in nw]
    A['s2'] = [w[-2:] for w in nw]
    for k, cl in clusters.items():
        A['k%d' % k] = [cl.get(w, -1) for w in nw]
    return A


# supertagger features per token
def tagger_feats(A, i):
    n = len(A['w'])
    def g(key, j):
        if j < 0: return '<S>'
        if j >= n: return '</S>'
        return A[key][j]
    w = A['nw'][i]; raw = A['raw'][i]
    f = ['b', 'w=' + g('w', i), 'nw=' + w, 'sh=' + A['shape'][i], 'raw=' + raw]
    for L in (1, 2, 3, 4):
        f.append('s%d=%s' % (L, w[-L:]));
    for L in (1, 2, 3):
        f.append('p%d=%s' % (L, w[:L]))
    f.append('cap=%d%d' % (raw[:1].isupper(), i == 0))
    if '-' in raw: f.append('hy')
    for off in (-3, -2, -1, 1, 2, 3):
        f.append('w%+d=%s' % (off, g('w', i + off)))
    for off in (-2, -1, 1, 2):
        f.append('s3%+d=%s' % (off, g('s3', i + off)))
        f.append('sh%+d=%s' % (off, g('shape', i + off)))
    f.append('ww-1=%s|%s' % (g('w', i - 1), g('w', i)))
    f.append('ww+1=%s|%s' % (g('w', i), g('w', i + 1)))
    f.append('ww-+=%s|%s' % (g('w', i - 1), g('w', i + 1)))
    f.append('ww-2=%s|%s' % (g('w', i - 2), g('w', i - 1)))
    f.append('ww+2=%s|%s' % (g('w', i + 1), g('w', i + 2)))
    for k in (32, 128, 512):
        key = 'k%d' % k
        for off in (-2, -1, 0, 1, 2):
            f.append('%s%+d=%s' % (key, off, g(key, i + off)))
    f.append('k-+=%s|%s|%s' % (g('k128', i - 1), g('k128', i), g('k128', i + 1)))
    f.append('pos=%d' % min(i, 5)); f.append('rpos=%d' % min(n - 1 - i, 5))
    f.append('len=%d' % min(n // 5, 8))
    return f


def tagger_feats_stage2(A, i, ptags):
    """Extra features using predicted tags of neighbours (stacking)."""
    n = len(A['w'])
    def t(j):
        if j < 0: return '<S>'
        if j >= n: return '</S>'
        return ptags[j]
    f = []
    for off in (-3, -2, -1, 1, 2, 3):
        f.append('t%+d=%s' % (off, t(i + off)))
    f.append('t-1t+1=%s|%s' % (t(i - 1), t(i + 1)))
    f.append('t-2t-1=%s|%s' % (t(i - 2), t(i - 1)))
    f.append('t+1t+2=%s|%s' % (t(i + 1), t(i + 2)))
    f.append('t-1w=%s|%s' % (t(i - 1), A['w'][i]))
    f.append('t+1w=%s|%s' % (t(i + 1), A['w'][i]))
    # predicted tags of whole sentence: does a verb-like tag exist left/right
    left = set(ptags[:i]); right = set(ptags[i + 1:])
    for tag in ('root', 'nsubj', 'obj', 'cop', 'aux', 'case', 'mark', 'cc'):
        f.append('L%s=%d' % (tag, any(x.startswith(tag) for x in left)))
        f.append('R%s=%d' % (tag, any(x.startswith(tag) for x in right)))
    return f


def hash_rows(rows, nbits=22):
    """rows: list of list of string features -> CSR binary matrix via python hash (stable via md5-less approach)."""
    import zlib
    m = (1 << nbits) - 1
    indptr = [0]; ind = []
    for r in rows:
        hs = {zlib.crc32(x.encode()) & m for x in r}
        ind.extend(hs); indptr.append(len(ind))
    data = np.ones(len(ind), dtype=np.float32)
    return sp.csr_matrix((data, np.array(ind, dtype=np.int32), np.array(indptr)), shape=(len(rows), m + 1))


# supertagger (2-stage stacked, jackknifed)
def supertag(s):
    return ['root' if r == 'root' else r + ('/L' if h < i + 1 else '/R')
            for i, (h, r) in enumerate(zip(s['head'], s['rel']))]


def _fit_predict(Xa, ya, Xb, C):
    from sklearn.svm import LinearSVC
    m = LinearSVC(C=C, max_iter=2000).fit(Xa, ya)
    return m.predict(Xb)


def jackknife_tags(trn, others, attrs_tr, attrs_ot, C=0.05, K=5, nbits=20, n_jobs=1, seed=0):
    from joblib import Parallel, delayed
    y = np.array([t for s in trn for t in supertag(s)])
    sid_tr = np.concatenate([[k] * len(s['tok']) for k, s in enumerate(trn)])
    fold = np.random.RandomState(seed).randint(0, K, len(trn))[sid_tr]

    def run(feat_tr, feat_ot):
        Xtr = hash_rows(feat_tr, nbits); Xot = hash_rows(feat_ot, nbits)
        jobs = [(Xtr[fold != k], y[fold != k], Xtr[fold == k]) for k in range(K)] + [(Xtr, y, Xot)]
        res = Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(a, b, c, C) for a, b, c in jobs)
        ptr = np.empty(len(y), dtype=object)
        for k in range(K):
            ptr[fold == k] = res[k]
        return ptr, res[K]

    def split(flat, sents):
        out, p = [], 0
        for s in sents:
            n = len(s['tok']); out.append(list(flat[p:p + n])); p += n
        return out

    f1_tr = [tagger_feats(A, i) for A in attrs_tr for i in range(len(A['w']))]
    f1_ot = [tagger_feats(A, i) for A in attrs_ot for i in range(len(A['w']))]
    p_tr, p_ot = run(f1_tr, f1_ot)
    t_tr, t_ot = split(p_tr, trn), split(p_ot, others)
    print('  stage1 jackknife acc %.4f' % (p_tr == y).mean(), flush=True)
    f2_tr = [f + tagger_feats_stage2(A, i, t) for A, t, base in zip(attrs_tr, t_tr, [None] * len(t_tr))
             for i, f in zip(range(len(A['w'])), [tagger_feats(A, j) for j in range(len(A['w']))])]
    f2_ot = [f + tagger_feats_stage2(A, i, t) for A, t in zip(attrs_ot, t_ot)
             for i, f in zip(range(len(A['w'])), [tagger_feats(A, j) for j in range(len(A['w']))])]
    p_tr, p_ot = run(f2_tr, f2_ot)
    print('  stage2 jackknife acc %.4f' % (p_tr == y).mean(), flush=True)
    return split(p_tr, trn), split(p_ot, others)


def tagger_feats_parse(A, i, ptags, guides):
    """Stage-3 features: parse-informed (stacked on jackknifed parser output)."""
    n = len(A['w'])
    W = ['<ROOT>'] + A['w']; T = ['<ROOT>'] + list(ptags); Cc = ['<ROOT>'] + [t.split('/')[0] for t in ptags]
    d = i + 1
    f = []
    for gi, gh in enumerate(guides):
        h = int(gh[d]); dr = 'L' if h < d else 'R'; dist = min(abs(h - d), 6)
        kids = [k for k in range(1, n + 1) if gh[k] == d]
        f += ['g%d_hw=%s' % (gi, W[h]), 'g%d_ht=%s' % (gi, T[h]), 'g%d_dir=%s%d' % (gi, dr, dist),
              'g%d_hcdir=%s|%s' % (gi, Cc[h], dr), 'g%d_hwdw=%s|%s' % (gi, W[h], W[d]),
              'g%d_htdt=%s|%s|%s' % (gi, T[h], T[d], dr), 'g%d_nk=%d' % (gi, min(len(kids), 4)),
              'g%d_root=%d' % (gi, h == 0), 'g%d_hh=%s' % (gi, Cc[int(gh[h])] if h > 0 else 'R')]
        for k in kids:
            f.append('g%d_kc=%s|%s' % (gi, Cc[k], 'L' if k < d else 'R'))
            if Cc[k] in ('case', 'mark', 'cop', 'aux', 'aux:pass', 'det', 'cc', 'nsubj', 'nsubj:pass', 'expl', 'obj'):
                f.append('g%d_kw=%s|%s' % (gi, Cc[k], W[k]))
        sib = [k for k in range(1, n + 1) if gh[k] == h and k != d]
        for k in sib:
            f.append('g%d_sib=%s|%s' % (gi, Cc[k], 'B' if k < d else 'A'))
    f.append('agree=%d' % (guides[0][d] == guides[1][d]))
    return f


# parser

NEG = -1e18

# eisner
def eisner(S):
    N = S.shape[0]
    Cr = np.full((N, N), NEG); Cl = np.full((N, N), NEG)
    Ir = np.full((N, N), NEG); Il = np.full((N, N), NEG)
    bCr = np.zeros((N, N), np.int32); bCl = np.zeros((N, N), np.int32)
    bIr = np.zeros((N, N), np.int32); bIl = np.zeros((N, N), np.int32)
    idx = np.arange(N)
    Cr[idx, idx] = 0; Cl[idx, idx] = 0
    for w in range(1, N):
        for s in range(0, N - w):
            t = s + w
            v = Cr[s, s:t] + Cl[s + 1:t + 1, t]
            if s == 0:
                Ir[s, t] = v[0] + S[s, t]; bIr[s, t] = s
            else:
                k = int(np.argmax(v)); m = v[k]
                Ir[s, t] = m + S[s, t]; bIr[s, t] = s + k
                Il[s, t] = m + S[t, s]; bIl[s, t] = s + k
            v = Cl[s, s:t] + Il[s:t, t]
            k = int(np.argmax(v)); Cl[s, t] = v[k]; bCl[s, t] = s + k
            v = Ir[s, s + 1:t + 1] + Cr[s + 1:t + 1, t]
            k = int(np.argmax(v)); Cr[s, t] = v[k]; bCr[s, t] = s + 1 + k
    head = np.zeros(N, np.int64)
    stack = [(0, N - 1, 'Cr')]
    while stack:
        s, t, typ = stack.pop()
        if s == t:
            continue
        if typ == 'Cr':
            k = bCr[s, t]; stack.append((s, k, 'Ir')); stack.append((k, t, 'Cr'))
        elif typ == 'Cl':
            k = bCl[s, t]; stack.append((s, k, 'Cl')); stack.append((k, t, 'Il'))
        elif typ == 'Ir':
            head[t] = s; k = bIr[s, t]; stack.append((s, k, 'Cr')); stack.append((k + 1, t, 'Cl'))
        else:
            head[s] = t; k = bIl[s, t]; stack.append((s, k, 'Cr')); stack.append((k + 1, t, 'Cl'))
    return head


# Chu-Liu-Edmonds
def _cle(S):
    """S[h,d]; returns heads (head[0]=-1) of max spanning arborescence rooted at 0 (multi-root allowed)."""
    N = S.shape[0]
    S = S.copy()
    np.fill_diagonal(S, NEG); S[:, 0] = NEG
    head = S.argmax(0); head[0] = -1
    # find cycle
    vis = np.zeros(N, int) - 1
    for st in range(1, N):
        v = st; path = []
        while v != 0 and vis[v] == -1:
            vis[v] = st; path.append(v); v = head[v]
        if v != 0 and vis[v] == st:
            cyc = [v]; u = head[v]
            while u != v:
                cyc.append(u); u = head[u]
            break
    else:
        return head
    cyc = np.array(cyc); inc = np.zeros(N, bool); inc[cyc] = True
    non = np.where(~inc)[0]  # includes 0
    cyc_score = S[head[cyc], cyc].sum()
    M = len(non) + 1  # new node index M-1 = contracted
    S2 = np.full((M, M), NEG)
    S2[:M - 1, :M - 1] = S[np.ix_(non, non)]
    # arcs into cycle: from non u to cycle v: S[u,v] - S[head[v],v]
    into = S[np.ix_(non, cyc)] - S[head[cyc], cyc][None, :]
    bi = into.argmax(1); S2[:M - 1, M - 1] = into[np.arange(len(non)), bi]
    # arcs out of cycle: from cycle v to non u
    out = S[np.ix_(cyc, non)]
    bo = out.argmax(0); S2[M - 1, :M - 1] = out[bo, np.arange(len(non))]
    h2 = _cle(S2)
    res = head.copy()
    for j, u in enumerate(non):
        if u == 0:
            continue
        hj = h2[j]
        res[u] = cyc[bo[j]] if hj == M - 1 else non[hj]
    # cycle entry
    u = non[h2[M - 1]]; jv = list(non).index(u)
    v = cyc[bi[jv]]
    res[v] = u
    return res


def cle(S):
    """Single-root MST: try best root-child choice by penalizing root arcs if needed."""
    h = _cle(S)
    roots = np.where(h[1:] == 0)[0] + 1
    if len(roots) <= 1:
        h[0] = 0; return h
    best, bs = None, NEG
    for r in roots:
        S2 = S.copy(); S2[0, :] = NEG; S2[0, r] = S[0, r]
        h2 = _cle(S2); sc = S[h2[1:], np.arange(1, len(h2))].sum()
        if sc > bs:
            bs, best = sc, h2
    best[0] = 0
    return best


# arc features
def dist_bucket(dd):
    a = np.abs(dd)
    b = np.where(a <= 5, a, np.where(a <= 7, 6, np.where(a <= 10, 7, np.where(a <= 15, 8, np.where(a <= 25, 9, 10)))))
    return np.sign(dd) * b + 20


CLAUSE = {'root', 'ccomp', 'advcl', 'xcomp', 'acl', 'acl:relcl', 'parataxis', 'csubj', 'conj', 'advcl:relcl'}
BETWEEN_CLASSES = [
    ('clause', lambda c, w: c in CLAUSE),
    ('punct', lambda c, w: c == 'punct'),
    ('comma', lambda c, w: w in (',', ';', ':', '-', '--')),
    ('cc', lambda c, w: c == 'cc'),
    ('nsubj', lambda c, w: c.startswith('nsubj')),
    ('mark', lambda c, w: c == 'mark'),
    ('case', lambda c, w: c == 'case'),
    ('auxcop', lambda c, w: c.startswith('aux') or c == 'cop'),
    ('obj', lambda c, w: c in ('obj', 'iobj')),
]

H_T = [[('h', 'w')], [('h', 't')], [('h', 'w'), ('h', 't')], [('h', 'k512')], [('h', 'k128'), ('h', 't')],
       [('h', 'c')], [('h', 's3'), ('h', 't')]]
D_T = [[('d', 'w')], [('d', 't')], [('d', 'w'), ('d', 't')], [('d', 'k512')], [('d', 'k128'), ('d', 't')],
       [('d', 'c')], [('d', 's3'), ('d', 't')]]
BI_T = [
    [('h', 'w'), ('h', 't'), ('d', 'w'), ('d', 't')],
    [('h', 't'), ('d', 'w'), ('d', 't')], [('h', 'w'), ('d', 'w'), ('d', 't')],
    [('h', 'w'), ('h', 't'), ('d', 't')], [('h', 'w'), ('h', 't'), ('d', 'w')],
    [('h', 'w'), ('d', 'w')], [('h', 't'), ('d', 't')], [('h', 'c'), ('d', 'c')],
    [('h', 'k512'), ('d', 'k512')], [('h', 'k128'), ('d', 'k128')], [('h', 'k32'), ('d', 'k32')],
    [('h', 'k128'), ('d', 't')], [('h', 't'), ('d', 'k128')],
    [('h', 'w'), ('d', 'k128')], [('h', 'k128'), ('d', 'w')],
    [('h', 'w'), ('d', 't')], [('h', 't'), ('d', 'w')],
    [('h', 't'), ('h', 'k32'), ('d', 't'), ('d', 'k32')],
]
CTX_T = []
for a in ('t', 'k32'):
    CTX_T += [[('h', a), ('h+1', a), ('d-1', a), ('d', a)], [('h-1', a), ('h', a), ('d-1', a), ('d', a)],
              [('h', a), ('h+1', a), ('d', a), ('d+1', a)], [('h-1', a), ('h', a), ('d', a), ('d+1', a)]]
CTX_T += [[('h', 't'), ('d', 't'), ('d-1', 't')], [('h', 't'), ('d', 't'), ('d+1', 't')],
          [('h', 't'), ('h-1', 't'), ('d', 't')], [('h', 't'), ('h+1', 't'), ('d', 't')],
          [('h', 'w'), ('d', 't'), ('d-1', 't')], [('h', 'w'), ('d', 't'), ('d+1', 't')],
          [('h', 't'), ('d', 'w'), ('h-1', 't')], [('h', 't'), ('d', 'w'), ('h+1', 't')],
          [('h', 'c'), ('d', 'c'), ('d-1', 'w')], [('h', 'c'), ('d', 'c'), ('d+1', 'w')],
          [('h', 'c'), ('d', 'c'), ('h-1', 'w')], [('h', 'c'), ('d', 'c'), ('h+1', 'w')]]
ALL_T = H_T + D_T + BI_T + CTX_T


class ArcFeaturizer:
    def __init__(self, vocab):
        self.V = vocab

    def sent_arrays(self, A, tags):
        """Integer attribute arrays of length N=n+1 (pos 0 = ROOT), plus shifted copies."""
        V = self.V
        n = len(A['w'])
        coarse = [t.split('/')[0] for t in tags]
        base = {
            'w': ['<ROOT>'] + A['w'],
            't': ['<ROOT>'] + list(tags),
            'c': ['<ROOT>'] + coarse,
            's3': ['<ROOT>'] + A['s3'],
            'k32': ['<ROOT>'] + ['k%s' % x for x in A['k32']],
            'k128': ['<ROOT>'] + ['k%s' % x for x in A['k128']],
            'k512': ['<ROOT>'] + ['k%s' % x for x in A['k512']],
        }
        arr = {}
        for k, v in base.items():
            a = np.array([V(k + ':' + str(x)) for x in v], dtype=np.int64)
            arr[k] = a
            arr[k + '-1'] = np.concatenate([[1], a[:-1]])
            arr[k + '+1'] = np.concatenate([a[1:], [2]])
        # between-count prefix sums
        words = ['<ROOT>'] + A['raw']
        cs = ['<ROOT>'] + coarse
        pref = []
        for name, fn in BETWEEN_CLASSES:
            ind = np.array([1 if (i > 0 and fn(cs[i], words[i])) else 0 for i in range(n + 1)])
            pref.append(np.concatenate([[0], np.cumsum(ind)]))
        arr['_pref'] = pref
        arr['_n'] = n
        return arr

    def feats(self, arr):
        N = arr['_n'] + 1
        hi = np.arange(N)[:, None]; di = np.arange(N)[None, :]
        dd = dist_bucket(hi - di)
        out = []
        tid = 0
        def comp(side, att):
            pos = side[0]; off = side[1:]
            a = arr[att + off] if off else arr[att]
            return a[:, None] if pos == 'h' else a[None, :]
        for tpl in ALL_T:
            cs = [comp(s, a) for s, a in tpl]
            out.append(np.broadcast_to(hcomb(tid, cs), (N, N))); tid += 1
            out.append(hcomb(tid, cs + [dd])); tid += 1
        # distance-only and with tags
        out.append(np.broadcast_to(hcomb(tid, [dd]), (N, N))); tid += 1
        out.append(hcomb(tid, [dd, arr['t'][:, None]])); tid += 1
        out.append(hcomb(tid, [dd, arr['t'][None, :]])); tid += 1
        out.append(hcomb(tid, [dd, arr['w'][None, :]])); tid += 1
        out.append(hcomb(tid, [dd, arr['w'][:, None]])); tid += 1
        # between features
        lo = np.minimum(hi, di); hi2 = np.maximum(hi, di)
        for ci, p in enumerate(arr['_pref']):
            cnt = p[hi2] - p[lo + 1]
            cnt = np.minimum(np.maximum(cnt, 0), 3)
            out.append(hcomb(tid, [arr['t'][:, None], arr['t'][None, :], cnt, np.sign(hi - di)])); tid += 1
            out.append(hcomb(tid, [arr['c'][:, None], arr['c'][None, :], cnt, dd])); tid += 1
        # guide features (stacking): agreement with other parsers' predicted heads
        if arr.get('_guides'):
            G = np.zeros((N, N), np.int64)
            for gi, gh in enumerate(arr['_guides']):
                M = np.zeros((N, N), np.int64); M[gh[1:], np.arange(1, N)] = 1
                G += M << gi
                # does h's guide-head equal d (reversal), sibling-in-guide
                R = np.zeros((N, N), np.int64); R[np.arange(1, N), gh[1:]] = 1
                G += R << (gi + 4)
            tid = 900
            for comps in ([], [arr['t'][:, None], arr['t'][None, :]], [dd], [arr['w'][:, None]], [arr['w'][None, :]],
                          [arr['c'][:, None], arr['c'][None, :], dd], [arr['t'][None, :]], [arr['t'][:, None]]):
                out.append(hcomb(tid, [G] + comps)); tid += 1
        # level-0 graph-parser tree (Martins et al. 2008 stacking): head/reverse/sibling/grandparent indicators
        if arr.get('_l0') is not None:
            L = np.asarray(arr['_l0']); t = arr['t']; c = arr['c']; w = arr['w']
            Lh = L[hi]; Ld = L[di]                           # L-head of h, of d  (L[0] = 0)
            Mt = (Ld == hi) & (di > 0)
            code = (Mt.astype(np.int64) + 2 * (Lh == di) + 4 * ((Lh == Ld) & (hi > 0)) + 8 * (L[Ld] == hi) * (Ld > 0)
                    + 16 * (L[Lh] == di) * (hi > 0))
            nk = np.bincount(L[1:], minlength=N)[:N]; nk = np.minimum(nk, 4)
            tid = 950
            for comps in ([], [t[:, None], t[None, :]], [dd], [c[:, None], c[None, :], dd], [w[:, None]], [w[None, :]],
                          [t[Lh], t[:, None], t[None, :]], [nk[None, :], t[None, :]], [t[Ld], t[None, :]],
                          [w[:, None], t[None, :]], [t[:, None], w[None, :]]):
                out.append(np.broadcast_to(hcomb(tid, [code] + comps), (N, N))); tid += 1
        return np.stack(out)  # (T, N, N)


# averaged MIRA/perceptron
class ArcModel:
    def __init__(self):
        self.W = np.zeros(D, np.float32)
        self.U = np.zeros(D, np.float64)
        self.c = 1

    def score(self, F, W=None):
        W = self.W if W is None else W
        return W[F].sum(0)

    def update(self, F, gold, pred, C=0.1, S=None):
        n = len(gold) - 1
        d = np.arange(1, n + 1)
        wrong = gold[1:] != pred[1:]
        if not wrong.any():
            self.c += 1; return 0
        dsel = d[wrong]
        g = F[:, gold[dsel], dsel].ravel(); p = F[:, pred[dsel], dsel].ravel()
        ids = np.concatenate([g, p]); vals = np.concatenate([np.ones(len(g)), -np.ones(len(p))])
        u, inv = np.unique(ids, return_inverse=True)
        delta = np.zeros(len(u)); np.add.at(delta, inv, vals)
        nz = delta != 0
        u, delta = u[nz], delta[nz]
        if len(u) == 0:
            self.c += 1; return 0
        loss = wrong.sum()
        margin = (self.W[u].astype(np.float64) * delta).sum()
        tau = min(C, max(0.0, (loss - margin) / (delta ** 2).sum()))
        self.W[u] += (tau * delta).astype(np.float32)
        self.U[u] += self.c * tau * delta
        self.c += 1
        return loss

    def averaged(self):
        return (self.W - self.U / self.c).astype(np.float32)


def decode(S, nonproj=True):
    return cle(S) if nonproj else eisner(S)


"""Second-order (sibling) graph-based parser: McDonald & Pereira (2006) Eisner with sibling factors."""

FIRST = 7  # placeholder attribute id for "no inner sibling"

SIB_T = [  # each: list of (role, attr); roles h, s, m
    [('s', 't'), ('m', 't')],
    [('h', 't'), ('s', 't'), ('m', 't')],
    [('s', 'w'), ('m', 't')],
    [('s', 't'), ('m', 'w')],
    [('s', 'w'), ('m', 'w')],
    [('h', 'c'), ('s', 'c'), ('m', 'c')],
    [('s', 'k128'), ('m', 'k128')],
    [('h', 'w'), ('s', 't'), ('m', 't')],
    [('s', 'c'), ('m', 'c'), 'smdist'],
    [('h', 't'), ('s', 'k32'), ('m', 'k32')],
    [('h', 'k128'), ('s', 'c'), ('m', 'c')],
    [('s', 'c'), ('m', 'c'), ('h', 'c'), 'hmdist'],
]


def sib_ids(arr, H, S, M, tid):
    first = (S == H)
    dr = np.sign(M - H)
    out = []
    for item in SIB_T[tid]:
        if item == 'smdist':
            out.append(np.where(first, 99, dist_bucket(M - S)))
        elif item == 'hmdist':
            out.append(dist_bucket(M - H))
        else:
            role, att = item
            a = arr[att]
            if role == 'h': out.append(a[H])
            elif role == 'm': out.append(a[M])
            else: out.append(np.where(first, FIRST, a[S]))
    out.append(dr)
    return hcomb(5000 + tid, out)


def l0_tri(arr):
    if '_l0tri' not in arr:
        N = arr['_n'] + 1
        T = np.zeros((N, N, N), bool); tr = triples(np.asarray(arr['_l0']))
        T[tr[:, 0], tr[:, 1], tr[:, 2]] = True
        arr['_l0tri'] = T
    return arr['_l0tri']


def l0_sib_ids(arr, H, S, M):
    """Stacked sibling features: is (h, s, m) an adjacent-sibling triple in the level-0 tree."""
    L = np.asarray(arr['_l0']); T = l0_tri(arr)
    I = T[H, S, M].astype(np.int64); J = (L[M] == H).astype(np.int64)
    dr = np.sign(M - H); first = (S == H); c = arr['c']
    sc = np.where(first, FIRST, c[S])
    return [hcomb(6000, [I, J, dr]), hcomb(6001, [I, J, sc, c[M], dr]), hcomb(6002, [I, J, c[H], c[M], dr])]


def sib_score(arr, W):
    N = arr['_n'] + 1
    H = np.arange(N)[:, None, None]; S = np.arange(N)[None, :, None]; M = np.arange(N)[None, None, :]
    tot = np.zeros((N, N, N), np.float32)
    for tid in range(len(SIB_T)):
        tot += W[sib_ids(arr, H, S, M, tid)]
    if arr.get('_l0') is not None:
        for ids in l0_sib_ids(arr, H, S, M):
            tot += W[ids]
    return tot


def triples(head):
    """(h, s, m) sibling triples of a tree; s == h means first child on that side."""
    N = len(head)
    ch = [[] for _ in range(N)]
    for m in range(1, N): ch[head[m]].append(m)
    out = []
    for h in range(N):
        right = sorted(c for c in ch[h] if c > h)
        left = sorted((c for c in ch[h] if c < h), reverse=True)
        for side in (right, left):
            prev = h
            for m in side:
                out.append((h, prev, m)); prev = m
    return np.array(out, dtype=np.int64).reshape(-1, 3)


def triple_feats(arr, T):
    H, S, M = T[:, 0], T[:, 1], T[:, 2]
    parts = [sib_ids(arr, H, S, M, tid) for tid in range(len(SIB_T))]
    if arr.get('_l0') is not None:
        parts += l0_sib_ids(arr, H, S, M)
    return np.concatenate(parts)


def eisner2(A, Sib):
    N = A.shape[0]
    Cr = np.full((N, N), NEG); Cl = np.full((N, N), NEG)
    Ir = np.full((N, N), NEG); Il = np.full((N, N), NEG); SB = np.full((N, N), NEG)
    bCr = np.zeros((N, N), np.int32); bCl = np.zeros((N, N), np.int32)
    bIr = np.zeros((N, N), np.int32); bIl = np.zeros((N, N), np.int32); bSB = np.zeros((N, N), np.int32)
    ix = np.arange(N); Cr[ix, ix] = 0; Cl[ix, ix] = 0
    for w in range(1, N):
        for s in range(0, N - w):
            t = s + w
            if s > 0:
                v = Cr[s, s:t] + Cl[s + 1:t + 1, t]
                k = int(np.argmax(v)); SB[s, t] = v[k]; bSB[s, t] = s + k
            # s -> t
            best = Cl[s + 1, t] + Sib[s, s, t]; br = s
            if s > 0 and t - s > 1:
                v = Ir[s, s + 1:t] + SB[s + 1:t, t] + Sib[s, s + 1:t, t]
                k = int(np.argmax(v))
                if v[k] > best: best = v[k]; br = s + 1 + k
            Ir[s, t] = best + A[s, t]; bIr[s, t] = br
            # t -> s
            if s > 0:
                best = Cr[s, t - 1] + Sib[t, t, s]; br = t
                if t - s > 1:
                    v = SB[s, s + 1:t] + Il[s + 1:t, t] + Sib[t, s + 1:t, s]
                    k = int(np.argmax(v))
                    if v[k] > best: best = v[k]; br = s + 1 + k
                Il[s, t] = best + A[t, s]; bIl[s, t] = br
            v = Cl[s, s:t] + Il[s:t, t]
            k = int(np.argmax(v)); Cl[s, t] = v[k]; bCl[s, t] = s + k
            v = Ir[s, s + 1:t + 1] + Cr[s + 1:t + 1, t]
            k = int(np.argmax(v)); Cr[s, t] = v[k]; bCr[s, t] = s + 1 + k
    head = np.zeros(N, np.int64)
    st = [(0, N - 1, 0)]  # 0 Cr,1 Cl,2 Ir,3 Il,4 SB
    while st:
        s, t, ty = st.pop()
        if s == t: continue
        if ty == 0:
            k = bCr[s, t]; st += [(s, k, 2), (k, t, 0)]
        elif ty == 1:
            k = bCl[s, t]; st += [(s, k, 1), (k, t, 3)]
        elif ty == 2:
            head[t] = s; r = bIr[s, t]
            if r == s: st.append((s + 1, t, 1))
            else: st += [(s, r, 2), (r, t, 4)]
        elif ty == 3:
            head[s] = t; r = bIl[s, t]
            if r == t: st.append((s, t - 1, 0))
            else: st += [(s, r, 4), (r, t, 3)]
        else:
            k = bSB[s, t]; st += [(s, k, 0), (k + 1, t, 1)]
    return head


def tree_score2(A, Sib, head):
    T = triples(head)
    return A[head[1:], np.arange(1, len(head))].sum() + Sib[T[:, 0], T[:, 1], T[:, 2]].sum()


"""Arc-hybrid transition parser, averaged perceptron, dynamic oracle with exploration."""

SH, LA, RA = 0, 1, 2
NB = 22
MASKT = (1 << NB) - 1


class TransParser:
    def __init__(self):
        self.W = np.zeros((1 << NB, 3), np.float32)
        self.U = np.zeros((1 << NB, 3), np.float64)
        self.c = 1

    
    @staticmethod
    def feats(arr, stack, buf, n, lc, rc, nl, nr):
        w = arr['w']; t = arr['t']; k = arr['k128']; c = arr['c']
        def g(a, i): return a[i] if i >= 0 else -5
        s0 = stack[-1] if stack else -1
        s1 = stack[-2] if len(stack) > 1 else -1
        s2 = stack[-3] if len(stack) > 2 else -1
        b0 = buf if buf <= n else -1
        b1 = buf + 1 if buf + 1 <= n else -1
        b2 = buf + 2 if buf + 2 <= n else -1
        s0l = lc[s0] if s0 >= 0 else -1; s0r = rc[s0] if s0 >= 0 else -1
        s1l = lc[s1] if s1 >= 0 else -1; s1r = rc[s1] if s1 >= 0 else -1
        b0l = lc[b0] if b0 >= 0 else -1
        s0w, s0t, s1w, s1t, b0w, b0t = g(w, s0), g(t, s0), g(w, s1), g(t, s1), g(w, b0), g(t, b0)
        b1w, b1t, b2t, s2t = g(w, b1), g(t, b1), g(t, b2), g(t, s2)
        s0k, s1k, b0k, b1k = g(k, s0), g(k, s1), g(k, b0), g(k, b1)
        dist = min(b0 - s0, 8) if (b0 >= 0 and s0 >= 0) else -1
        dist1 = min(s0 - s1, 8) if (s1 >= 0 and s0 >= 0) else -1
        s0lt, s0rt, s1lt, s1rt, b0lt = g(c, s0l), g(c, s0r), g(c, s1l), g(c, s1r), g(c, b0l)
        vs0 = (nl[s0], nr[s0]) if s0 >= 0 else (-1, -1)
        vb0 = nl[b0] if b0 >= 0 else -1
        f = [
            (1, s0w), (2, s0t), (3, s0w, s0t), (4, s1w), (5, s1t), (6, s1w, s1t), (7, b0w), (8, b0t), (9, b0w, b0t),
            (10, b1w), (11, b1t), (12, b1w, b1t), (13, b2t), (14, s2t), (15, s0k), (16, b0k), (17, s1k),
            (20, s0w, s0t, b0w, b0t), (21, s0w, s0t, b0w), (22, s0w, b0w, b0t), (23, s0w, s0t, b0t), (24, s0t, b0w, b0t),
            (25, s0w, b0w), (26, s0t, b0t), (27, b0t, b1t), (28, s1w, s0w), (29, s1t, s0t), (30, s1w, s0t), (31, s1t, s0w),
            (40, b0t, b1t, b2t), (41, s0t, b0t, b1t), (42, s1t, s0t, b0t), (43, s2t, s1t, s0t), (44, s1t, s0t, b1t),
            (45, s0k, b0k), (46, s1k, s0k), (47, s0k, b0t), (48, s0t, b0k), (49, s0k, b0k, b1k),
            (50, s0w, dist), (51, s0t, dist), (52, b0w, dist), (53, b0t, dist), (54, s0w, b0w, dist), (55, s0t, b0t, dist),
            (56, s1t, s0t, dist1), (57, s1w, s0w, dist1),
            (60, s0w, vs0[0], vs0[1]), (61, s0t, vs0[0], vs0[1]), (62, b0w, vb0), (63, b0t, vb0), (64, s1t, s0t, vs0[0], vs0[1]),
            (70, s0t, s0lt), (71, s0t, s0rt), (72, s0t, s0lt, s0rt), (73, s1t, s1lt), (74, s1t, s1rt), (75, b0t, b0lt),
            (76, s0t, s0lt, b0t), (77, s0t, s0rt, b0t), (78, s1t, s1rt, s0t), (79, s0w, s0rt, b0t), (80, b0t, b0lt, s0t),
            (81, s1t, s0t, s0rt), (82, s1t, s0t, s0lt), (83, s0w, s0lt), (84, s0w, s0rt),
            (90, len(stack) > 2, b0 < 0, s1 == 0),
        ]
        return [hash(x) & MASKT for x in f]

    def scores(self, idx, W=None):
        W = self.W if W is None else W
        return W[idx].sum(0)

    
    @staticmethod
    def legal(stack, buf, n):
        L = [False, False, False]
        if buf <= n: L[SH] = True
        if stack and stack[-1] != 0 and buf <= n: L[LA] = True
        if len(stack) > 1:
            if stack[-2] != 0 or buf > n: L[RA] = True
        return L

    @staticmethod
    def costs(stack, buf, n, gold, gchildren_in, L):
        """Dynamic oracle costs (Goldberg & Nivre 2013) for arc-hybrid. gold: head array."""
        INF = 10 ** 6
        C = [INF, INF, INF]
        s0 = stack[-1] if stack else None
        s1 = stack[-2] if len(stack) > 1 else None
        if L[SH]:
            b = buf; c = 0
            for k in stack[:-1]:
                if gold[b] == k: c += 1
            for k in stack:
                if gold[k] == b: c += 1
            C[SH] = c
        if L[LA]:
            c = 0
            if s1 is not None and gold[s0] == s1: c += 1
            if gold[s0] > buf: c += 1
            c += gchildren_in(s0, buf)
            C[LA] = c
        if L[RA]:
            c = 0
            if gold[s0] >= buf and gold[s0] <= n: c += 1
            c += gchildren_in(s0, buf)
            C[RA] = c
        return C

    # parsing
    def parse(self, arr, W=None, gold=None, explore=0.0, rng=None):
        n = arr['_n']
        stack = [0]; buf = 1
        head = np.zeros(n + 1, np.int64)
        lc = [-1] * (n + 1); rc = [-1] * (n + 1); nl = [0] * (n + 1); nr = [0] * (n + 1)
        if gold is not None:
            remaining = [0] * (n + 1)
            ch = [[] for _ in range(n + 1)]
            for d in range(1, n + 1): ch[gold[d]].append(d)
            def gin(x, b):
                return sum(1 for y in ch[x] if y >= b and head[y] == 0 and y != 0)
            updates = 0
        while not (buf > n and len(stack) == 1):
            L = self.legal(stack, buf, n)
            idx = self.feats(arr, stack, buf, n, lc, rc, nl, nr)
            sc = self.scores(idx, W)
            pred = max((a for a in range(3) if L[a]), key=lambda a: sc[a])
            act = pred
            if gold is not None:
                C = self.costs(stack, buf, n, gold, gin, L)
                mc = min(C)
                zero = [a for a in range(3) if L[a] and C[a] == mc]
                if pred not in zero:
                    best_ok = max(zero, key=lambda a: sc[a])
                    self.W[idx, best_ok] += 1; self.W[idx, pred] -= 1
                    self.U[idx, best_ok] += self.c; self.U[idx, pred] -= self.c
                    act = pred if (rng.random() < explore) else best_ok
                self.c += 1
            if act == SH:
                stack.append(buf); buf += 1
            elif act == LA:
                s0 = stack.pop(); head[s0] = buf
                if lc[buf] == -1 or s0 < lc[buf]: lc[buf] = s0
                nl[buf] += 1
            else:
                s0 = stack.pop(); h = stack[-1]; head[s0] = h
                if rc[h] == -1 or s0 > rc[h]: rc[h] = s0
                nr[h] += 1
        return head

    def averaged(self):
        return (self.W - self.U / self.c).astype(np.float32)


def is_projective(h):
    n = len(h) - 1
    for d in range(1, n + 1):
        a, b = sorted((d, h[d]))
        for e in range(1, n + 1):
            if e == d: continue
            x, y = sorted((e, h[e]))
            if a < x < b < y or x < a < y < b:
                return False
    return True


def reverse_arr(arr):
    """Reverse token order (keeping ROOT at 0) for a right-to-left parser."""
    n = arr['_n']
    perm = np.concatenate([[0], np.arange(n, 0, -1)])
    out = {k: (v[perm] if isinstance(v, np.ndarray) and len(v) == n + 1 else v) for k, v in arr.items()}
    return out, perm


def train_trans(arrs, golds, epochs=10, explore=0.9, seed=0, reverse=False, log=None):
    p = TransParser(); rng = random.Random(seed)
    data = []
    for a, g in zip(arrs, golds):
        if reverse:
            a, perm = reverse_arr(a)
            inv = np.argsort(perm); g = inv[g[perm]]  # new index of old head
        if is_projective(g):
            data.append((a, g))
    for ep in range(epochs):
        rng.shuffle(data)
        for a, g in data:
            p.parse(a, gold=g, explore=explore if ep > 0 else 0.0, rng=rng)
        if log: log(ep)
    p.Wavg = p.averaged()
    return p


def trans_predict(p, arr, reverse=False):
    if not reverse:
        return p.parse(arr, W=p.Wavg)
    a, perm = reverse_arr(arr)
    h = p.parse(a, W=p.Wavg)
    # h indexes reversed positions; map back
    n = len(h) - 1
    out = np.zeros(n + 1, np.int64)
    for j in range(1, n + 1):
        out[perm[j]] = perm[h[j]]
    return out


def delta_update(m, ids_g, ids_p, loss, C):
    ids = np.concatenate([ids_g, ids_p]); vals = np.concatenate([np.ones(len(ids_g)), -np.ones(len(ids_p))])
    u, inv = np.unique(ids, return_inverse=True)
    delta = np.zeros(len(u)); np.add.at(delta, inv, vals)
    nz = delta != 0; u, delta = u[nz], delta[nz]
    if len(u) == 0: m.c += 1; return
    margin = (m.W[u].astype(np.float64) * delta).sum()
    tau = min(C, max(0.0, (loss - margin) / (delta ** 2).sum()))
    m.W[u] += (tau * delta).astype(np.float32); m.U[u] += m.c * tau * delta; m.c += 1

def parse2(fz, arr, W):
    F = fz.feats(arr); A = W[F].sum(0); Sb = sib_score(arr, W)
    return eisner2(A, Sb), A, Sb

def train_epoch(m, fz, arrs, golds, seed, C=0.05, maxlen=1000):
    order = np.random.RandomState(seed).permutation(len(arrs)); err = tot = 0
    for i in order:
        a = arrs[i]; g = golds[i]
        if a['_n'] > maxlen: continue
        F = fz.feats(a); A = m.W[F].sum(0); Sb = sib_score(a, m.W)
        A = A + 1.0; A[g[1:], np.arange(1, len(g))] -= 1.0
        p = eisner2(A, Sb)
        loss = (p[1:] != g[1:]).sum(); err += loss; tot += len(g) - 1
        Tg = triples(g); Tp = triples(p)
        # only differing triples/arcs
        sg = set(map(tuple, Tg)); spp = set(map(tuple, Tp))
        dg = np.array([t for t in sg - spp], dtype=np.int64).reshape(-1, 3)
        dp = np.array([t for t in spp - sg], dtype=np.int64).reshape(-1, 3)
        if len(dg) == 0 and len(dp) == 0 and loss == 0: m.c += 1; continue
        d = np.arange(1, len(g)); w = g[1:] != p[1:]
        ig = [F[:, g[1:][w], d[w]].ravel()]; ip = [F[:, p[1:][w], d[w]].ravel()]
        if len(dg): ig.append(triple_feats(a, dg))
        if len(dp): ip.append(triple_feats(a, dp))
        delta_update(m, np.concatenate(ig), np.concatenate(ip), max(loss, 1), C)
    return err / max(tot, 1)


def label_feats(A, tags, heads, pl=None):
    n = len(A['w'])
    W = ['<ROOT>'] + A['w']; T = ['<ROOT>'] + list(tags); C = ['<ROOT>'] + [t.split('/')[0] for t in tags]
    K = ['<ROOT>'] + [str(x) for x in A['k128']]; S3 = ['<ROOT>'] + A['s3']; SH = ['<ROOT>'] + A['shape']
    R = ['<ROOT>'] + A['raw']
    ch = [[] for _ in range(n + 1)]
    for d in range(1, n + 1): ch[heads[d]].append(d)
    def g(arr, i):
        if i < 0: return '<S>'
        if i > n: return '</S>'
        return arr[i]
    rows = []
    for d in range(1, n + 1):
        h = heads[d]
        dr = 'L' if h < d else 'R'
        dist = min(abs(h - d), 10)
        dkids = ch[d]; hkids = ch[h]
        side = [k for k in hkids if (k < h) == (d < h)]
        f = ['b', 'dir=' + dr, 'dist=%s%d' % (dr, dist),
             'dw=' + W[d], 'dt=' + T[d], 'dc=' + C[d], 'ds3=' + S3[d], 'dsh=' + SH[d], 'dk=' + K[d],
             'hw=' + W[h], 'ht=' + T[h], 'hc=' + C[h], 'hk=' + K[h], 'hs3=' + S3[h],
             'dwhw=%s|%s' % (W[d], W[h]), 'dthc=%s|%s|%s' % (T[d], C[h], dr), 'dwht=%s|%s' % (W[d], T[h]),
             'dthw=%s|%s' % (T[d], W[h]), 'dkhk=%s|%s|%s' % (K[d], K[h], dr), 'dchc=%s|%s|%s' % (C[d], C[h], dr),
             'dtdir=%s|%s' % (T[d], dr), 'dwdir=%s|%s' % (W[d], dr),
             'dw-1=' + g(W, d - 1), 'dw+1=' + g(W, d + 1), 'dt-1=' + g(T, d - 1), 'dt+1=' + g(T, d + 1),
             'ht-1=' + g(T, h - 1), 'ht+1=' + g(T, h + 1),
             'hroot=%d' % (h == 0), 'hhroot=%d' % (h > 0 and heads[h] == 0),
             'nk=%d' % min(len(dkids), 4), 'first=%d' % (side and d == min(side, key=lambda k: abs(k - h))),
             'outer=%d' % (side and d == max(side, key=lambda k: abs(k - h))), 'nsib=%d' % min(len(side) - 1, 4),
             'dfirst=%d' % (d == 1), 'dlast=%d' % (d == n)]
        for k in dkids:
            if k < d and k >= d - 6:
                f.append('lkw=' + W[k]); f.append('lkc=' + C[k]); f.append('lkcdc=%s|%s' % (C[k], C[d]))
            elif k > d:
                f.append('rkc=' + C[k])
            if C[k] in ('case', 'mark', 'cop', 'aux', 'det', 'cc'):
                f.append('fn=%s|%s' % (C[k], W[k])); f.append('fn2=%s|%s|%s' % (C[k], W[k], C[h]))
        for k in hkids:
            if k != d:
                f.append('sib=%s|%s' % (C[k], 'L' if k < h else 'R'))
                f.append('sibd=%s|%s|%s' % (C[k], 'B' if k < d else 'A', C[d]))
        if h > 0:
            hh = heads[h]
            f.append('hht=%s|%s' % (g(T, hh), dr))
            for k in hkids:
                if C[k] in ('case', 'mark', 'cop', 'aux', 'aux:pass'):
                    f.append('hfn=%s|%s' % (C[k], W[k]))
        if pl is not None:
            P = ['<ROOT>'] + list(pl)
            f.append('pl=' + P[d]); f.append('hpl=' + P[h]); f.append('plhpl=%s|%s' % (P[d], P[h]))
            f.append('pldw=%s|%s' % (P[d], W[d]))
            for k in hkids:
                if k != d:
                    f.append('sibpl=%s|%s|%s' % (P[k], 'B' if k < d else 'A', 'L' if k < h else 'R'))
            for k in dkids:
                f.append('kidpl=%s|%s' % (P[k], 'L' if k < d else 'R'))
        rows.append(f)
    return rows


class Labeler:
    def __init__(self, C=0.05, nbits=20):
        self.C = C; self.nbits = nbits

    def fit(self, attrs, tags, heads, rels, pls=None):
        from sklearn.svm import LinearSVC
        pls = pls or [None] * len(attrs)
        rows = [r for A, t, h, p in zip(attrs, tags, heads, pls) for r in label_feats(A, t, h, p)]
        y = np.array([x for rr in rels for x in rr])
        X = hash_rows(rows, self.nbits)
        self.m = LinearSVC(C=self.C, max_iter=3000).fit(X, y)
        return self

    def predict(self, attrs, tags, heads, pls=None):
        pls = pls or [None] * len(attrs)
        rows = [r for A, t, h, p in zip(attrs, tags, heads, pls) for r in label_feats(A, t, h, p)]
        X = hash_rows(rows, self.nbits)
        sc = self.m.decision_function(X)
        cls = self.m.classes_
        ri = list(cls).index('root')
        hflat = np.concatenate([h[1:] for h in heads])
        sc[hflat == 0, :] = -1e9; sc[hflat == 0, ri] = 0
        sc[hflat != 0, ri] = -1e9
        lab = cls[sc.argmax(1)]
        out, p = [], 0
        for h in heads:
            n = len(h) - 1; out.append(list(lab[p:p + n])); p += n
        return out


def jackknife_labels(attrs_tr, tags_tr, heads_tr, rels_tr, attrs_ot, tags_ot, heads_ot, K=5, seed=0, C=0.05):
    """Two-stage stacked labeler. Returns predicted labels for 'other' sentences."""
    fold = np.random.RandomState(seed).randint(0, K, len(attrs_tr))
    sel = lambda L, idx: [L[i] for i in idx]
    pl_tr = [None] * len(attrs_tr)
    for k in range(K):
        a = np.where(fold != k)[0]; b = np.where(fold == k)[0]
        lab = Labeler(C).fit(sel(attrs_tr, a), sel(tags_tr, a), sel(heads_tr, a), sel(rels_tr, a))
        for i, p in zip(b, lab.predict(sel(attrs_tr, b), sel(tags_tr, b), sel(heads_tr, b))): pl_tr[i] = p
    lab1 = Labeler(C).fit(attrs_tr, tags_tr, heads_tr, rels_tr)
    pl_ot = lab1.predict(attrs_ot, tags_ot, heads_ot)
    lab2 = Labeler(C).fit(attrs_tr, tags_tr, heads_tr, rels_tr, pl_tr)
    return lab2.predict(attrs_ot, tags_ot, heads_ot, pl_ot), pl_ot


"""Globally trained beam-search arc-hybrid parser (Zhang & Nivre 2011 style rich non-local features),
structured perceptron with max-violation updates (Huang et al. 2012), plus stacked guide features
from other parsers (Nivre & McDonald 2008). Non-neural: linear model over hashed indicator features."""

BNB = 22
BMASK = (1 << BNB) - 1
SH, LA, RA = 0, 1, 2


class St:
    __slots__ = ('stack', 'buf', 'head', 'lc1', 'lc2', 'rc1', 'rc2', 'nl', 'nr', 'score', 'parent', 'act', 'idx', 'gold')

    def copy(self):
        s = St.__new__(St)
        s.stack = self.stack; s.buf = self.buf
        s.head = self.head[:]; s.lc1 = self.lc1[:]; s.lc2 = self.lc2[:]
        s.rc1 = self.rc1[:]; s.rc2 = self.rc2[:]; s.nl = self.nl[:]; s.nr = self.nr[:]
        return s


def init_state(n):
    s = St.__new__(St)
    s.stack = (0,); s.buf = 1
    s.head = [-1] * (n + 1); s.lc1 = [-1] * (n + 1); s.lc2 = [-1] * (n + 1)
    s.rc1 = [-1] * (n + 1); s.rc2 = [-1] * (n + 1); s.nl = [0] * (n + 1); s.nr = [0] * (n + 1)
    s.score = 0.0; s.parent = None; s.act = -1; s.idx = None; s.gold = True
    return s


def legal(s, n):
    st = s.stack
    L0 = s.buf <= n
    L1 = len(st) >= 1 and st[-1] != 0 and s.buf <= n
    L2 = len(st) >= 2 and (st[-2] != 0 or (s.buf > n and len(st) == 2))
    return (L0, L1, L2)


def apply(s, a, n):
    t = s.copy()
    if a == SH:
        t.stack = s.stack + (s.buf,); t.buf = s.buf + 1
    elif a == LA:
        d = s.stack[-1]; h = s.buf
        t.stack = s.stack[:-1]; t.head[d] = h
        t.lc2[h] = t.lc1[h]; t.lc1[h] = d; t.nl[h] += 1
    else:
        d = s.stack[-1]; h = s.stack[-2]
        t.stack = s.stack[:-1]; t.head[d] = h
        t.rc2[h] = t.rc1[h]; t.rc1[h] = d; t.nr[h] += 1
    return t


def static_oracle(gold, n):
    """Canonical arc-hybrid action sequence for a projective tree, or None if unreachable."""
    nkids = [0] * (n + 1)
    for d in range(1, n + 1): nkids[gold[d]] += 1
    got = [0] * (n + 1)
    s = init_state(n); acts = []
    for _ in range(2 * n):
        L = legal(s, n); st = s.stack
        if L[1] and gold[st[-1]] == s.buf and got[st[-1]] == nkids[st[-1]]:
            a = LA
        elif L[2] and gold[st[-1]] == st[-2] and got[st[-1]] == nkids[st[-1]]:
            a = RA
        elif L[0]:
            a = SH
        else:
            return None
        if a != SH:
            got[gold[st[-1]]] += 1
        acts.append(a); s = apply(s, a, n)
    if any(s.head[d] != gold[d] for d in range(1, n + 1)):
        return None
    return acts


def feats(arr, s, n):
    P = arr.get('_py')
    if P is None:
        P = arr['_py'] = {key: [int(x) for x in arr[key]] for key in ('w', 't', 'c', 'k128', 'k512')}
    w = P['w']; t = P['t']; c = P['c']; k = P['k128']; k5 = P['k512']
    st = s.stack; b = s.buf
    s0 = st[-1] if st else -1
    s1 = st[-2] if len(st) > 1 else -1
    s2 = st[-3] if len(st) > 2 else -1
    b0 = b if b <= n else -1
    b1 = b + 1 if b + 1 <= n else -1
    b2 = b + 2 if b + 2 <= n else -1
    NA = -7
    def W(i): return w[i] if i >= 0 else NA
    def T(i): return t[i] if i >= 0 else NA
    def C(i): return c[i] if i >= 0 else NA
    def K(i): return k[i] if i >= 0 else NA
    lc1, lc2, rc1, rc2 = s.lc1, s.lc2, s.rc1, s.rc2
    s0l = lc1[s0] if s0 >= 0 else -1; s0l2 = lc2[s0] if s0 >= 0 else -1
    s0r = rc1[s0] if s0 >= 0 else -1; s0r2 = rc2[s0] if s0 >= 0 else -1
    s1l = lc1[s1] if s1 >= 0 else -1; s1r = rc1[s1] if s1 >= 0 else -1; s1r2 = rc2[s1] if s1 >= 0 else -1
    b0l = lc1[b0] if b0 >= 0 else -1; b0l2 = lc2[b0] if b0 >= 0 else -1
    s0w, s0t, s1w, s1t, b0w, b0t = W(s0), T(s0), W(s1), T(s1), W(b0), T(b0)
    b1w, b1t, b2t, s2t = W(b1), T(b1), T(b2), T(s2)
    s0c, s1c, b0c = C(s0), C(s1), C(b0)
    s0k, s1k, b0k, b1k = K(s0), K(s1), K(b0), K(b1)
    d0 = min(b0 - s0, 9) if (b0 >= 0 and s0 >= 0) else -1
    d1 = min(s0 - s1, 9) if (s1 >= 0 and s0 >= 0) else -1
    vl0 = s.nl[s0] if s0 >= 0 else -1; vr0 = s.nr[s0] if s0 >= 0 else -1
    vl1 = s.nl[s1] if s1 >= 0 else -1; vr1 = s.nr[s1] if s1 >= 0 else -1
    vlb = s.nl[b0] if b0 >= 0 else -1
    s0lc, s0l2c, s0rc, s0r2c = C(s0l), C(s0l2), C(s0r), C(s0r2)
    s1lc, s1rc, s1r2c, b0lc, b0l2c = C(s1l), C(s1r), C(s1r2), C(b0l), C(b0l2)
    f = [
        # unigrams
        (1, s0w), (2, s0t), (3, s0w, s0t), (4, s1w), (5, s1t), (6, s1w, s1t), (7, b0w), (8, b0t), (9, b0w, b0t),
        (10, b1w), (11, b1t), (12, b1w, b1t), (13, b2t), (14, s2t), (15, s0k), (16, b0k), (17, s1k), (18, b1k),
        (19, k5[s0] if s0 >= 0 else NA), (19, k5[b0] if b0 >= 0 else NA, 1),
        # pairs
        (20, s0w, s0t, b0w, b0t), (21, s0w, s0t, b0w), (22, s0w, b0w, b0t), (23, s0w, s0t, b0t), (24, s0t, b0w, b0t),
        (25, s0w, b0w), (26, s0t, b0t), (27, b0t, b1t), (28, s1w, s0w), (29, s1t, s0t), (30, s1w, s0t), (31, s1t, s0w),
        (32, s1w, s1t, s0w, s0t), (33, s1t, s0t, s0w), (34, s1w, s1t, s0t),
        (35, s0k, b0k), (36, s1k, s0k), (37, s0k, b0t), (38, s0t, b0k), (39, s0k, b0k, b1k), (40, s1k, s0k, b0k),
        # trigrams
        (41, b0t, b1t, b2t), (42, s0t, b0t, b1t), (43, s1t, s0t, b0t), (44, s2t, s1t, s0t), (45, s1t, s0t, b1t),
        (46, s0w, b0t, b1t), (47, s0t, b0w, b1t), (48, s1t, s0w, b0t),
        # distance
        (50, s0w, d0), (51, s0t, d0), (52, b0w, d0), (53, b0t, d0), (54, s0w, b0w, d0), (55, s0t, b0t, d0),
        (56, s1t, s0t, d1), (57, s1w, s0w, d1), (58, s1t, s0t, d1, b0t),
        # valency
        (60, s0w, vl0, vr0), (61, s0t, vl0, vr0), (62, b0w, vlb), (63, b0t, vlb), (64, s1t, s0t, vl0, vr0),
        (65, s1t, vl1, vr1), (66, s1w, vl1, vr1), (67, s1t, vr1, s0t),
        # children unigrams / third order
        (70, s0t, s0lc), (71, s0t, s0rc), (72, s0t, s0lc, s0l2c), (73, s0t, s0rc, s0r2c), (74, s1t, s1lc), (75, s1t, s1rc),
        (76, s1t, s1rc, s1r2c), (77, b0t, b0lc), (78, b0t, b0lc, b0l2c), (79, s0t, s0lc, b0t), (80, s0t, s0rc, b0t),
        (81, s1t, s1rc, s0t), (82, s0w, s0rc, b0t), (83, b0t, b0lc, s0t), (84, s1t, s0t, s0rc), (85, s1t, s0t, s0lc),
        (86, s0w, s0lc), (87, s0w, s0rc), (88, b0w, b0lc), (89, W(s0r)), (90, W(s0l)), (91, W(b0l)), (92, W(s1r)),
        (93, s0w, W(s0r)), (94, b0w, W(b0l)),
        # misc
        (95, len(st) > 2, b0 < 0, s1 == 0), (96, s1 == 0, s0t), (97, b0 < 0, s0t, s1t),
    ]
    # stacked guide features
    G = arr.get('_bguides')
    if G is not None:
        for gi, gh in enumerate(G):
            if s0 >= 1:
                hs0 = gh[s0]
                code = (1 if hs0 == b0 else 0) + (2 if hs0 == s1 else 0) + (4 if (b0 >= 1 and gh[b0] == s0) else 0) \
                    + (8 if (s1 >= 1 and gh[s1] == s0) else 0) + (16 if hs0 > b and b0 >= 0 else 0)
                # guide children of s0 still in buffer
                pend = arr['_gkids'][gi][s0]; pb = 0
                for x in pend:
                    if x >= b: pb += 1
                pb = min(pb, 3)
            else:
                code = 32 if b0 >= 0 else 64; pb = 0
            if b0 >= 1:
                pk = 0
                for x in arr['_gkids'][gi][b0]:
                    if x < b0 and s.head[x] == -1 and x not in st: pk += 1  # unreachable kids
                pk = min(pk, 3)
            else:
                pk = -1
            base = 200 + 20 * gi
            f += [(base, code), (base + 1, code, pb), (base + 2, code, s0t, b0t), (base + 3, code, s0c, s1c),
                  (base + 4, code, pb, pk), (base + 5, code, s0w), (base + 6, code, b0w), (base + 7, code, d0),
                  (base + 8, code, s0c, b0c, d0), (base + 9, pk, b0t)]
        if s0 >= 1:
            agree = sum(1 for gh in G if gh[s0] == b0) * 10 + sum(1 for gh in G if gh[s0] == s1)
        else:
            agree = -1
        f += [(300, agree), (301, agree, s0c), (302, agree, s0c, b0c, s1c)]
    return np.fromiter((hash(x) & BMASK for x in f), dtype=np.int64, count=len(f))


def prep(arr, guides=None):
    """Attach guide trees (list of head arrays) + their child lists to a sentence's attribute dict."""
    if guides:
        arr['_bguides'] = [[int(x) for x in g] for g in guides]
        n = arr['_n']; kids = []
        for g in guides:
            ch = [[] for _ in range(n + 1)]
            for d in range(1, n + 1): ch[g[d]].append(d)
            kids.append(ch)
        arr['_gkids'] = kids
    return arr


class BeamParser:
    def __init__(self, beam=8):
        self.W = np.zeros((1 << BNB, 3), np.float32)
        self.U = np.zeros((1 << BNB, 3), np.float32)
        self.c = 1
        self.beam = beam
        self.Wavg = None

    def _expand(self, beam, arr, n, W, gold_act=None):
        idxs = [feats(arr, s, n) for s in beam]
        I = np.stack(idxs)
        sc = W[I].sum(1)  # (B, 3)
        cands = []
        for bi, s in enumerate(beam):
            L = legal(s, n)
            for a in (0, 1, 2):
                if L[a]:
                    cands.append((s.score + float(sc[bi, a]), bi, a))
        cands.sort(key=lambda x: -x[0])
        out = []
        for score, bi, a in cands[:self.beam]:
            p = beam[bi]; t = apply(p, a, n)
            t.score = score; t.parent = p; t.act = a; t.idx = idxs[bi]
            t.gold = p.gold and gold_act is not None and a == gold_act
            out.append(t)
        return out

    def parse(self, arr, W=None):
        W = self.Wavg if W is None else W
        n = arr['_n']
        beam = [init_state(n)]
        for _ in range(2 * n):
            beam = self._expand(beam, arr, n, W)
        best = beam[0]
        h = np.array(best.head, dtype=np.int64); h[0] = 0
        return h

    def train_one(self, arr, gold_acts, gold_idx):
        n = arr['_n']; W = self.W
        gsc = np.cumsum(W[gold_idx, np.array(gold_acts)[:, None]].sum(1))  # gold prefix scores
        beam = [init_state(n)]
        best_v, best_t, best_item = 0.0, -1, None
        for t in range(2 * n):
            beam = self._expand(beam, arr, n, W, gold_acts[t])
            top = beam[0]
            v = top.score - gsc[t]
            if not top.gold and v >= best_v:
                best_v, best_t, best_item = v, t, top
        if best_item is None:
            self.c += 1; return 0
        # collect predicted prefix features
        p_idx, p_act = [], []
        x = best_item
        while x.parent is not None:
            p_idx.append(x.idx); p_act.append(x.act); x = x.parent
        p_idx = np.stack(p_idx[::-1]); p_act = np.array(p_act[::-1])
        g_idx = gold_idx[:best_t + 1]; g_act = np.array(gold_acts[:best_t + 1])
        rows = np.concatenate([g_idx.ravel(), p_idx.ravel()])
        cols = np.concatenate([np.repeat(g_act, g_idx.shape[1]), np.repeat(p_act, p_idx.shape[1])])
        vals = np.concatenate([np.ones(g_idx.size, np.float32), -np.ones(p_idx.size, np.float32)])
        np.add.at(self.W, (rows, cols), vals)
        np.add.at(self.U, (rows, cols), vals * self.c)
        self.c += 1
        return 1

    def averaged(self):
        return self.W - self.U / self.c


def gold_path(arr, gold):
    n = arr['_n']
    acts = static_oracle(gold, n)
    if acts is None:
        return None
    s = init_state(n); idx = []
    for a in acts:
        idx.append(feats(arr, s, n)); s = apply(s, a, n)
    return acts, np.stack(idx)


def train_beam(arrs, golds, epochs=8, beam=8, seed=0, log=None, dev=None):
    p = BeamParser(beam)
    data = []
    for a, g in zip(arrs, golds):
        gp = gold_path(a, g)
        if gp is not None:
            data.append((a, gp[0], gp[1]))
    rng = np.random.RandomState(seed)
    for ep in range(epochs):
        upd = 0
        for i in rng.permutation(len(data)):
            a, acts, gi = data[i]
            upd += p.train_one(a, acts, gi)
        p.Wavg = p.averaged()
        if log: log(ep, upd / len(data), p)
    p.Wavg = p.averaged()
    return p


def reverse_beam_arr(arr):
    """Right-to-left copy of an attribute dict (ROOT stays at 0); guides are remapped too."""
    n = arr['_n']
    perm = np.concatenate([[0], np.arange(n, 0, -1)])
    out = {}
    for k, v in arr.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n + 1:
            out[k] = v[perm]
        elif k in ('_n',):
            out[k] = v
    if '_bguides' in arr:
        inv = np.argsort(perm)
        prep(out, [inv[np.asarray(g)[perm]] for g in arr['_bguides']])
    return out, perm


def unreverse(h, perm):
    n = len(h) - 1; out = np.zeros(n + 1, np.int64)
    for j in range(1, n + 1): out[perm[j]] = perm[h[j]]
    return out


def label_feats_x(A, tags, heads):
    rows = label_feats(A, tags, heads)
    n = len(A['w'])
    NW = ['<ROOT>'] + A['nw']; RAW = ['<ROOT>'] + A['raw']; C = ['<ROOT>'] + [t.split('/')[0] for t in tags]
    T = ['<ROOT>'] + list(tags); K5 = ['<ROOT>'] + [str(x) for x in A['k512']]; K3 = ['<ROOT>'] + [str(x) for x in A['k32']]
    ch = [[] for _ in range(n + 1)]
    for d in range(1, n + 1): ch[heads[d]].append(d)
    for d in range(1, n + 1):
        h = heads[d]; f = rows[d - 1]; dr = 'L' if h < d else 'R'
        nw = NW[d]; hw = NW[h]
        f += ['xnw=' + nw, 'xhnw=' + hw, 'xnwhnw=%s|%s' % (nw, hw), 'xk5=' + K5[d], 'xhk5=' + K5[h],
              'xk5hk5=%s|%s|%s' % (K5[d], K5[h], dr), 'xs2=' + nw[-2:], 'xs4=' + nw[-4:], 'xp3=' + nw[:3],
              'xhs2=' + hw[-2:], 'xhs4=' + hw[-4:], 'xcap=%d%d' % (RAW[d][:1].isupper(), d == 1),
              'xhcap=%d' % (RAW[h][:1].isupper() if h else 9), 'xnwdt=%s|%s' % (nw, T[h]),
              'xs3hc=%s|%s|%s' % (nw[-3:], C[h], dr), 'xk3hk3dist=%s|%s|%s%d' % (K3[d], K3[h], dr, min(abs(h - d), 5)),
              'xdtht=%s|%s' % (T[d], T[h]), 'xdthtdist=%s|%s|%d' % (T[d], T[h], min(abs(h - d), 5))]
        kids = ch[d]
        if kids:
            f.append('xrk=' + NW[max(kids)]); f.append('xlk=' + NW[min(kids)])
            f.append('xkcs=' + '|'.join(sorted(set(C[k] for k in kids))))
        hk = [k for k in ch[h] if k != d]
        f.append('xhkcs=%s|%s' % ('|'.join(sorted(set(C[k] for k in hk))), dr))
        pas = any(NW[k] in ('was', 'were', 'been', 'being', 'is', 'are', 'be', "'s", 'get', 'got', 'gets') for k in hk if k < h)
        f.append('xpass=%d|%s|%s' % (pas, hw[-2:], C[d]))
        by = any(NW[k] == 'by' for k in kids)
        f.append('xby=%d|%s' % (by, C[h]))
        if h > 0 and h + 1 <= n: f.append('xhnext=' + NW[h + 1])
        if d + 1 <= n: f.append('xdnextc=%s|%s' % (NW[d + 1], C[d]))
    return rows

class LabelerX(Labeler):
    def __init__(self, C=0.05, nbits=21, fx=label_feats_x):
        super().__init__(C, nbits); self.fx = fx
    def fit(self, attrs, tags, heads, rels):
        rows = [r for A, t, h in zip(attrs, tags, heads) for r in self.fx(A, t, h)]
        y = np.array([x for rr in rels for x in rr])
        self.m = LinearSVC(C=self.C, max_iter=3000).fit(hash_rows(rows, self.nbits), y); return self
    def predict(self, attrs, tags, heads):
        rows = [r for A, t, h in zip(attrs, tags, heads) for r in self.fx(A, t, h)]
        sc = self.m.decision_function(hash_rows(rows, self.nbits)); cls = self.m.classes_; ri = list(cls).index('root')
        hf = np.concatenate([h[1:] for h in heads]); sc[hf == 0, :] = -1e9; sc[hf == 0, ri] = 0; sc[hf != 0, ri] = -1e9
        lab = cls[sc.argmax(1)]; out, p = [], 0
        for h in heads:
            n = len(h) - 1; out.append(list(lab[p:p + n])); p += n
        return out


T0 = time.time()
def log(*a):
    print('[%6.0fs]' % (time.time() - T0), *a, flush=True)


def cached(path, fn):
    if path and os.path.exists(path):
        log('  loaded cache', os.path.basename(path)); return pickle.load(open(path, 'rb'))
    out = fn()
    if path:
        pickle.dump(out, open(path, 'wb'))
    return out


def split_flat(flat, lens):
    out, p = [], 0
    for n in lens:
        out.append(list(flat[p:p + n])); p += n
    return out


def tag_stage(rows_tr, rows_ot, y, fold, K, n_jobs, C=0.05, nbits=20):
    Xtr = hash_rows(rows_tr, nbits); Xot = hash_rows(rows_ot, nbits)
    jobs = [(Xtr[fold != k], y[fold != k], Xtr[fold == k]) for k in range(K)] + [(Xtr, y, Xot)]
    res = Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(a, b, c, C) for a, b, c in jobs)
    ptr = np.empty(len(y), dtype=object)
    for k in range(K):
        ptr[fold == k] = res[k]
    return ptr, res[K]


def _trans_job(arrs_fit, golds_fit, arrs_pred, rev, epochs, seed):
    p = train_trans(arrs_fit, golds_fit, epochs=epochs, reverse=rev, seed=seed)
    return [trans_predict(p, a, reverse=rev) for a in arrs_pred]


def trans_round(arr_tr, gold_tr, arr_ot, K, n_jobs, ep_jk=10, ep_full=15):
    """Jackknifed (train) + full-model (other) predictions of L2R and R2L arc-hybrid parsers."""
    fold = np.random.RandomState(1).randint(0, K, len(arr_tr))
    jobs = []
    for rev in (False, True):
        for k in range(K):
            a = np.where(fold != k)[0]; b = np.where(fold == k)[0]
            jobs.append(([arr_tr[i] for i in a], [gold_tr[i] for i in a], [arr_tr[i] for i in b], rev, ep_jk, k))
        jobs.append((arr_tr, gold_tr, arr_ot, rev, ep_full, 0))
    res = Parallel(n_jobs=n_jobs)(delayed(_trans_job)(*j) for j in jobs)
    out = {}
    for r, rev in enumerate((False, True)):
        base = r * (K + 1)
        jk = [None] * len(arr_tr)
        for k in range(K):
            for i, h in zip(np.where(fold == k)[0], res[base + k]): jk[i] = h
        out[rev] = (jk, res[base + K])
    return out[False][0], out[True][0], out[False][1], out[True][1]


def train_graph(fz, arrs, golds, epochs, order):
    m = ArcModel()
    for ep in range(epochs):
        if order == 2:
            train_epoch(m, fz, arrs, golds, ep)
        else:
            for i in np.random.RandomState(ep).permutation(len(arrs)):
                F = fz.feats(arrs[i]); S = m.score(F); g = golds[i]
                S = S + 1.0; S[g[1:], np.arange(1, len(g))] -= 1.0
                m.update(F, g, cle(S), C=0.05)
        log('    graph order %d epoch %d done' % (order, ep))
    return m.averaged()


ENS_W = {'G1': 1.5, 'G2': 2, 'Gb': 1.5, 'bL': 1, 'bR': 1, 'bG4': 2.5, 'bGG4': 2, 'g2': 1}   # validated on held-out data


def _l1_job(fz, arrs, golds, arrs_pred, order, epochs):
    W = train_graph(fz, arrs, golds, epochs, order)
    return [parse2(fz, a, W)[0] if order == 2 else cle(W[fz.feats(a)].sum(0)) for a in arrs_pred]


def _beam_job(arrs, golds, pred, rev, epochs, beam, seed):
    if rev:
        R = [reverse_beam_arr(a) for a in arrs]
        golds = [np.argsort(p)[g[p]] for (a, p), g in zip(R, golds)]; arrs = [a for a, p in R]
        RP = [reverse_beam_arr(a) for a in pred]
        p = train_beam(arrs, golds, epochs=epochs, beam=beam, seed=seed)
        return [unreverse(p.parse(a), pp) for a, pp in RP]
    p = train_beam(arrs, golds, epochs=epochs, beam=beam, seed=seed)
    return [p.parse(a) for a in pred]


def beam_round(bt, gold_t, bo, gbt, gbo, K, n_jobs, epochs, beam):
    fold = np.random.RandomState(3).randint(0, K, len(bt))
    jobs = []
    for rev in (False, True):
        for k in range(K):
            a = np.where(fold != k)[0]; b = np.where(fold == k)[0]
            jobs.append(([bt[i] for i in a], [gold_t[i] for i in a], [bt[i] for i in b], rev, epochs, beam, k))
        jobs.append((bt, gold_t, bo, rev, epochs, beam, 0))
    jobs.append((gbt, gold_t, gbo, False, epochs, beam, 0))   # guided L2R beam, test predictions only
    res = Parallel(n_jobs=n_jobs)(delayed(_beam_job)(*j) for j in jobs)
    out = []
    for r in range(2):
        base = r * (K + 1); jk = [None] * len(bt)
        for k in range(K):
            for i, h in zip(np.where(fold == k)[0], res[base + k]): jk[i] = h
        out.append((jk, res[base + K]))
    return out[0][0], out[1][0], out[0][1], out[1][1], res[-1]


def _graph_job(fz, arrs, golds, pred_sets, epochs):
    W = train_graph(fz, arrs, golds, epochs, 2)
    return [[parse2(fz, a, W)[0] for a in S] for S in pred_sets]


def graph_round(fz, arr_tr, gold_tr, arr_ot, K, n_jobs, epochs):
    """Jackknifed second-order graph parser: held-out predictions on train, full-model + fold-model predictions on 'other'."""
    fold = np.random.RandomState(2).randint(0, K, len(arr_tr))
    jobs = []
    for k in range(K):
        a = np.where(fold != k)[0]; b = np.where(fold == k)[0]
        jobs.append(([arr_tr[i] for i in a], [gold_tr[i] for i in a], [[arr_tr[i] for i in b], arr_ot]))
    jobs.append((arr_tr, gold_tr, [arr_ot]))
    res = Parallel(n_jobs=min(n_jobs, K + 1))(delayed(_graph_job)(fz, x, g, p, epochs) for x, g, p in jobs)
    jk = [None] * len(arr_tr)
    for k in range(K):
        for i, h in zip(np.where(fold == k)[0], res[k][0]): jk[i] = h
    return jk, res[K][0], [res[k][1] for k in range(K)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='.')
    ap.add_argument('--out', default='submission.csv')
    ap.add_argument('--dev', action='store_true', help='hold out 10%% of train and report UAS/LAS instead')
    ap.add_argument('--jobs', type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument('--tag_jobs', type=int, default=None, help='parallel LinearSVC fits (each needs ~2-3 GB RAM)')
    ap.add_argument('--beam', type=int, default=8)
    ap.add_argument('--beam_epochs', type=int, default=8)
    ap.add_argument('--beam_jobs', type=int, default=None, help='parallel beam-parser trainings (each ~1 GB RAM); default = --jobs, max 13')
    ap.add_argument('--no_copy_dups', action='store_true', help='do not copy analyses of test sentences that occur verbatim in train')
    ap.add_argument('--cache', default='cache')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--g2_epochs', type=int, default=5)
    ap.add_argument('--g1_epochs', type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    mode = 'dev' if args.dev else 'full'
    cp = lambda name: os.path.join(args.cache, '%s_%s.pkl' % (mode, name))
    K, NJ = args.folds, args.jobs
    TJ = args.tag_jobs or NJ
    BJ = min(13, args.beam_jobs or NJ)

    train = load(os.path.join(args.data, 'train.csv'), True)
    test = load(os.path.join(args.data, 'test.csv'), False)
    log('train sents %d, test sents %d' % (len(train), len(test)))
    clusters = cached(os.path.join(args.cache, 'clusters.pkl'), lambda: build_clusters(train + test))
    if args.dev:
        perm = np.random.RandomState(0).permutation(len(train))
        tst = [train[i] for i in perm[:1250]]; trn = [train[i] for i in perm[1250:]]
    else:
        trn, tst = train, test
    wc = Counter(norm(w) for s in trn for w in s['tok'])
    At = [token_attrs(s, wc, clusters) for s in trn]; Ao = [token_attrs(s, wc, clusters) for s in tst]
    len_t = [len(s['tok']) for s in trn]; len_o = [len(s['tok']) for s in tst]
    y = np.array([t for s in trn for t in supertag(s)])
    sid = np.concatenate([[k] * n for k, n in enumerate(len_t)])
    tfold = np.random.RandomState(0).randint(0, K, len(trn))[sid]
    gold_t = [np.concatenate([[0], s['head']]) for s in trn]
    fz = ArcFeaturizer(V)

    # 1. supertagger, stages 1+2 (jackknifed)
    def tags12():
        r1t = [tagger_feats(A, i) for A in At for i in range(len(A['w']))]
        r1o = [tagger_feats(A, i) for A in Ao for i in range(len(A['w']))]
        p_t, p_o = tag_stage(r1t, r1o, y, tfold, K, TJ)
        log('  tagger stage1 jackknife acc %.4f' % (p_t == y).mean())
        t1t, t1o = split_flat(p_t, len_t), split_flat(p_o, len_o)
        r2t = [f + tagger_feats_stage2(A, i, t) for A, t in zip(At, t1t) for i, f in enumerate([tagger_feats(A, j) for j in range(len(A['w']))])]
        r2o = [f + tagger_feats_stage2(A, i, t) for A, t in zip(Ao, t1o) for i, f in enumerate([tagger_feats(A, j) for j in range(len(A['w']))])]
        p_t, p_o = tag_stage(r2t, r2o, y, tfold, K, TJ)
        log('  tagger stage2 jackknife acc %.4f' % (p_t == y).mean())
        return split_flat(p_t, len_t), split_flat(p_o, len_o)
    log('1. supertagger stages 1-2')
    tags_t, tags_o = cached(cp('tags12'), tags12)

    # 2. transition parsers round A (guides for parse-informed tagger)
    log('2. transition parsers, round A')
    arr_t = [fz.sent_arrays(A, t) for A, t in zip(At, tags_t)]; arr_o = [fz.sent_arrays(A, t) for A, t in zip(Ao, tags_o)]
    jA0, jA1, oA0, oA1 = cached(cp('transA'), lambda: trans_round(arr_t, gold_t, arr_o, K, NJ))
    log('  round A jackknife UAS L2R %.4f R2L %.4f' % tuple(np.mean(np.concatenate([h[1:] == g[1:] for h, g in zip(j, gold_t)])) for j in (jA0, jA1)))

    # 3. supertagger stage 3 (parse-informed)
    def tags3():
        r3t = [tagger_feats(A, i) + tagger_feats_stage2(A, i, t) + tagger_feats_parse(A, i, t, [g0, g1])
               for A, t, g0, g1 in zip(At, tags_t, jA0, jA1) for i in range(len(A['w']))]
        r3o = [tagger_feats(A, i) + tagger_feats_stage2(A, i, t) + tagger_feats_parse(A, i, t, [g0, g1])
               for A, t, g0, g1 in zip(Ao, tags_o, oA0, oA1) for i in range(len(A['w']))]
        p_t, p_o = tag_stage(r3t, r3o, y, tfold, K, TJ)
        log('  tagger stage3 jackknife acc %.4f' % (p_t == y).mean())
        return split_flat(p_t, len_t), split_flat(p_o, len_o)
    log('3. supertagger stage 3 (parse-informed)')
    tags_t, tags_o = cached(cp('tags3'), tags3)
    yo = np.array([t for s in tst for t in supertag(s)]) if args.dev else None
    if args.dev:
        log('  DEV supertag acc %.4f' % (np.array([t for x in tags_o for t in x]) == yo).mean())

    # 4. transition parsers round B (final tags)
    log('4. transition parsers, round B')
    arr_t = [fz.sent_arrays(A, t) for A, t in zip(At, tags_t)]; arr_o = [fz.sent_arrays(A, t) for A, t in zip(Ao, tags_o)]
    jB0, jB1, oB0, oB1 = cached(cp('transB'), lambda: trans_round(arr_t, gold_t, arr_o, K, NJ))
    for a, g0, g1 in zip(arr_t, jB0, jB1): a['_guides'] = [g0, g1]
    for a, g0, g1 in zip(arr_o, oB0, oB1): a['_guides'] = [g0, g1]

    # 5. level-0 second-order graph parser (guided by round-B transition parsers), jackknifed
    log('5. level-0 graph parser (jackknifed, %d folds)' % K)
    for a, g0, g1 in zip(arr_t, jB0, jB1): a['_guides'] = [g0, g1]
    for a, g0, g1 in zip(arr_o, oB0, oB1): a['_guides'] = [g0, g1]
    jkG, oG, oG_bag = cached(cp('graphL0'), lambda: graph_round(fz, arr_t, gold_t, arr_o, K, NJ, args.g2_epochs))
    log('  level-0 graph jackknife UAS %.4f' % np.mean(np.concatenate([h[1:] == g[1:] for h, g in zip(jkG, gold_t)])))

    # 6. globally trained beam-search parsers on stage-3 tags: L2R + R2L jackknifed (for stacking)
    log('6. beam-search parsers (%d parallel jobs)' % BJ)
    def plain(a):
        return {k: v for k, v in a.items() if k not in ('_guides', '_l0', '_bguides', '_gkids', '_py')}
    bt = [plain(a) for a in arr_t]; bo = [plain(a) for a in arr_o]
    gbt = [prep(plain(a), [x, y, z]) for a, x, y, z in zip(arr_t, jB0, jB1, jkG)]
    gbo = [prep(plain(a), [x, y, z]) for a, x, y, z in zip(arr_o, oB0, oB1, oG)]
    jbL, jbR, obL, obR, _ = cached(cp('beam'), lambda: beam_round(bt, gold_t, bo, gbt, gbo, K, BJ, args.beam_epochs, args.beam))
    log('  beam jackknife UAS L2R %.4f R2L %.4f' % tuple(np.mean(np.concatenate([h[1:] == g[1:] for h, g in zip(j, gold_t)])) for j in (jbL, jbR)))

    # 7. supertagger stage 4: message passing from the level-0 graph tree + transition trees
    def tags4():
        r4t = [tagger_feats(A, i) + tagger_feats_stage2(A, i, t) + tagger_feats_parse(A, i, t, [g0, g1, g2])
               for A, t, g0, g1, g2 in zip(At, tags_t, jB0, jB1, jkG) for i in range(len(A['w']))]
        r4o = [tagger_feats(A, i) + tagger_feats_stage2(A, i, t) + tagger_feats_parse(A, i, t, [g0, g1, g2])
               for A, t, g0, g1, g2 in zip(Ao, tags_o, oB0, oB1, oG) for i in range(len(A['w']))]
        p_t, p_o = tag_stage(r4t, r4o, y, tfold, K, TJ)
        log('  tagger stage4 jackknife acc %.4f' % (p_t == y).mean())
        return split_flat(p_t, len_t), split_flat(p_o, len_o)
    log('7. supertagger stage 4 (graph-informed)')
    tags4_t, tags4_o = cached(cp('tags4'), tags4)
    if args.dev:
        log('  DEV supertag acc %.4f' % (np.array([t for x in tags4_o for t in x]) == yo).mean())
    a4t = [fz.sent_arrays(A, t) for A, t in zip(At, tags4_t)]; a4o = [fz.sent_arrays(A, t) for A, t in zip(Ao, tags4_o)]

    # 8. guided beam parsers on stage-4 tags (test predictions only)
    log('8. guided beam parsers on stage-4 tags')
    def beam4():
        jobs = [([prep(plain(a), [x, y_, z]) for a, x, y_, z in zip(a4t, jB0, jB1, jkG)], gold_t,
                 [prep(plain(a), [x, y_, z]) for a, x, y_, z in zip(a4o, oB0, oB1, oG)], False, args.beam_epochs, args.beam, 0),
                ([prep(plain(a), [x, y_, z, w]) for a, x, y_, z, w in zip(a4t, jkG, jbL, jbR, jB0)], gold_t,
                 [prep(plain(a), [x, y_, z, w]) for a, x, y_, z, w in zip(a4o, oG, obL, obR, oB0)], False, args.beam_epochs, args.beam, 0)]
        return Parallel(n_jobs=min(2, BJ))(delayed(_beam_job)(*j) for j in jobs)
    obG4, obGG4 = cached(cp('beam4'), beam4)

    # 9. level-1 graph parsers on stage-4 tags, guided by beam (+ transition) parsers
    log('9. level-1 graph parsers')
    def with_guides(arrs, guides):
        out = []
        for i, a in enumerate(arrs):
            x = plain(a); x['_guides'] = [g[i] for g in guides]; out.append(x)
        return out
    cfgs = [((jbL, jbR, jB0, jB1), (obL, obR, oB0, oB1), 2, args.g2_epochs),   # G2
            ((jbL, jbR, jB0, jB1), (obL, obR, oB0, oB1), 1, args.g1_epochs),   # G1
            ((jbL, jbR), (obL, obR), 2, args.g2_epochs)]                       # Gb
    def level1():
        jobs = [(with_guides(a4t, gt_), gold_t, with_guides(a4o, go_), o, e) for gt_, go_, o, e in cfgs]
        return Parallel(n_jobs=min(3, NJ))(delayed(_l1_job)(fz, *j) for j in jobs)
    oG2, oG1, oGb = cached(cp('graphL1t4'), level1)

    # 10. ensemble re-parsing (weighted arc votes + MST)
    log('10. ensemble')
    P = {'G1': oG1, 'G2': oG2, 'Gb': oGb, 'bL': obL, 'bR': obR, 'bG4': obG4, 'bGG4': obGG4, 'g2': oG}
    def vote(w):
        H = []
        for i in range(len(tst)):
            n = len_o[i]; S = np.zeros((n + 1, n + 1)); ar = np.arange(1, n + 1)
            for k_, x in w.items(): S[P[k_][i][1:], ar] += x
            S += 1e-3 * np.random.RandomState(i).rand(n + 1, n + 1)
            H.append(cle(S))
        return H
    heads = vote(ENS_W)

    # 11. labels: extended features, stage-4 tags, trained on gold trees + correctly attached
    #          tokens in jackknifed predicted trees (matches test-time context noise)
    log('11. labeler')
    rel_t = [s['rel'] for s in trn]
    rows = [r for A, t, h in zip(At, tags4_t, gold_t) for r in label_feats_x(A, t, h)]
    yl = [x for rr in rel_t for x in rr]
    for A, t, p, g, r in zip(At, tags4_t, jkG, gold_t, rel_t):
        fr = label_feats_x(A, t, p)
        for k_ in range(len(r)):
            if p[k_ + 1] == g[k_ + 1]: rows.append(fr[k_]); yl.append(r[k_])
    lab = LabelerX(0.05)
    lab.m = LinearSVC(C=0.05, max_iter=3000).fit(hash_rows(rows, 21), np.array(yl)); del rows
    rels = lab.predict(Ao, tags4_o, heads)

    # 12. exact duplicates of training sentences: copy their gold analysis
    seen = {}
    for s in trn:
        seen.setdefault(tuple(s['tok']), Counter())[(tuple(s['head']), tuple(s['rel']))] += 1
    model_heads, model_rels = list(heads), list(rels)
    if not args.no_copy_dups:
        nd_ = 0
        for i, s in enumerate(tst):
            c = seen.get(tuple(s['tok']))
            if c:
                (h, r), _ = c.most_common(1)[0]
                heads[i] = np.concatenate([[0], h]); rels[i] = list(r); nd_ += 1
        log('  copied %d exact-duplicate sentences' % nd_)

    if args.dev:
        nd = np.array([tuple(s['tok']) not in seen for s in tst])
        def ev(H, R=None, mask=None):
            mask = np.ones(len(tst), bool) if mask is None else mask
            u = np.concatenate([h[1:] == s['head'] for h, s, m in zip(H, tst, mask) if m])
            if R is None: return u.mean()
            l = np.concatenate([(h[1:] == s['head']) & (np.array(r) == np.array(s['rel'])) for h, r, s, m in zip(H, R, tst, mask) if m])
            return u.mean(), l.mean()
        for nm, p in P.items():
            log('  DEV UAS %-3s %.4f   (unseen sentences only %.4f)' % (nm, ev(p), ev(p, mask=nd)))
        log('  DEV ENSEMBLE  UAS %.4f  LAS %.4f   (all held-out sentences, duplicates copied)' % ev(heads, rels))
        log('  DEV ENSEMBLE  UAS %.4f  LAS %.4f   (unseen sentences only = honest estimate)' % ev(model_heads, model_rels, nd))
        return

    # 13. submission: every ID of the sample file; train IDs get their given labels
    pred = {}
    for s, h, r in zip(tst, heads, rels):
        for j, tid in enumerate(s['ids']):
            pred[int(tid)] = '%d|%s' % (h[j + 1], r[j])
    trdf = pd.read_csv(os.path.join(args.data, 'train.csv'), keep_default_na=False)
    for tid, t in zip(trdf.ID.values, trdf.target.values):
        pred.setdefault(int(tid), t)
    sample_path = None
    for nm in ('sample_submission.csv', 'sample.csv'):
        if os.path.exists(os.path.join(args.data, nm)): sample_path = os.path.join(args.data, nm)
    if sample_path:
        smp = pd.read_csv(sample_path)
        idcol, tcol = smp.columns[0], smp.columns[1]
        ids = smp[idcol].values
    else:
        idcol, tcol = 'ID', 'target'
        ids = sorted(pred)
    sub = pd.DataFrame({idcol: ids, tcol: [pred.get(int(i), '0|root') for i in ids]})
    sub.to_csv(args.out, index=False)
    log('wrote %s (%d rows, columns %s)' % (args.out, len(sub), list(sub.columns)))


if __name__ == '__main__':
    main()