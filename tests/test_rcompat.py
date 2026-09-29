import numpy as np

from datanuggets._rcompat import split_indices
from rfixtures import load_group


def test_split_indices_matches_r():
    (case,) = load_group("rcompat_split_indices")
    nx, ncl, nchunks = case.array("nx"), case.array("ncl"), case.array("nchunks")
    lengths = case.array("chunk_len")
    pos = 0
    for n, c, k in zip(nx.tolist(), ncl.tolist(), nchunks.tolist(), strict=True):
        chunks = split_indices(n, c)
        expected = lengths[pos : pos + k].tolist()
        pos += k
        assert [ch.size for ch in chunks] == expected, (n, c)
        np.testing.assert_array_equal(np.concatenate(chunks), np.arange(n))
    assert pos == lengths.size
