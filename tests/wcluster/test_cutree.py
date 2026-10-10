"""cut_tree against R's cutree: every k and h R cut, every stop, and the numpy loop against a
statement-for-statement copy of R's C routine on fuzzed trees."""

import numpy as np
import pytest

from datanuggets import WhclustResult, cut_tree
from rfixtures import load_group

CASES = load_group("plot")
TREES = [c for c in CASES if c.id.startswith("tree_") and c.id != "tree_hang"]
STOPS = next(c for c in CASES if c.id == "stops")


def port_tree(case):
    """The recorded R tree in the port's numbering."""
    m = case.array("merge").astype(np.int64)
    n = m.shape[0] + 1
    return WhclustResult(
        np.where(m < 0, -m - 1, n + m - 1),
        case.array("height"),
        case.array("order").astype(np.int64) - 1,
    )


def c_cutree(merge_r, k):
    """R's C cutree for one k, line by line, 1-based like the C."""
    n = len(merge_r) + 1
    sing = [True] * (n + 1)
    m_nr = [0] * (n + 1)
    z = [0] * (n + 1)
    ans = [0] * (n + 1)
    for step in range(1, n):
        m1, m2 = merge_r[step - 1]
        if m1 < 0 and m2 < 0:
            m_nr[-m1] = m_nr[-m2] = step
            sing[-m1] = sing[-m2] = False
        elif m1 < 0 or m2 < 0:
            if m1 < 0:
                j, m1 = -m1, m2
            else:
                j = -m2
            for l in range(1, n + 1):
                if m_nr[l] == m1:
                    m_nr[l] = step
            m_nr[j] = step
            sing[j] = False
        else:
            for l in range(1, n + 1):
                if m_nr[l] == m1 or m_nr[l] == m2:
                    m_nr[l] = step
        if k == n - step:
            z = [0] * (n + 1)
            nclust = 0
            for l in range(1, n + 1):
                if sing[l]:
                    nclust += 1
                    ans[l] = nclust
                else:
                    if z[m_nr[l]] == 0:
                        nclust += 1
                        z[m_nr[l]] = nclust
                    ans[l] = z[m_nr[l]]
    if k == n:
        ans = list(range(n + 1))
    return np.array(ans[1:]) - 1


@pytest.mark.parametrize("case", TREES, ids=[c.id for c in TREES])
def test_every_k_matches_r(case):
    tree = port_tree(case)
    want = case.array("cut_k").astype(np.int64) - 1
    for k in range(1, want.shape[0] + 1):
        assert cut_tree(tree, k=k).tolist() == want[:, k - 1].tolist(), k


@pytest.mark.parametrize("case", TREES, ids=[c.id for c in TREES])
def test_every_h_matches_r(case):
    tree = port_tree(case)
    if case.manifest["cut_h_error"]:
        with pytest.raises(ValueError) as e:
            cut_tree(tree, h=1.0)
        assert str(e.value) == case.manifest["cut_h_error"]
        return
    want = case.array("cut_h_labels").astype(np.int64) - 1
    for col, h in enumerate(case.array("cut_h").tolist()):
        assert cut_tree(tree, h=h).tolist() == want[:, col].tolist(), h


def test_stops_match_r():
    e = STOPS.manifest["errors"]
    tree = port_tree(STOPS)
    nan_h = WhclustResult(tree.merge, [np.nan, tree.height[1]], tree.order)
    unsorted = WhclustResult(tree.merge, tree.height[::-1] + [0, 1], tree.order)
    none = WhclustResult(np.empty((0, 2), np.int64), np.empty(0), [0])
    for call, key in [
        (lambda: cut_tree(tree), "no_k_or_h"),
        (lambda: cut_tree(tree, k=0), "k_0"),
        (lambda: cut_tree(tree, k=4), "k_4"),
        (lambda: cut_tree(tree, k=float("nan")), "k_nan"),
        (lambda: cut_tree(tree, k=float("inf")), "k_inf"),
        (lambda: cut_tree(tree, k=3e9), "k_big"),
        (lambda: cut_tree(tree, h=float("nan")), "h_nan"),
        (lambda: cut_tree(nan_h, h=1), "h_nan_height"),
        (lambda: cut_tree(unsorted, h=1), "h_unsorted"),
        (lambda: cut_tree(none, k=1), "none_cut"),
    ]:
        with pytest.raises(ValueError) as exc:
            call()
        assert str(exc.value) == e[key], key


def test_k_truncates_like_as_integer():
    tree = port_tree(STOPS)
    assert cut_tree(tree, k=2.9).tolist() == (STOPS.array("k_2_9").astype(np.int64) - 1).tolist()


def test_k_wins_over_h():
    tree = port_tree(STOPS)
    assert cut_tree(tree, k=1, h=-1.0).tolist() == [0, 0, 0]


def test_one_cut_per_call():
    with pytest.raises(ValueError, match="one k or one h"):
        cut_tree(port_tree(STOPS), k=[1, 2])


def _fuzz_merge(rng, n, self_merges):
    """A random tree in R's numbering, or with self_merges, arbitrary earlier entries."""
    active = [-(i + 1) for i in range(n)]
    rows = []
    for step in range(1, n):
        if self_merges and rng.random() < 0.2:
            a = b = int(rng.choice(active))
        else:
            i, j = rng.choice(len(active), 2, replace=False)
            a, b = active[i], active[j]
            active = [c for c in active if c not in (a, b)]
        active.append(step)
        rows.append((a, b))
    return np.array(rows, dtype=np.int64)


@pytest.mark.parametrize("self_merges", [False, True])
def test_numpy_loop_matches_c_on_fuzzed_trees(self_merges):
    from datanuggets.wcluster.cutree import _cutree

    rng = np.random.default_rng(20261010)
    for _ in range(300):
        n = int(rng.integers(2, 40))
        rm = _fuzz_merge(rng, n, self_merges)
        for k in range(1, n + 1):
            assert _cutree(rm, k).tolist() == c_cutree(rm.tolist(), k).tolist()


def test_r_merge_inverts_the_port_numbering():
    from datanuggets.wcluster.cutree import r_merge

    case = TREES[0]
    assert r_merge(port_tree(case).merge).tolist() == case.array("merge").tolist()
