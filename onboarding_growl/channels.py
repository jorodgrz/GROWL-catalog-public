"""Sort merging binary black holes into formation paths (channels) from their mass-transfer history.

These are the same rules as the extra section at the end of Notebook 2, where each step is
explained and checked against the data file. Notebook 3 imports this file so both notebooks use identical definitions.

Star 1 is the star born heavier. A history like "SMT(1) to CE(2)" means star 1 did stable mass
transfer, then star 2 overflowed and started a common envelope (CE). We keep the three paths that
make up most of the population (92.5% of the weight) and put everything else in "Other".
"""
import h5py
import numpy as np
import pandas as pd

# plotting order; every merging BBH lands in exactly one of these
PATHS = ["Only stable", "Classic", "No first transfer", "Other"]

# what each path means, in words
MEANING = {"Only stable":       "stable mass transfer from both stars, no CE",
           "Classic":           "stable mass transfer, then star 2 starts a CE",
           "No first transfer": "star 1 never transfers, star 2 starts a CE",
           "Other":             "everything else, e.g. star 1 transfers on the main sequence or starts a CE"}


def classify_paths(bbh, data_path):
    """Return a DataFrame, indexed like ``bbh``, with each binary's ``path`` and ``mt_history``.

    ``bbh`` needs a ``SEED`` column, and its index must be the row number in
    BSE_Double_Compact_Objects (true for the ``bbh`` table built in Notebooks 2 and 3).
    """
    with h5py.File(data_path, "r") as f:
        g = f["BSE_RLOF"]
        rlof_seed = g["SEED"][()]
        keep = np.isin(rlof_seed, bbh.SEED.values)
        events = pd.DataFrame({"SEED": rlof_seed[keep],
                               "event": g["MT_Event_Counter"][()][keep],
                               "is_CE": g["CEE>MT"][()][keep],
                               "rlof1": g["RLOF(1)>MT"][()][keep],
                               "rlof2": g["RLOF(2)>MT"][()][keep]})
        donor_types_1 = np.char.strip(
            f["BSE_Double_Compact_Objects"]["MT_Donor_Hist(1)"][()][bbh.index.values].astype(str))
        sp = f["BSE_System_Parameters"]
        sp_seed = sp["SEED"][()]
        keep = np.isin(sp_seed, bbh.SEED.values)
        che = pd.Series((sp["Stellar_Type@ZAMS(1)"][()][keep] == 16) | (sp["Stellar_Type@ZAMS(2)"][()][keep] == 16),
                        index=sp_seed[keep])
    events = events.sort_values(["SEED", "event"])

    # the raw history, e.g. "SMT(1) to CE(2)"
    donor = np.where(events.rlof1 == 1, np.where(events.rlof2 == 1, "1+2", "1"), "2")
    events["token"] = np.where(events.is_CE == 1, "CE(", "SMT(") + donor + ")"

    # one row per binary, same order as bbh (NaN = never transferred)
    per = events.groupby("SEED").agg(first_CE=("is_CE", "first"), first_r1=("rlof1", "first"),
                                     first_r2=("rlof2", "first"), n_CE=("is_CE", "sum"),
                                     history=("token", " to ".join)).reindex(bbh.SEED.values)

    has_CE = (per.n_CE > 0).values
    first_is_CE = (per.first_CE == 1).values
    first_by_star1 = (per.first_r1 == 1).values
    first_by_both = first_by_star1 & (per.first_r2 == 1).values       # both stars flagged, can't tell who
    star1_never_donated = donor_types_1 == "NA"
    case_A = pd.Series(donor_types_1).str.split("-").str[0].eq("1").values   # star 1 first donated on the main sequence
    is_che = bbh.SEED.map(che).fillna(False).astype(bool).values           # chemically homogeneous always goes to Other

    standard = first_by_star1 & ~first_by_both & ~case_A     # star 1 goes first, and not on the main sequence
    conditions = [~is_che & standard & ~has_CE,
                  ~is_che & standard & has_CE & ~first_is_CE,
                  ~is_che & star1_never_donated & has_CE & ~first_by_both]
    assert (np.sum(conditions, axis=0) <= 1).all(), "a binary matched more than one path"
    return pd.DataFrame({"path": np.select(conditions, PATHS[:3], default="Other"),
                         "mt_history": per.history.fillna("none").values}, index=bbh.index)
