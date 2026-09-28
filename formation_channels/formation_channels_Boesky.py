import numpy as np
import h5py


def get_COMPAS_vars(compas_file, group, variables, mask=None):
    """Return a variable from the COMPAS output data

    Parameters
    ----------
    compas_file : `hdf5 File`
        COMPAS file
    group : `str`
        Group within COMPAS file
    variables : `str` or `list`
        List of column names to access in group (or single column name)
    mask : `bool/array`
        Mask of binaries with same shape as each variable

    Returns
    -------
    var_list : `various`
        Single variable or list of variables (all masked)
    """
    var_list = None
    if isinstance(variables, str):
        var_list = compas_file[group][variables][...].squeeze()
        if mask is not None:
            var_list = var_list[mask]
    else:
        if mask is not None:
            var_list = [compas_file[group][var][...].squeeze()[mask]
                        for var in variables]
        else:
            var_list = [compas_file[group][var][...].squeeze()
                        for var in variables]

    return var_list


RLOF_COLUMNS = ["SEED", "MT_Event_Counter", "RLOF(1)>MT", "RLOF(2)>MT", "CEE>MT",
                "Stellar_Type(1)<MT", "Stellar_Type(2)<MT"]


def identify_formation_channels(seeds, file):
    """Identify the formation channel that produced each seed, following the
    definitions of Broekgaarden et al. (2021, MNRAS 508, 5028; arXiv:2103.02608),
    section 3.1. The primary is the initially more massive star (star 1). The
    numbers are what is put in the ``channels`` output:

    Classic (1) -- In the first mass transfer, the primary overflows its
    Roche lobe stably and has a clear core-envelope structure (HG - TPAGB)
    whilst the secondary star is still on the main sequence. The first mass
    transfer from the secondary afterwards starts before the secondary is
    stripped and leads to a common envelope event.

    Only stable (2) -- The first mass transfer is as in Classic, and the binary
    never goes through a common envelope.

    Single core CEE (3) -- In the first mass transfer, the primary overflows
    its Roche lobe unstably (leading to a CEE) and has a clear core-envelope
    structure (FGB - TPAGB) whilst the secondary star is still on the main
    sequence.

    Double core CEE (4) -- In the first mass transfer, the primary overflows
    its Roche lobe unstably (leading to a CEE) and both the primary and
    secondary have a clear core-envelope structure (FGB - TPAGB).

    Other -- Anything else, e.g. case A mass transfer (primary still on the
    main sequence), or the primary collapsing before any mass transfer:
    -1 if the binary had a CE, -2 if it did not.

    Chemically homogeneous binaries are not separated; they end up in Other.

    Parameters
    ----------
    seeds : `int/array`
        List of seeds that each correspond to a binary (this should be a subset
        of the seeds in the COMPAS output file)
    file : `hdf5 File`
        An open hdf5 file (returned by h5py.File) with the COMPAS output. Its
        BSE_RLOF group needs the columns in ``RLOF_COLUMNS``.

    Returns
    -------
    channels : `int/array`
        List of channels through with each binary formed
    """
    seeds = np.asarray(seeds)
    missing = [c for c in RLOF_COLUMNS if c not in file["BSE_RLOF"]]
    if missing:
        raise KeyError(f"BSE_RLOF is missing {missing}, which are needed to "
                       "identify the formation channels")

    all_rlof_seeds = get_COMPAS_vars(file, "BSE_RLOF", "SEED")
    rlof_mask = np.isin(all_rlof_seeds, seeds)
    rlof_seed, count, rlof_primary, rlof_secondary, cee_flag, \
        stellar_type_1, stellar_type_2 = get_COMPAS_vars(file, "BSE_RLOF",
                                                         RLOF_COLUMNS, rlof_mask)

    # one entry per mass-transfer episode (an episode can span several rows):
    # stellar types from its first row, flags set if set in any of its rows
    order = np.lexsort((count, rlof_seed))
    rlof_seed, count = rlof_seed[order], count[order]
    new_episode = np.r_[True, (rlof_seed[1:] != rlof_seed[:-1])
                        | (count[1:] != count[:-1])]
    starts = np.flatnonzero(new_episode)
    ep_seed, ep_count = rlof_seed[starts], count[starts]
    ep_primary = np.maximum.reduceat(rlof_primary[order].astype(int), starts) > 0
    ep_secondary = np.maximum.reduceat(rlof_secondary[order].astype(int), starts) > 0
    ep_cee = np.maximum.reduceat(cee_flag[order].astype(int), starts) > 0
    ep_type_1 = stellar_type_1[order][starts]
    ep_type_2 = stellar_type_2[order][starts]

    # the first episode of each binary (episodes are sorted by seed, then count)
    first = np.r_[True, ep_seed[1:] != ep_seed[:-1]]
    mt1_seed = ep_seed[first]
    mt1_primary, mt1_cee = ep_primary[first], ep_cee[first]
    mt1_type_1, mt1_type_2 = ep_type_1[first], ep_type_2[first]

    cee_seeds = np.unique(ep_seed[ep_cee])

    # CLASSIC and ONLY STABLE channels
    # 1st transfer, stable RLOF from primary (post-MS, unstripped) onto MS
    classic_or_OS_MT1 = np.logical_and.reduce((mt1_primary,
                                               np.logical_not(mt1_cee),
                                               mt1_type_1 > 1,
                                               mt1_type_1 < 7,
                                               mt1_type_2 <= 1))
    classic_or_OS_seeds = mt1_seed[classic_or_OS_MT1]

    # first transfer from the secondary after that: unstripped, into a CE
    later_secondary = ep_secondary & np.logical_not(first)
    sec_seed, sec_first = np.unique(ep_seed[later_secondary], return_index=True)
    sec_is_classic = np.logical_and(ep_cee[later_secondary][sec_first],
                                    ep_type_2[later_secondary][sec_first] < 7)
    classic_seeds = np.intersect1d(classic_or_OS_seeds, sec_seed[sec_is_classic])
    classic_mask = np.isin(seeds, classic_seeds)

    # never a CE at all
    only_stable_seeds = np.setdiff1d(classic_or_OS_seeds, cee_seeds)
    only_stable_mask = np.isin(seeds, only_stable_seeds)

    # SINGLE CORE CEE channel
    # 1st transfer unstable, primary giant branch onto MS secondary
    single_core = np.logical_and.reduce((mt1_primary,
                                         mt1_cee,
                                         mt1_type_1 > 2,
                                         mt1_type_1 < 7,
                                         mt1_type_2 <= 1))
    single_core_mask = np.isin(seeds, mt1_seed[single_core])

    # DOUBLE CORE CEE channel
    # 1st transfer unstable, primary giant branch onto giant branch secondary
    double_core = np.logical_and.reduce((mt1_primary,
                                         mt1_cee,
                                         mt1_type_1 > 2,
                                         mt1_type_1 < 7,
                                         mt1_type_2 > 2,
                                         mt1_type_2 < 7))
    double_core_mask = np.isin(seeds, mt1_seed[double_core])

    channels = np.zeros(len(seeds), dtype=int)
    channels[classic_mask] = 1
    channels[only_stable_mask] = 2
    channels[single_core_mask] = 3
    channels[double_core_mask] = 4

    # everything else is 'other': -1 with a CE, -2 without
    had_cee = np.isin(seeds, cee_seeds)
    channels[(channels == 0) & had_cee] = -1
    channels[(channels == 0) & ~had_cee] = -2

    return channels
