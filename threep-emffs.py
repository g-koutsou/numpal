from collections import defaultdict
import sys
import os
import re

import numpy as np
import click
import h5py
import tqdm

NS = 4
I = complex(0,1)

gx = np.array([[ 0, 0, 0, I],
               [ 0, 0, I, 0],
               [ 0,-I, 0, 0],
               [-I, 0, 0, 0]])

gy = np.array([[ 0, 0, 0,+1],
               [ 0, 0,-1, 0],
               [ 0,-1, 0, 0],
               [+1, 0, 0, 0]])

gz = np.array([[ 0, 0, I, 0],
               [ 0, 0, 0,-I],
               [-I, 0, 0, 0],
               [ 0,+I, 0, 0]])

gt = np.array([[ 1, 0, 0, 0],
               [ 0, 1, 0, 0],
               [ 0, 0,-1, 0],
               [ 0, 0, 0,-1]])

g5 = gx.dot(gy.dot(gz.dot(gt)))

si = {
    ("t", "x"): -I/2*(np.dot(gt, gx) - np.dot(gx, gt)),
    ("t", "y"): -I/2*(np.dot(gt, gy) - np.dot(gy, gt)),
    ("t", "z"): -I/2*(np.dot(gt, gz) - np.dot(gz, gt)),
    ("x", "y"): -I/2*(np.dot(gx, gy) - np.dot(gy, gx)),
    ("x", "z"): -I/2*(np.dot(gx, gz) - np.dot(gz, gx)),
    ("y", "z"): -I/2*(np.dot(gy, gz) - np.dot(gz, gy)),
}
      
eps = np.zeros([3, 3, 3])
eps[0, 1, 2] = +1
eps[2, 0, 1] = +1
eps[1, 2, 0] = +1
eps[2, 1, 0] = -1
eps[1, 0, 2] = -1
eps[0, 2, 1] = -1

one = np.eye(NS, dtype=complex)

operators = {
    #
    # kind      : e.g. "local", "deriv"
    # oneend    : "std" or "gen" 
    # gammas    : list of tuples, ("name", matrix to dot with loop)
    # projs     : list of nucleon projectors to include. P0, Px, Py, or Pz
    #
    
    "vector": {
        "kind": "local",
        "oneend": "gen",
        "gammas": [
            ("gt", gt),
            ("gx", gx),
            ("gy", gy),
            ("gz", gz),
        ],
        "projs": [
            "P0",
            "Px",
            "Py",
            "Pz",
        ]
    }
}

loops_names = {
    "D": {("stoch", "gen", "l"): "stoch_part_gen.h5",
          ("stoch", "std", "l"): "stoch_part_std.h5",
          ("stoch", "gen", "s"): "stoch_part_gen.h5",
          ("stoch", "std", "s"): "stoch_part_std.h5",
          ("stoch", "gen", "c"): "stoch_part_gen.h5",
          ("stoch", "std", "c"): "stoch_part_std.h5",},
    "C": {("exact", "gen", "l"): "exact_part_gen.h5",
          ("exact", "std", "l"): "exact_part_std.h5",
          ("stoch", "gen", "l"): "stoch_part_gen.h5",
          ("stoch", "std", "l"): "stoch_part_std.h5",
          ("stoch", "gen", "s"): "stoch_part_gen.h5",
          ("stoch", "std", "s"): "stoch_part_std.h5",
          ("stoch", "gen", "c"): "stoch_part_gen.h5",
          ("stoch", "std", "c"): "stoch_part_std.h5",},
    "B": {("exact", "gen", "l"): "loop_probD8.%TRAJ%_exact_NeV200_Qsq64.h5",
          ("exact", "std", "l"): "loop_probD8.%TRAJ%_exact_NeV200_Qsq64.h5",
          ("stoch", "gen", "l"): "loop_probD8.%TRAJ%_stoch_NeV200_Ns0001_step0001_Qsq64.h5",
          ("stoch", "std", "l"): "loop_probD8.%TRAJ%_stoch_NeV200_Ns0001_step0001_Qsq64.h5",
          ("stoch", "gen", "s"): "loop_probD4.%TRAJ%_stoch_NeV0_Ns0012_step0001_Qsq64.h5",
          ("stoch", "std", "s"): "loop_probD4.%TRAJ%_stoch_NeV0_Ns0012_step0001_Qsq64.h5",
          ("stoch", "gen", "c"): "loop_probD4.%TRAJ%_stoch_NeV0_Ns0012_step0001_Qsq64.h5",
          ("stoch", "std", "c"): "loop_probD4.%TRAJ%_stoch_NeV0_Ns0012_step0001_Qsq64.h5",},
}

NSs = {
    "B": {"l": 1, "s": 12,"c": 12},
    "C": {"l": 0, "s": 0, "c": 0},
    "D": {"l": 0, "s": 0, "c": 0}    
}

T_extents = {"D": 192, "C": 160, "B": 128}

def get_spos(fname_twop, traj):
    """
    Return array of all source positions
    """
    with h5py.File(fname_twop, "r") as fp:
        spos = list(fp[traj].keys())
    match = "^sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)$"
    spos = np.array(list(map(lambda x: re.match(match, x).groups(), spos)), int)
    return spos

def get_loop(fname, traj, max_msq_ins=None, Ns=None, kind="local", conv="C", oneend="std", flav="l"):
    """
    Retrieves the loop with open dirac indices
    """
    T = T_extents[conv]
    kinds = ["local",] ### Add more here as we implement them
    assert kind in kinds, " `kind' should be one of: [{}]".format(", ".join(kinds))
    if conv in ["C","D"]:
        Ns = "" if Ns is None else f"Ns{Ns}"
        with h5py.File(fname, "r") as fp:
            mvec = np.array(fp[f"Conf{traj}/{Ns}/localLoops/mvec"])
            if max_msq_ins is None:
                mvec_idx = slice(0, None)
            else:
                mvec_idx = (mvec**2).sum(axis=1) <= max_msq_ins
            if kind == "local":
                l = np.array(fp[f"Conf{traj}/{Ns}/localLoops/loop"])
    if conv == "B":
        Ns = "" if Ns is None else "Nstoch_{:04.0f}".format(Ns)
        ncnf = traj.split("_")[0]
        with h5py.File(fname, "r") as fp:
            mvec = np.array(fp[f"Momenta_list_xyz"])
            if max_msq_ins is None:
                mvec_idx = slice(0, None)
            else:
                mvec_idx = (mvec**2).sum(axis=1) <= max_msq_ins
            if kind == "local":
                if oneend == "std":
                    end = "Scalar/loop"
                if oneend == "gen":
                    end = "dOp/loop"                
                l = np.array(fp[f"conf_{ncnf}/{Ns}/{end}"])
        l = l.reshape(T, mvec.shape[0], 4, 4, 2).transpose(0, 2, 3, 1, 4)
    return mvec[mvec_idx, :], l[...,mvec_idx,0] + I*l[...,mvec_idx,1]

def loop_contract(dname, traj, max_msq_ins=None, parts=["stoch"], conv="C", flav="l", Ns=0):
    """
    Contract and return the loop
    - If max_msq_ins is None, return all mvec
    - Return shape is [part, ngamma, nt, nmvec]
    - `part' is 0: exact, 1: stochastic
    - `ngamma' depends on the entry in the `quantities' dict
    """
    kind = operators["vector"]["kind"]
    oneend = operators["vector"]["oneend"]
    gammas = operators["vector"]["gammas"]
    ret = list()
    for i,part in enumerate(parts):
        fn = loops_names[conv][part, oneend, flav].replace("%TRAJ%", f"{traj}")
        N = {"stoch": Ns, "exact": None}[part]
        fname = f"{dname}/{traj}/{fn}"
        mvec,l = get_loop(fname, traj, max_msq_ins=max_msq_ins, Ns=N, kind=kind, conv=conv, oneend=oneend, flav=flav)
        ### Transpose dirac indices and move insertion momentum vector
        ### index to second-from-left
        l = l.transpose(0, 3, 2, 1)
        ### Dot with gammas
        ll = [l.dot(ga).trace(axis1=-2,axis2=-1) for _,ga in gammas]
        ret.append(ll)
    return mvec, np.array(ret)

def get_twop(fname, traj, spo, msq_snk=0, projs=["P0"], conv="C"):
    """
    Return two-point function for a given source position
    - Return shape is [nprojs, 2 (fwd/bwd), nt, nmvec]
    - Returned array is averaged over nucl1, nucl2
    """
    ret = list()
    with h5py.File(fname, "r") as fp:
        if conv == "C":
            s = f"sx{spo[0]:02.0f}sy{spo[1]:02.0f}sz{spo[2]:02.0f}st{spo[3]:03.0f}"
        if conv in ["B","D"]:
            s = f"sx{spo[0]:02.0f}sy{spo[1]:02.0f}sz{spo[2]:02.0f}st{spo[3]:02.0f}"            
        top = fp[f"{traj}/{s}"]
        for i,nucl in enumerate(["nucl1","nucl2"]):
            for j,proj in enumerate(projs):
                for k,di in enumerate(("fwd","bwd")):
                    arr = np.array(top[f"{nucl}/msq{msq_snk:03.0f}/{proj}/{di}"])
                    nt,nmvec = arr.shape
                    ret.append(arr)
        mvec = np.array(top[f"nucl1/msq{msq_snk:03.0f}/mvec"])
    return mvec, np.array(ret).reshape([2, len(projs), 2, nt, nmvec]).mean(axis=0)

def get_E0(mvec, thrp):
    qsqs = sorted(set((mvec**2).sum(axis=1)))
    e0 = list()
    for qsq in qsqs:
        idx = (mvec**2).sum(axis=1) == qsq
        e0.append(thrp[:, 0, :, idx, 0, :].mean(axis=0))
    return {"q2": np.array(qsqs), "thrp": np.array(e0)}

def get_E1(mvec, thrp):
    """
    -\frac{i}{N} \sum_{q_i \ne 0} \frac{1}{q_i}\Pi^i(\Gamma_0, \vec{q})
    """
    y = list()
    for imv,k in zip(*np.where(mvec != 0)):
        qsq = np.sum(mvec[imv,:]**2)
        y.append((qsq, thrp[:, 0, :, imv, k, :]/mvec[imv, k]))
    qsqs = sorted(set([x for x,_ in y]))
    e1 = list()
    for qsq in qsqs:
        e1.append(np.mean([t for x,t in y if x == qsq], axis=0))
    return {"q2": np.array(qsqs), "thrp": -I*np.array(e1)}

def get_EM(mvec, thrp):
    """
    \frac{1}{N} \sum_{\substack{i\ne j\ne k\ne i\\ q_j \ne 0}} \frac{1}{\epsilon_{ijk} q_j}\Pi^i(\Gamma_k, \vec{q})
    """
    y = list()
    for imv,j in zip(*np.where(mvec != 0)):
        for i,k in zip(*np.where(eps[:, j, :] != 0)):
            qsq = np.sum(mvec[imv,:]**2)
            y.append((qsq, thrp[:, k, :, imv, i, :]/(eps[i,j,k]*mvec[imv, j])))
    qsqs = sorted(set([x for x,_ in y]))
    e2 = list()
    for qsq in qsqs:
        e2.append(np.mean([t for x,t in y if x == qsq], axis=0))
    return {"q2": np.array(qsqs), "thrp": np.array(e2)}

@click.command()
@click.argument("two_point_filename")
@click.argument("loops_dirname")
@click.argument("traj")
@click.option("-o", "--output", "oname", default="out.h5", help="output filename")
@click.option("-c", "--convention", "conv", default="C", help="file convention, e.g. `B', `C', or `D'")
@click.option("-i", "--max-msq-ins", default=None, help="max. insertion momentum squared")
@click.option("-s", "--msq-snk", default=0, help="sink momentum squared")
@click.option("-d", "--tsinks", default="4,30", help="tsinks, give as `min(tsink),max(tsink)'")
@click.option("-p", "--parts", default="stoch", help="parts, e.g. `exact,stoch'")
@click.option("-f", "--flavor", default="l", help="flavor, e.g. `l', `s', or `c'")
@click.option("-q", "--quiet/--no-quiet", default=False, help="suppress progress bar")
def main(two_point_filename, loops_dirname, traj, oname, conv, max_msq_ins, msq_snk, tsinks, parts, flavor, quiet):
    T = T_extents[conv]
    fname_twop = two_point_filename
    dname_loop = loops_dirname
    parts = parts.split(",")
    if max_msq_ins is not None:
        max_msq_ins = int(max_msq_ins)
    dts = list(map(int, tsinks.split(",")))
    dts = range(dts[0], dts[1]+1)
    spos = get_spos(fname_twop, traj)
    thrp = defaultdict(np.complex128)
    Ns = NSs[conv][flavor]
    mvec_ins,loops = loop_contract(dname_loop, traj, max_msq_ins=max_msq_ins, parts=parts, conv=conv, flav=flavor, Ns=Ns)
    spos_iter = spos if quiet else tqdm.tqdm(spos, ncols=72)
    for spo in spos_iter:
        projs = operators["vector"]["projs"]
        mv_snk,twop = get_twop(fname_twop, traj, spo, msq_snk=msq_snk, projs=projs, conv=conv)
        t0 = spo[-1]
        tf = (T + np.arange(T) + t0) % T
        tb = (T - np.arange(T) + t0) % T
        for dt in dts:
            #
            # rprod: product of real part of loop with two-point
            #
            # iprod: product of imag part of loop with two-point                
            #
            for prod in ["rprod", "iprod"]:
                p = lambda x: {"rprod": x.real, "iprod": x.imag}[prod]
                lf = np.multiply.outer(p(loops[:,:,tf[:dt+1],...]), twop[:, 0, dt, :])
                lb = np.multiply.outer(p(loops[:,:,tb[:dt+1],...]), twop[:, 1, dt, :])
                arr = 0.5*(lf-lb) ### Backwards nucleon has a sign flipped 
                ### Transpose to: [2 (exact/stoch), nprojs, nmvec_snk, nmvec_ins, ngammas, dt]
                arr = np.array([lf, lb]).transpose(0, 1, 5, 6, 4, 2, 3)
                thrp[dt, prod, "fwd"] += arr[0,...]
                thrp[dt, prod, "bwd"] += arr[1,...]
                
    ### Normalize over number of source positions
    thrp = {k: np.array(v)/len(spos) for k,v in thrp.items()}

    ### Get intermediate terms E0, E1, and EM
    ems = {k: {"E0": get_E0(mvec_ins, v),
               "E1": get_E1(mvec_ins, v),
               "EM": get_EM(mvec_ins, v)}
           for k,v in thrp.items()}

    ### Store
    with h5py.File(oname, "a") as fp:
        for key in ems.keys():
            dt,prod,di = key
            for i,part in enumerate(parts): 
                for ff in ("E0","E1","EM"):                    
                    qsq = ems[key][ff]["q2"]
                    dat = ems[key][ff]["thrp"][:,i,:,:]
                    grp_name = f"{traj}/{prod}/{di}/{part}/{ff}/dt{dt}"
                    grp = fp.require_group(grp_name)
                    if flavor in grp:
                        del grp[flavor]
                    grp.create_dataset(flavor, shape=dat.shape, dtype=dat.dtype, data=dat)
                    grp_name = f"{traj}/{prod}/{di}/{part}/{ff}/"
                    grp = fp.require_group(grp_name)
                    if "q2" in grp:
                        del grp["q2"]
                    grp.create_dataset("q2", shape=qsq.shape, dtype=qsq.dtype, data=qsq)
    return 0

if __name__ == "__main__":
    main()
