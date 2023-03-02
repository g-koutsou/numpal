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

NSs = {
    "B": {"l": 1, "s": 12,"c": 12},
    "C": {"l": 0, "s": 0, "c": 0},
    "D": {"l": 0, "s": 0, "c": 0}    
}

T_extents = {"D": 192, "C": 160, "B": 128} ### Could be inferred from shape of loops
L_extents = {"D":  96, "C":  80, "B":  64} ### Needed for correcting phase due to source-position

def get_spos(fname_twop, traj):
    """
    Return array of all source positions
    """
    with h5py.File(fname_twop, "r") as fp:
        spos = list(fp[traj].keys())
    match = "^sx([0-9]*)sy([0-9]*)sz([0-9]*)st([0-9]*)$"
    spos = np.array(list(map(lambda x: re.match(match, x).groups(), spos)), int)
    return spos

def get_loop(fname, traj, max_msq_ins=None, Ns=None, kind="local", conv="v2", oneend="std", flav="l"):
    """
    Retrieves the loop with open dirac indices
    """
    kinds = ["local",] ### Add more here as we implement them
    assert kind in kinds, " `kind' should be one of: [{}]".format(", ".join(kinds))
    if conv == "v2":
        Ns = "" if Ns is None else f"Ns{Ns}"
        with h5py.File(fname, "r") as fp:
            mvec = np.array(fp[f"Conf{traj}/{Ns}/localLoops/mvec"])
            if max_msq_ins is None:
                mvec_idx = slice(0, None)
            else:
                mvec_idx = (mvec**2).sum(axis=1) <= max_msq_ins
            if kind == "local":
                l = np.array(fp[f"Conf{traj}/{Ns}/localLoops/loop"])
    if conv == "v1":
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
        l = l.reshape(-1, mvec.shape[0], 4, 4, 2).transpose(0, 2, 3, 1, 4)
    return mvec[mvec_idx, :], l[...,mvec_idx,0] + I*l[...,mvec_idx,1]

def loop_contract(fname, traj, rename=None, max_msq_ins=None, parts=["stoch"], conv="v2", flav="l", Ns=0):
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
        if rename is None:
            ren = {x: x for x in ["exact","stoch"]}
        else:
            ren = {x.split(":")[0]: x.split(":")[1] for x in rename.split(",")}
        fn = fname.replace("%part%", ren[part])
        if conv == "v2":
            fn = fn.replace("%oe%", oneend)
        N = {"stoch": Ns, "exact": None}[part]
        mvec,l = get_loop(fn, traj, max_msq_ins=max_msq_ins, Ns=N, kind=kind, conv=conv, oneend=oneend, flav=flav)
        ### Transpose dirac indices and move insertion momentum vector
        ### index to second-from-left
        l = l.transpose(0, 3, 2, 1)
        ### Dot with gammas
        ll = [l.dot(ga).trace(axis1=-2,axis2=-1) for _,ga in gammas]
        ret.append(ll)
    return mvec, np.array(ret)

def get_twop(fname, traj, spo, msq_snk=0, projs=["P0"]):
    """
    Return two-point function for a given source position
    - Return shape is [nprojs, 2 (fwd/bwd), nt, nmvec]
    - Returned array is averaged over nucl1, nucl2
    """
    ret = list()
    with h5py.File(fname, "r") as fp:
        s0 = f"sx{spo[0]:02.0f}sy{spo[1]:02.0f}sz{spo[2]:02.0f}st{spo[3]:03.0f}"
        s1 = f"sx{spo[0]:02.0f}sy{spo[1]:02.0f}sz{spo[2]:02.0f}st{spo[3]:02.0f}"            
        top0 = fp[f"{traj}"].get(f"{s0}")
        top1 = fp[f"{traj}"].get(f"{s1}")
        top = top0 if top1 is None else top1
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
        y.append((qsq, thrp[:, 0, :, imv, k+1, :]/mvec[imv, k]))
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
            y.append((qsq, thrp[:, k+1, :, imv, i+1, :]/(eps[i,j,k]*mvec[imv, j])))
    qsqs = sorted(set([x for x,_ in y]))
    e2 = list()
    for qsq in qsqs:
        e2.append(np.mean([t for x,t in y if x == qsq], axis=0))
    return {"q2": np.array(qsqs), "thrp": np.array(e2)}

@click.command()
@click.argument("two_point_filename")
@click.argument("loops_filename")
@click.argument("traj")
@click.option("-o", "--output", "oname", default="out.h5", help="output filename")
@click.option("-e", "--ensemble", "ens", default="C", help="ensemble, e.g. `B', `C', or `D'")
@click.option("-c", "--old-convention/--no-old-convention", "conv", default=False, help="whether to expect the `old' storage convention used for the B ensemble")
@click.option("-i", "--max-msq-ins", default=None, help="max. insertion momentum squared")
@click.option("-s", "--msq-snk", default=0, help="sink momentum squared")
@click.option("-d", "--tsinks", default="4,30", help="tsinks, give as `min(tsink),max(tsink)'")
@click.option("-p", "--parts", default="stoch", help="parts, e.g. `exact,stoch'")
@click.option("-f", "--flavor", default="l", help="flavor, e.g. `l', `s', or `c'")
@click.option("-n", "--numb-sources", default=None, help="number of stochastic sources. If not specified, use hard-coded defaults")
@click.option("-r", "--rename", default=None, help="renaming convention; give e.g.: `exact:exact_NeV200,stoch:stoch_NeV200_Ns0001_step0001'")
@click.option("-q", "--quiet/--no-quiet", default=False, help="suppress progress bar")
def main(two_point_filename, loops_filename, traj, oname, ens, conv, max_msq_ins, msq_snk, tsinks, parts, flavor, numb_sources, rename, quiet):
    T = T_extents[ens]
    L = L_extents[ens]
    fname_twop = two_point_filename
    fname_loop = loops_filename
    parts = parts.split(",")
    conv = "v1" if conv is True else "v2"
    if max_msq_ins is not None:
        max_msq_ins = int(max_msq_ins)
    dts = list(map(int, tsinks.split(",")))
    dts = range(dts[0], dts[1]+1)
    spos = get_spos(fname_twop, traj)
    thrp = defaultdict(np.complex128)
    if numb_sources is None:
        Ns = NSs[ens][flavor]
    else:
        Ns = int(numb_sources)
    mvec_ins,loops = loop_contract(fname_loop, traj, rename=rename, max_msq_ins=max_msq_ins, parts=parts, conv=conv, flav=flavor, Ns=Ns)
    qx,qy,qz = mvec_ins.T
    spos_iter = spos if quiet else tqdm.tqdm(spos, ncols=72)
    for spo in spos_iter:
        sx,sy,sz,st = spo
        projs = operators["vector"]["projs"]
        mv_snk,twop = get_twop(fname_twop, traj, spo, msq_snk=msq_snk, projs=projs)
        tf = (T + np.arange(T) + st) % T
        tb = (T - np.arange(T) + st) % T
        phase = np.exp(I*(qx*sx + qy*sy + qz*sz)*2.0*np.pi/L)
        for dt in dts:
            #
            # rprod: product of real part of loop with two-point
            #
            # iprod: product of imag part of loop with two-point                
            #
            for prod in ["rprod", "iprod"]:
                p = lambda x: {"rprod": x.real, "iprod": x.imag}[prod]
                lf = np.multiply.outer(twop[:, 0, dt, :], phase*p(loops[:,:,tf[:dt+1],...]))
                lb = np.multiply.outer(twop[:, 1, dt, :], phase*p(loops[:,:,tb[:dt+1],...]))
                ### lf.shape and lb.shape is    : [nprojs, nmvec_snk, 2 (exact/stoch), ngammas, dt, nmvec_ins] 
                ### Transpose to                : [2 (exact/stoch), nprojs, nmvec_snk, nmvec_ins, ngammas, dt]
                arr = np.array([lf, lb]).transpose(0, 3, 1, 2, 6, 4, 5)
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
