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
      
      

one = np.eye(NS, dtype=complex)

quantities = {
    #
    # kind      : e.g. "local", "deriv"
    # oneend    : "std" or "gen" 
    # gammas    : list of tuples, ("name", matrix to dot with loop)
    # projs     : list of nucleon projectors to include. P0, Px, Py, or Pz
    #
    
    "scalar": {
        "kind": "local",
        "oneend": "std",
        "gammas": [
            ("I*g5", I*g5),
        ],
        "projs": [
            "P0"
        ]
    },
    "axial": {
        "kind": "local",
        "oneend": "gen",
        "gammas": [
            ("I*g5*gt", I*g5.dot(gt)),
            ("I*g5*gx", I*g5.dot(gx)),
            ("I*g5*gy", I*g5.dot(gy)),
            ("I*g5*gz", I*g5.dot(gz)),
        ],
        "projs": [
            "P0",
            "Px",
            "Py",
            "Pz",
        ]
    },
    "tensor": {
        "kind": "local",
        "oneend": "std",
        "gammas": [
            ("I*g5*sitx", I*g5.dot(si["t", "x"])),
            ("I*g5*sity", I*g5.dot(si["t", "y"])),
            ("I*g5*sitz", I*g5.dot(si["t", "z"])),
            ("I*g5*sixy", I*g5.dot(si["x", "y"])),
            ("I*g5*sixz", I*g5.dot(si["x", "z"])),
            ("I*g5*siyz", I*g5.dot(si["y", "z"])),
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
          ("stoch", "gen", "s-probD8-src01"): "loop_probD8.%TRAJ%_stoch_NeV0_Ns0001_step0001_Qsq64.h5",
          ("stoch", "std", "s-probD8-src01"): "loop_probD8.%TRAJ%_stoch_NeV0_Ns0001_step0001_Qsq64.h5",
          ("stoch", "gen", "s-probD8-src02"): "stoch_part_gen.h5",
          ("stoch", "std", "s-probD8-src02"): "stoch_part_std.h5",
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

def get_loop(fname, traj, msq_ins=0, Ns=None, kind="local", conv="v2", oneend="std", flav="l"):
    """
    Retrieves the loop with open dirac indices
    """
    kinds = ["local",] ### Add more here as we implement them
    assert kind in kinds, " `kind' should be one of: [{}]".format(", ".join(kinds))
    if conv == "v2":
        Ns = "" if Ns is None else f"Ns{Ns}"
        with h5py.File(fname, "r") as fp:
            mvec = np.array(fp[f"Conf{traj}/{Ns}/localLoops/mvec"])
            mvec_idx = (mvec**2).sum(axis=1) == msq_ins
            if kind == "local":
                l = np.array(fp[f"Conf{traj}/{Ns}/localLoops/loop"])
    if conv == "v1":
        Ns = "" if Ns is None else "Nstoch_{:04.0f}".format(Ns)
        ncnf = traj.split("_")[0]
        with h5py.File(fname, "r") as fp:
            mvec = np.array(fp[f"Momenta_list_xyz"])
            mvec_idx = (mvec**2).sum(axis=1) == msq_ins
            if kind == "local":
                if oneend == "std":
                    end = "Scalar/loop"
                if oneend == "gen":
                    end = "dOp/loop"                
                l = np.array(fp[f"conf_{ncnf}/{Ns}/{end}"])
        l = l.reshape(-1, mvec.shape[0], 4, 4, 2).transpose(0, 2, 3, 1, 4)
    return mvec[mvec_idx, :], l[...,mvec_idx,0] + I*l[...,mvec_idx,1]

def loop_contract(fname, traj, rename=None, msq_ins=0, quantity="scalar", parts=["stoch"], conv="v2", flav="l", Ns=0):
    """
    Contract and return the loop according to `quantity'
    - Return shape is [part, ngamma, nt, nmvec]
    - `part' is 0: exact, 1: stochastic
    - `ngamma' depends on the entry in the `quantities' dict
    """
    assert quantity in quantities, " `quantity' should be one of: [{}]".format(", ".join(quantities.keys()))
    kind = quantities[quantity]["kind"]
    oneend = quantities[quantity]["oneend"]
    gammas = quantities[quantity]["gammas"]
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
        mvec,l = get_loop(fn, traj, msq_ins=msq_ins, Ns=N, kind=kind, conv=conv, oneend=oneend, flav=flav)
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
    # .mean(axis=0) averages over nucl1, nucl2
    # keep fwd/bwd index since this will be contracted with a different loop time-slice
    return mvec, np.array(ret).reshape([2, len(projs), 2, nt, nmvec]).mean(axis=0)
    
#fname_twop = "/p/scratch/pr74yo/koutsou1/cC80/twop/nucl-twop-sepsrc.h5"
#dname_loop = "/p/scratch/pr74yo/koutsou1/cC80/loop/l/"

@click.command()
@click.argument("two_point_filename")
@click.argument("loops_filename")
@click.argument("traj")
@click.option("-o", "--output", "oname", default="out.h5", help="output filename")
@click.option("-e", "--ensemble", "ens", default="C", help="ensemble, e.g. `B', `C', or `D'")
@click.option("-c", "--old-convention/--no-old-convention", "conv", default=False, help="whether to expect the `old' storage convention used for the B ensemble")
@click.option("-i", "--msq-ins", default=0, help="insertion momentum squared")
@click.option("-s", "--msq-snk", default=0, help="sink momentum squared")
@click.option("-d", "--tsinks", default="4,30", help="tsinks, give as `min(tsink),max(tsink)'")
@click.option("-t", "--quantities", "quants", default="scalar", help="quantity to compute")
@click.option("-p", "--parts", default="stoch", help="parts, e.g. `exact,stoch'")
@click.option("-f", "--flavor", default="l", help="flavor, e.g. `l' or `s'")
@click.option("-n", "--numb-sources", default=None, help="number of stochastic sources. If not specified, use hard-coded defaults")
@click.option("-r", "--rename", default=None, help="renaming convention; give e.g.: `exact:exact_NeV200,stoch:stoch_NeV200_Ns0001_step0001'")
@click.option("-q", "--quiet/--no-quiet", default=False, help="suppress progress bar")
def main(two_point_filename, loops_filename, traj, oname, ens, conv, msq_ins, msq_snk, tsinks, quants, parts, flavor, numb_sources, rename, quiet):
    T = T_extents[ens]
    fname_twop = two_point_filename
    fname_loop = loops_filename
    parts = parts.split(",")
    conv = "v1" if conv is True else "v2"
    dts = list(map(int, tsinks.split(",")))
    dts = range(dts[0], dts[1]+1)
    spos = get_spos(fname_twop, traj)
    thrp = defaultdict(list)
    quants = quants.split(",")
    if numb_sources is None:
        Ns = NSs[ens][flavor]
    else:
        Ns = int(numb_sources)
    loops = {quant: loop_contract(fname_loop, traj, rename=rename, msq_ins=msq_ins, quantity=quant, parts=parts, conv=conv, flav=flavor, Ns=Ns)
             for quant in quants}
    spos_iter = spos if quiet else tqdm.tqdm(spos, ncols=72)
    for spo in spos_iter:
        for quant in quants:
            mv_ins, loop = loops[quant]
            projs = quantities[quant]["projs"]
            mv_snk,twop = get_twop(fname_twop, traj, spo, msq_snk=msq_snk, projs=projs)
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
                    lf = np.multiply.outer(p(loop[:,:,tf[:dt+1],...]), twop[:, 0, dt, :])
                    lb = np.multiply.outer(p(loop[:,:,tb[:dt+1],...]), twop[:, 1, dt, :])
                    # ...
                    # ### Shape: [2 (exact/stoch), ngammas, dt, nmvec_ins, nprojs, nmvec_snk]
                    # # if quant == "axial":
                    # #     arr = lf#0.5*(lf+lb)
                    # # elif quant == "scalar":
                    # arr = 0.5*(lf-lb) ### Backwards nucleon has a sign flipped 
                    # ### Transpose to: [2 (exact/stoch), nprojs, nmvec_snk, nmvec_ins, ngammas, dt]
                    # arr = arr.transpose(0, 4, 5, 3, 1, 2)
                    # thrp[quant, dt].append(arr)
                    #...
                    # arr = 0.5*(lf-lb) ### Backwards nucleon has a sign flipped 
                    ### Transpose to: [2 (exact/stoch), nprojs, nmvec_snk, nmvec_ins, ngammas, dt]
                    arr = np.array([lf, lb]).transpose(0, 1, 5, 6, 4, 2, 3)
                    thrp[quant, dt, prod, "fwd"].append(arr[0,...])
                    thrp[quant, dt, prod, "bwd"].append(arr[1,...])
                
    ### Average over source positions
    thrp = {k: np.array(v).mean(axis=0) for k,v in thrp.items()}
    with h5py.File(oname, "a") as fp:
        for quant in quants:
            mv_ins,loop = loops[quant]
            for i,part in enumerate(parts):
                for k,(ga,_) in enumerate(quantities[quant]["gammas"]):
                    grp_name = "{}/{}/ins_msq{:03.0f}/ins:{}/".format(
                        traj, part, msq_ins, ga)
                    grp = fp.require_group(grp_name)
                    vev = loop[i,k,:,:].sum(axis=0)
                    grp.require_dataset("mvec_ins", shape=mv_ins.shape, dtype=mv_ins.dtype, data=mv_ins)
                    grp.require_dataset("vev", shape=vev.shape, dtype=vev.dtype, data=vev)
                    for j,proj in enumerate(quantities[quant]["projs"]):                    
                        for dt in dts:
                            for prod in ["rprod", "iprod"]:
                                grp_name = "{}/{}/ins_msq{:03.0f}/ins:{}/snk_msq{:03.0f}/{}/dt{:03.0f}/{}".format(
                                    traj, part, msq_ins, ga, msq_snk, proj, dt, prod)
                                for d in ("fwd", "bwd"):
                                    arr = thrp[quant, dt, prod, d][i, j, :, :, k, :]
                                    grp = fp.require_group(grp_name)
                                    grp.require_dataset(d, shape=arr.shape, dtype=arr.dtype, data=arr)
                                    grp.require_dataset("mvec_snk", shape=mv_snk.shape, dtype=mv_snk.dtype, data=mv_snk)

    return 0

if __name__ == "__main__":
    main()
