###
# Original script: collect_nucl0.py (simone bacchio)
###
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

gy = np.array([[ 0, 0, 0, 1],
               [ 0, 0,-1, 0],
               [ 0,-1, 0, 0],
               [+1, 0, 0, 0]])

gz = np.array([[ 0, 0, I, 0],
               [ 0, 0, 0,-I],
               [-I, 0, 0, 0],
               [0,+I, 0, 0]])

gt = np.array([[ 1, 0, 0, 0],
               [ 0, 1, 0, 0],
               [ 0, 0,-1, 0],
               [ 0, 0, 0,-1]])

g5 = gx.dot(gy.dot(gz.dot(gt)))

one = np.eye(NS, dtype=complex)

P0p = (1/4)*(one + gt)
P0m = (1/4)*(one - gt)

Pkp = [I*P0p.dot(g5.dot(gk)) for gk in [gx, gy, gz]]
Pkm = [I*P0m.dot(g5.dot(gk)) for gk in [gx, gy, gz]]

@click.command()
@click.argument("fname")
@click.option("-c", "--convention", "conv", default="C", help="ensemble (for changing conventions)")
@click.option("-o", "--output", "oname", default="out.h5", help="output filename")
@click.option("-m", "--momenta", "momname", default="mom.h5", help="momenta list filename")
@click.option("-s", "--momsq", default=None, help="momenta squared to be extracted")
@click.option("-p", "--projs", default="0,x,y,z", help="projectors (any combination of \"0,x,y,z\")")
@click.option("-a", "--average/--no-average", default=False, help="average over source positions")
@click.option("-r", "--root", default="/", help="choose root group of output file. Default is `/'")
@click.option("-q", "--quiet/--no-quiet", default=False, help="suppress progress bar")
def main(fname, conv, oname, momname, momsq, projs, average, root, quiet):
    if momsq is not None:
        momsq = list(map(int, momsq.split(",")))
    prjs = projs.split(",")
    with h5py.File(momname, "r") as fp:
        momvecs = np.array(fp["mvec"][()])
        if momsq is None:
            momsq = sorted(set((momvecs**2).sum(axis=1)))
    with h5py.File(oname, "w") as fo:
        with h5py.File(fname, "r") as fp:
            assert conv in ["C", "B", "D"]
            nucls = {
                "C": [
                    ("nucl1", "nucl1"),
                    ("nucl2", "nucl2")
                ],
                "B": [
                    ("nucl1", "nucl_nucl/twop_baryon_1"),
                    ("nucl2", "nucl_nucl/twop_baryon_2")
                ],
                "D": [
                    ("nucl1", "baryons/nucl_nucl/twop_baryon_1"),
                    ("nucl2", "baryons/nucl_nucl/twop_baryon_2")
                ],
            }[conv]
            if conv in ["C", "B"]:
                cnf = None
                if len(fp) == 1:
                    cnf = list(fp.keys())[0]
                else:
                    for key in fp.keys():
                        if 'conf' in key:
                            cnf = key
                        if re.match("[0-9]{4}_r[01]", key) is not None:
                            cnf = key
                assert cnf is not None
                srcs = fp[cnf]
            if conv == "D":
                cnf = "/"
                srcs = fp[nucls[0][1]]
            dsets = defaultdict(int)
            src_iter =  srcs if quiet else tqdm.tqdm(srcs, ncols=72)
            nsrc = len(src_iter)
            for src in src_iter:
                for nn,nucl in nucls:
                    if conv in ["B", "C"]:
                        grp = fp[cnf][src][nucl]
                    if conv == "D":
                        grp = fp[cnf][nucl][src]
                    itype = grp.dtype
                    assert itype in [np.float32, np.float64]
                    otype = np.complex64 if itype == np.float32 else np.complex128
                    arr = grp[()]
                    for msq in momsq:
                        nmom = momvecs.shape[0]
                        idx = np.arange(nmom, dtype=int)[msq == (momvecs**2).sum(axis=1)]
                        subarr = arr[:,idx,:,:]
                        nt,nm,_,_ = subarr.shape
                        subarr = (subarr[...,0] + I*subarr[...,1]).reshape(nt, nm, NS, NS)
                        for di in ("fwd","bwd"):
                            P = {"fwd": {"0": P0p, "x": Pkp[0], "y": Pkp[1], "z": Pkp[2]},
                                 "bwd": {"0": P0m, "x": Pkm[0], "y": Pkm[1], "z": Pkm[2]}}[di]
                            s = {"fwd": range(0, nt//4, 1),
                                 "bwd": range(0,-nt//4,-1)}[di]
                            for i,p in enumerate(prjs):
                                dset = subarr.dot(P[p])[s,:,:,:].trace(axis1=2, axis2=3)
                                dsets[nn,msq,di,p] += np.array(dset, dtype=otype)
                        if not average:
                            grp = fo.require_group("/"+root+"/"+cnf+"/"+src+"/"+nn+"/msq{:03.0f}".format(msq))
                            grp.create_dataset("mvec", data=momvecs[idx, :])
                            for _,_,di,p in dsets.keys():
                                dgrp = grp.require_group("P"+p)
                                dgrp.create_dataset(di, data=dsets[nn,msq,di,p])
                            dsets = defaultdict(int)
            if average:
                for nucl,msq,di,p in dsets:
                    grp = fo.require_group(cnf+"/"+nucl+"/msq{:03.0f}".format(msq))
                    idx = np.arange(nmom, dtype=int)[msq == (momvecs**2).sum(axis=1)]
                    if "mvec" not in grp:
                        grp.create_dataset("mvec", data=momvecs[idx, :])
                    dgrp = grp.require_group("P"+p)
                    dgrp.create_dataset(di, data=dsets[nucl,msq,di,p]/nsrc)
    return 0


if __name__ == "__main__":
    main()
