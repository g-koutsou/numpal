from collections import defaultdict
import os
import re

import numpy as np
import click
import h5py
import tqdm

@click.command()
@click.argument("fname", nargs=1)
@click.option("-o", "--output", "oname", default="out.h5", help="output filename")
@click.option("-a", "--append/--no-append", default=None, help="if `--append' open in r/w mode. If `--no-append' file will be truncated if it exists. If neither specified, will fail if file exists.")
@click.option("-c", "--common-datasets", "common", default=None, help="names (comma delim.) of common datasets -- will replace first trajectories")
@click.option("-q", "--quiet/--no-quiet", default=False, help="suppress progress bar")
def main(fname, oname, append, common, quiet):
    if common is None:
        common = []
    else:
        common = common.split(",")
    mode = "w"
    if append is None:
        if os.path.isfile(oname):
            raise ValueError(f" {oname}: file exists. Use --append or --no-append to specify behavior")
    elif append:
        mode = "a"
    with h5py.File(fname, "r") as fp:
        trajs = list(filter(lambda x: re.match("[0-9]{4}_r.", x), fp))
        assert trajs != [], " No trajectory pattern found under root group"
        data = defaultdict(list)
        iter_trajs = trajs if quiet else tqdm.tqdm(trajs, ncols=72)
        for traj in iter_trajs:
            names = list()
            fp[traj].visititems(lambda x,y: names.append(x) if isinstance(y, h5py.Dataset) else None)
            for name in names:
                data[name].append(fp[traj][name][()])
    with h5py.File(oname, mode) as op:
        confs = np.array(trajs, "S")
        op.require_dataset("confs", confs.shape, confs.dtype, data=confs)
        for key,val in data.items():
            arr = np.array(val)
            grp_name = "/".join(key.split("/")[:-1])
            dst_name = key.split("/")[-1]
            grp = op.require_group(grp_name)
            if dst_name in common:
                arr = arr[0,...]
            grp.require_dataset(dst_name, arr.shape, arr.dtype, data=arr)
    return 0

if __name__ == "__main__":
    main()
