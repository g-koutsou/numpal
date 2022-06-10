import click
import h5py
import tqdm
import os
import re

@click.command()
@click.argument("fname1", nargs=1)
@click.argument("fname2", nargs=-1)
@click.option("-o", "--output", "oname", default="out.h5", help="output filename")
@click.option("-r", "--replace", default=None, help="a regular expression applied to the source group names to obtain the destination group names. Use `@pattern@replacement@'")
@click.option("-q", "--quiet/--no-quiet", default=False, help="suppress progress bar")
@click.option("-q", "--quiet/--no-quiet", default=False, help="suppress progress bar")
@click.option("-a", "--append/--no-append", default=None, help="If `--append' open in r/w mode. If `--no-append' file will be truncated if it exists. If neither specified, will fail if file exists.")
def main(fname1, fname2, oname, replace, quiet, append):
    mode = "w"
    if append is None:
        if os.path.isfile(oname):
            raise ValueError(f" {oname}: file exists. Use --append or --no-append to specify behavior")
    elif append:
        mode = "a"
    fnames = (fname1,) + fname2
    iter_fns = fnames if quiet else tqdm.tqdm(fnames, ncols=72)
    with h5py.File(oname, mode) as op:
        if replace is not None:
            assert len(replace.split("@")) == 4, "malformed replacement string"
            patt = replace.split("@")[1]
            repl = replace.split("@")[2]
        for fn in iter_fns:
            with h5py.File(fn, "r") as fp:
                names = list()
                fp.visititems(lambda x,y: names.append(x) if isinstance(y, h5py.Dataset) else None)
                for d in names:
                    if d in op:
                        del op[d]
                    if replace is not None:
                        dest = re.sub(patt, repl, d)
                    else:
                        dest = d
                    fp.copy(d, op, name=dest)
    return 0

if __name__ == "__main__":
    main()
