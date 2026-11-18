"""Verify the five fold_* model files using the published SHA256 values."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent

def verify(directory=None,quiet=False):
    spec=json.loads((ROOT/'references/models.json').read_text(encoding='utf-8'))
    directory=Path(directory).expanduser() if directory else ROOT/spec['directory']
    failures=[]
    for name,expected in spec['sha256'].items():
        path=directory/name
        if not path.is_file():
            failures.append(f'Missing {path}')
            continue
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                digest.update(block)
        if digest.hexdigest()!=expected:
            failures.append(f'SHA256 mismatch: {path}')
        elif not quiet:
            print('OK',path)
    if failures:
        raise ValueError('\n'.join(failures)+'\nDownload the matching five files from '+spec['download_url'])
    return directory.resolve()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--quiet',action='store_true')
    args=parser.parse_args()
    try:verify(args.directory,args.quiet)
    except (OSError,ValueError) as error:parser.exit(2,str(error)+'\n')
if __name__=='__main__':main()
