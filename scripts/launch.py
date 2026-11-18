"""Launch this skill with local configuration and local model files."""
import json
import os
from pathlib import Path
import sys
import sysconfig
from verify_weights import verify
ROOT=Path(__file__).resolve().parent.parent

def main():
    args=sys.argv[1:]
    weights=None
    for index,arg in enumerate(args):
        if arg=='--weights-dir' and index+1<len(args):weights=args[index+1]
        elif arg.startswith('--weights-dir='):weights=arg.split('=',1)[1]
    spec=json.loads((ROOT/'references/models.json').read_text(encoding='utf-8'))
    if not any(arg in ('-h','--help') for arg in args):
        try:verify(weights,quiet=True)
        except (OSError,ValueError) as error:
            print(error,file=sys.stderr)
            return 2
    env=os.environ.copy()
    env['PYTHONNOUSERSITE']='1'
    env['PYTHONDONTWRITEBYTECODE']='1'
    libs=[Path(sys.prefix)/'lib',Path(sys.base_prefix)/'lib']
    if spec['format']=='onnx':
        site=Path(sysconfig.get_path('purelib'))
        libs += [site/'nvidia'/name/'lib' for name in ('cublas','cudnn','cufft','curand','cuda_runtime')]
    paths=list(dict.fromkeys(str(p) for p in libs if p.is_dir()))
    if env.get('LD_LIBRARY_PATH'):paths.append(env['LD_LIBRARY_PATH'])
    env['LD_LIBRARY_PATH']=':'.join(paths)
    os.execve(sys.executable,[sys.executable,'-B',str(ROOT/'scripts/predict.py'),*args],env)
if __name__=='__main__':raise SystemExit(main())
