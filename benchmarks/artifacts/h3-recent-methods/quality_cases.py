import subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
cases=[('texture',431927,'A fixed camera close-up of a blue ceramic teacup on a finely woven linen tablecloth. Thin steam rises gently from the tea. Sunlight illuminates the ceramic glaze and fabric threads. Keep the cup stationary and the background stable. Audio: a quiet room with faint birds outside.'),('motion',957130,'A woman in a green jacket turns her head and smiles while walking slowly along a brick garden path. The camera pans gently with her. Natural daylight, detailed face and hands, leafy background moving smoothly. Audio: soft footsteps and leaves rustling. No speech.')]
modes=['exact','exact','ui-spectrum-protected','spectrum-safe','ui-spectrum-fast','spectrum','ui-motion','motion0.15','speed3-cal01','exact']
for name,seed,prompt in cases:
    cmd=[sys.executable,str(R/'run.py'),'--config','optimized','--model','h3','--frames','22','--steps','20','--candidate',str(R/'candidate.py'),'--modes',','.join(modes),'--label','recent-'+name,'--seed',str(seed),'--prompt-text',prompt]
    with (R/f'recent-{name}.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (R/f'recent-{name}-quality.log').open('w') as log:
        subprocess.run([sys.executable,str(R/'compare.py'),f'recent-{name}-1-exact',*[f'recent-{name}-{i}-{mode}' for i,mode in enumerate(modes) if i>=2]],stdout=log,check=True,env={**__import__('os').environ,'OMP_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2'})
