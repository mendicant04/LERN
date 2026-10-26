"""Small CPU example; no dataset or checkpoint is needed."""
import numpy as np
import torch
from lern import LERN

torch.set_num_threads(2)
rng=np.random.default_rng(0)
h=rng.normal(size=(160,16))
v=rng.normal(size=(160,8))
text=rng.normal(size=(4,8))

adapter=LERN(text,dataset='synthetic',order_seed=0,device='cpu')
predictions=[]
for i in range(len(h)):
    result=adapter.observe(v[i:i+1],h[i:i+1],[i],chunk=1)
    predictions.append(int(result['online_local_statistic'][0]))
print('Processed',adapter.n,'images. First predictions:',predictions[:10])
