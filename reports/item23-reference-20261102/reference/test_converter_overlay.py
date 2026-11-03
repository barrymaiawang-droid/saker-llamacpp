import hashlib,io,json,os,tempfile
from pathlib import Path
import numpy as np
from gguf.gguf_writer import _write_numpy_without_fallocate
from gguf.lazy import LazyNumpyTensor

class ShortSink(io.BytesIO):
 def write(self,b): return super().write(b[:7])
def test(label,arr,lazy=False,short=False):
 with tempfile.TemporaryFile() as f:
  arr.tofile(f);f.seek(0);expected=f.read()
 tensor=LazyNumpyTensor.from_eager(arr) if lazy else arr
 out=ShortSink() if short else io.BytesIO();_write_numpy_without_fallocate(tensor,out);actual=out.getvalue()
 if actual!=expected:raise ValueError(f'{label}: byte mismatch')
 return {'name':label,'length':len(actual),'sha256':hashlib.sha256(actual).hexdigest(),'ok':True}
a=np.arange(35,dtype=np.float32).reshape(5,7)
cases=[('contiguous',a,False,False),('transposed',a.T,False,False),('reversed',a[:,::-1],False,False),('fortran',np.asfortranarray(a),False,False),('big-endian',a.astype('>f4'),False,False),('empty',np.array([],dtype=np.float32),False,False),('scalar',np.array(1.25,dtype='>f4'),False,False),('lazy-reversed',a[:,::-1],True,False),('partial-writes',a,False,True),('multi-chunk',np.arange(5_000_000,dtype=np.float32),False,False)]
results=[test(*x) for x in cases]
root=Path.home()/'item23-independent-reference-20261102';(root/'converter-overlay-tests.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
