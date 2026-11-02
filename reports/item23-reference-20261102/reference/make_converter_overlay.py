from pathlib import Path
import shutil
root=Path.home()/'item23-independent-reference-20261102'
source=root/'llama/source-unpatched/gguf-py/gguf'; dest=root/'converter-overlay/gguf'
if dest.exists():shutil.rmtree(dest)
shutil.copytree(source,dest)
p=dest/'gguf_writer.py';s=p.read_text().replace('import numpy as np\n','import numpy as np\n\nfrom .lazy import LazyNumpyTensor\n')
anchor='logger = logging.getLogger(__name__)\n'
helper='''logger = logging.getLogger(__name__)


def _write_numpy_without_fallocate(array: Any, fout: IO[bytes]) -> None:
    # Preserve ndarray.tofile C-order bytes, including lazy/noncontiguous
    # tensors, without NumPy fallocate disabling Btrfs compression.
    if isinstance(array, LazyNumpyTensor):
        array = LazyNumpyTensor.to_eager(array)
    data = np.ascontiguousarray(array)
    view = memoryview(data).cast('B')
    step = 16 * 1024 * 1024
    for start in range(0, len(view), step):
        chunk = view[start:start + step]
        while len(chunk):
            written = fout.write(chunk)
            if written is None or written <= 0:
                raise OSError('short or failed tensor write')
            chunk = chunk[written:]
'''
if anchor not in s:raise SystemExit('missing anchor')
s=s.replace(anchor,helper,1)
s=s.replace('ti.tensor.tofile(fout)','_write_numpy_without_fallocate(ti.tensor, fout)')
s=s.replace('tensor.tofile(self.temp_file)','_write_numpy_without_fallocate(tensor, self.temp_file)')
s=s.replace('tensor.tofile(fout)','_write_numpy_without_fallocate(tensor, fout)')
if '.tofile(' in s:raise SystemExit('unreplaced tofile')
p.write_text(s)
