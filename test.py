import numpy as np
import ctypes
import pathlib
import os
import tqdm.cli as tqdm

dll_directories = [
    "D:\\Libraries\\libtorch\\lib",
    "D:\\Libraries\\boost_1_81_0\\stage\\lib"
]

for dir in dll_directories:
    dir_path = pathlib.Path(dir)
    if dir_path.exists():
        dll_files = list(dir_path.glob("**/*.dll"))
        dll_bar = tqdm.tqdm(desc=f'dll loading at {dir}',
                        total=len(dll_files),
                        position=0)
        for dll_file in dll_files:
            dll_bar.set_postfix(file_name=dll_file.name)
            dll_bar.update()
            file_path = os.path.join(dir, dll_file.name)
            ctypes.CDLL(file_path)

import cython_module.train_wrapper as tw

if __name__ == "__main__":
    arr = np.random.random((100, 100))
    print(arr)
    
    print("------------")

    tw.tensor_transformation_test(arr)
    