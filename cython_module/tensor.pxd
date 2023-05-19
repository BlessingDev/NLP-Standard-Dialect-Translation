from libc.stdint cimport int64_t

cdef extern from "torch/torch.h" namespace "c10":
    cdef cppclass Scalar :
        pass

    cdef cppclass IntArrayRef :
        pass

cdef extern from "torch/torch.h" namespace "at":
    cdef cppclass Tensor:
        Tesnsor() except +

        Tensor operator[](const Scalar& index) except+
        Tensor operator[](const Tensor& index) except+
        Tensor operator[](int64_t index) except+
        IntArrayRef sizes() const