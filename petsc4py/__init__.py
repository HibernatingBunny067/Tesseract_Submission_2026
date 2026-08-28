import numpy as np
import scipy.sparse

class DummyMat:
    class Option:
        KEEP_NONZERO_PATTERN = 1

    def __init__(self):
        self.shape = None
        self.I = None
        self.J = None
        self.V = None
        self.csr = None

    def createAIJ(self, size, csr=None):
        self.shape = tuple(size)
        if csr is not None:
            indptr, indices, data = csr
            self.csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=self.shape)
        return self

    def setOption(self, opt, val):
        pass

    def setPreallocationCOO(self, I, J):
        self.I = I
        self.J = J

    def setValuesCOO(self, V):
        self.V = V

    def assemble(self):
        self.csr = scipy.sparse.coo_matrix((self.V, (self.I, self.J)), shape=self.shape).tocsr()

    def zeroRows(self, rows):
        # zero rows and set diagonal to 1.0 as PETSc zeroRows typically does
        for row in rows:
            start = self.csr.indptr[row]
            end = self.csr.indptr[row+1]
            self.csr.data[start:end] = 0.0
            cols = self.csr.indices[start:end]
            diag_idx = np.where(cols == row)[0]
            if len(diag_idx) > 0:
                self.csr.data[start + diag_idx[0]] = 1.0

    def getValuesCSR(self):
        return self.csr.indptr, self.csr.indices, self.csr.data

    def matMult(self, other):
        res = DummyMat()
        res.shape = (self.shape[0], other.shape[1])
        res.csr = self.csr.dot(other.csr)
        return res

    def transpose(self, out=None):
        res = out if out is not None else DummyMat()
        res.shape = (self.shape[1], self.shape[0])
        res.csr = self.csr.transpose().tocsr()
        return res

class DummyPETSc:
    IntType = np.int32
    ScalarType = np.float64
    class NormType:
        NORM_INFINITY = 0
    class Vec:
        def createSeq(self, n):
            pass
    class KSP:
        def create(self):
            pass
    class PC:
        pass
    Mat = DummyMat

PETSc = DummyPETSc()
