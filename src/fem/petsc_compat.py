"""
PETSc / SuperLU Compatibility Layer for JAX-FEM.
Provides high-performance SciPy sparse matrix assembly when PETSc C libraries are not installed.
"""
import sys
import types
import numpy as np
import scipy.sparse


class _ScipyMatMock:
    """Emulates PETSc.Mat for JAX-FEM SuperLU direct sparse solver."""
    class Option:
        KEEP_NONZERO_PATTERN = 1

    def __init__(self, size=(1, 1)):
        self.size = size
        self.coo_i = None
        self.coo_j = None
        self.csr = None

    def createAIJ(self, size, **kwargs):
        self.size = size
        return self

    def setOption(self, opt, val):
        pass

    def setPreallocationCOO(self, coo_i, coo_j):
        self.coo_i = np.asarray(coo_i, dtype=np.int32)
        self.coo_j = np.asarray(coo_j, dtype=np.int32)

    def setValuesCOO(self, values):
        coo = scipy.sparse.coo_matrix((values, (self.coo_i, self.coo_j)), shape=self.size)
        self.csr = coo.tocsr()

    def assemble(self):
        pass

    def zeroRows(self, row_inds):
        if self.csr is not None and len(row_inds) > 0:
            for r in row_inds:
                self.csr.data[self.csr.indptr[r]:self.csr.indptr[r+1]] = 0.0
                self.csr[r, r] = 1.0

    def getValuesCSR(self):
        if self.csr is not None:
            return self.csr.indptr, self.csr.indices, self.csr.data
        return np.array([0], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64)

    def getSize(self):
        if self.csr is not None:
            return self.csr.shape
        return self.size

    def transpose(self, out=None):
        if self.csr is not None:
            trans_csr = self.csr.transpose().tocsr()
        else:
            trans_csr = None
        if out is not None and isinstance(out, _ScipyMatMock):
            out.csr = trans_csr
            if trans_csr is not None:
                out.size = trans_csr.shape
            return out
        else:
            new_mat = _ScipyMatMock(self.size[::-1] if self.size else (1, 1))
            new_mat.csr = trans_csr
            return new_mat

    def copy(self, result=None):
        if result is not None and isinstance(result, _ScipyMatMock):
            result.csr = self.csr.copy() if self.csr is not None else None
            result.size = self.size
            return result
        new_m = _ScipyMatMock(self.size)
        new_m.csr = self.csr.copy() if self.csr is not None else None
        return new_m

    def destroy(self):
        pass


class _MockPETSc:
    IntType = np.int32
    ScalarType = np.float64
    RealType = np.float64
    ComplexType = np.complex128
    Mat = _ScipyMatMock


def setup_petsc_mock():
    """Configures sys.modules with lightweight SuperLU PETSc emulator if petsc4py is not present."""
    if "petsc4py" not in sys.modules:
        try:
            import petsc4py  # type: ignore
        except ImportError:
            petsc4py_mod = types.ModuleType("petsc4py")
            petsc4py_mod.PETSc = _MockPETSc
            sys.modules["petsc4py"] = petsc4py_mod
            sys.modules["petsc4py.PETSc"] = _MockPETSc

    if "gmsh" not in sys.modules:
        try:
            import gmsh  # type: ignore
        except ImportError:
            sys.modules["gmsh"] = types.ModuleType("gmsh")


# Automatically initialize on import
setup_petsc_mock()
