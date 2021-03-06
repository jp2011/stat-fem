import numpy as np
from firedrake import Function
from firedrake.petsc import PETSc
from firedrake.functionspace import VectorFunctionSpace
from firedrake.functionspaceimpl import WithGeometry
from firedrake.interpolation import interpolate
from firedrake.vector import Vector

class InterpolationMatrix(object):
    "class representing an interpolation matrix"
    def __init__(self, function_space, coords):
        "create and assemble interpolation matrix"

        if not isinstance(function_space, WithGeometry):
            raise TypeError("bad input type for function_space: must be a FunctionSpace")

        self.coords = np.copy(coords)
        self.function_space = function_space
        self.comm = function_space.comm

        self.n_data = coords.shape[0]
        assert (coords.shape[1] == self.function_space.mesh().cell_dimension()
                ), "shape of coordinates does not match mesh dimension"

        # allocate working vectors to handle parallel matrix operations and data transfer

        # dataspace_vector is a distributed PETSc vector in the data space

        self.dataspace_distrib = PETSc.Vec().create(comm=self.comm)
        self.dataspace_distrib.setSizes((-1, self.n_data))
        self.dataspace_distrib.setFromOptions()

        self.n_data_local = self.dataspace_distrib.getSizes()[0]

        # all data computations are done on root process, so create gather method to
        # facilitate this data transfer

        self.petsc_scatter, self.dataspace_gathered = PETSc.Scatter.toZero(self.dataspace_distrib)

        self.meshspace_vector = Function(self.function_space).vector()

        self.n_mesh_local = self.meshspace_vector.local_size()
        self.n_mesh = self.meshspace_vector.size()

        nnz = len(self.function_space.cell_node_list[0])

        self.interp = PETSc.Mat().create(comm=self.comm)
        self.interp.setSizes(((self.n_mesh_local, -1), (self.n_data_local, -1)))
        self.interp.setPreallocationNNZ(nnz)
        self.interp.setFromOptions()
        self.interp.setUp()

        self.is_assembled = False

    def assemble(self):
        "compute values and assemble interpolation matrix"

        if self.is_assembled:
            return

        mesh = self.function_space.ufl_domain()
        W = VectorFunctionSpace(mesh, self.function_space.ufl_element())
        X = interpolate(mesh.coordinates, W)
        meshvals_local = np.array(X.dat.data_with_halos)
        imin, imax = self.interp.getOwnershipRange()

        # loop over all data points

        for i in range(self.n_data):
            cell = self.function_space.mesh().locate_cell(self.coords[i])
            if not cell is None:
                nodes = self.function_space.cell_node_list[cell]
                points = meshvals_local[nodes]
                interp_coords = interpolate_cell(self.coords[i], points)
                for (node, val) in zip(nodes, interp_coords):
                    if node < self.n_mesh_local:
                        self.interp.setValue(imin + node, i, val)

        self.interp.assemble()

        self.is_assembled = True

    def _gather(self):
        "wrapper to transfer data from distributed dataspace vector to root"

        self.petsc_scatter.scatter(self.dataspace_distrib, self.dataspace_gathered,
                                   mode=PETSc.ScatterMode.SCATTER_FORWARD)

    def _scatter(self):
        "wrapper to transfer data from root to distributed dataspace vector"

        self.petsc_scatter.scatter(self.dataspace_gathered, self.dataspace_distrib,
                                   mode=PETSc.ScatterMode.SCATTER_REVERSE)

    def destroy(self):
        "deallocate memory for PETSc vectors and matrix"

        self.dataspace_gathered.destroy()
        self.dataspace_distrib.destroy()
        self.interp.destroy()

    def interp_data_to_mesh(self, data_array):
        "take a gathered numpy array in the data space and interpolate to a distributed mesh vector"

        if not self.is_assembled:
            self.assemble()

        data_array = np.array(data_array)

        assert data_array.ndim == 1, "input data must be a 1D array of sensor values"
        assert data_array.shape[0] == self.dataspace_gathered.getSizes()[0], "data_array has an incorrect number of sensor measurements"

        # scatter into dataspace_distrib

        self.dataspace_gathered.array = np.copy(data_array)
        self._scatter()

        with self.meshspace_vector.dat.vec as vec:
            self.interp.mult(self.dataspace_distrib, vec)

        return self.meshspace_vector.copy()

    def interp_mesh_to_data(self, input_mesh_vector):
        "take a distributed mesh vector and interpolate to a gathered numpy array"

        if not self.is_assembled:
            self.assemble()

        # check vector local sizes and copy values
        if not isinstance(input_mesh_vector, Vector):
            raise TypeError("input_mesh_vector must be a firedrake vector")
        assert (input_mesh_vector.local_size() == self.meshspace_vector.local_size() and
                input_mesh_vector.size() == self.meshspace_vector.size()), "input vector must be the same length as the FEM DOFs and be distributed across processes in the same way"
        self.meshspace_vector.set_local(input_mesh_vector.get_local())

        # interpolate to dataspace and gather, returning numpy array

        with self.meshspace_vector.dat.vec_ro as b:
            self.interp.multTranspose(b, self.dataspace_distrib)

        self._gather()

        return np.copy(self.dataspace_gathered.array)

    def get_meshspace_column_vector(self, idx):
        "returns distributed meshspace column vector for a given data point"

        assert idx >= 0 and idx < self.n_data, "idx out of range"

        if not self.is_assembled:
            self.assemble()

        f = Function(self.function_space).vector()

        f.set_local(self.interp.getColumnVector(idx).array)

        return f

    def __str__(self):
        "return string representation of interpolation matrix"

        return "Interpolation matrix from {} mesh points to {} data points".format(self.n_mesh, self.n_data)

def interpolate_cell(data_coord, nodal_points):
    """
    interpolate between nodal points and data point
    note at present this technically does a projection by performing a least squares solution
    in the case that there are more constraints than free parameters
    """

    if nodal_points.ndim == 1:
        nodal_points = np.reshape(nodal_points, (-1, 1))
    npts, ndim = nodal_points.shape
    assert len(data_coord) == ndim, "data provided for interpolation has the wrong number of spatial dimensions"

    A = np.ones((ndim + 1, npts))
    A[0:-1,:] = np.transpose(nodal_points)
    b = np.array(list(data_coord) + [1.])

    val = np.linalg.lstsq(A, b, rcond=None)[0]

    eps = 1.e-10

    assert np.abs(np.sum(val) - 1.) <= eps, "interpolation between data and coordinates failed"

    return val
