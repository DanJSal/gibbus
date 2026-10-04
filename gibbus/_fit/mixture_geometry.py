"""Reduced physical coordinates and exact cone faces for natural mixtures."""

from dataclasses import dataclass

import numpy as np

from .conic_qp import _ConicRepresentation, _support_representation


class _MixtureNaturalMap:
    """Map private polynomial shapes and shared physical boundary amplitudes.

    Parameters
    ----------
    specs : sequence of model specifications
        Nonempty component specifications with matching physical boundary bases.
    layouts : sequence of _NaturalLayout or None, optional
        Component layouts; ``None`` uses each specification's own layout.
    """

    def __init__(self, specs, layouts=None):
        """Build component-to-joint indices and a cache of exact cone faces."""
        self.specs = tuple(specs)
        self.layouts = (
            tuple(spec.layout for spec in self.specs)
            if layouts is None
            else tuple(layouts)
        )
        if not self.specs or len(self.specs) != len(self.layouts):
            raise ValueError("a mixture map requires matching nonempty specs/layouts")
        self.physical_indices = tuple(
            (
                (layout.lower_a_index, layout.upper_a_index)
                if spec.coordinate.direction > 0.0
                else (layout.upper_a_index, layout.lower_a_index)
            )
            for spec, layout in zip(self.specs, self.layouts, strict=True)
        )
        private = []
        offset = 0
        for layout in self.layouts:
            amplitudes = {layout.lower_a_index, layout.upper_a_index}
            indices = np.full(layout.n_params, -1, dtype=np.intp)
            columns = [j for j in range(layout.n_params) if j not in amplitudes]
            indices[columns] = np.arange(offset, offset + len(columns))
            offset += len(columns)
            private.append(indices)
        shared = []
        for side in range(2):
            enabled = [indices[side] is not None for indices in self.physical_indices]
            if any(enabled) and not all(enabled):
                raise ValueError("physical boundary bases must agree across components")
            shared.append(offset if all(enabled) else None)
            if all(enabled):
                for indices, physical in zip(
                    private, self.physical_indices, strict=True
                ):
                    indices[physical[side]] = offset
                offset += 1
        self.local_indices = tuple(private)
        self.shared_parameter_indices = tuple(shared)
        self.n_params = offset
        self.expanded_indices = np.concatenate(self.local_indices)
        self.matrix = np.eye(offset)[self.expanded_indices]
        self._faces = {}

    def expand(self, params):
        """Expand one stable reduced vector without duplicating fitted variables.

        Parameters
        ----------
        params : numpy.ndarray, shape (n_params,)
            Finite joint shape/amplitude vector, without mixture logits.
        """
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,) or not np.all(np.isfinite(params)):
            raise ValueError("invalid reduced mixture parameter vector")
        return tuple(params[indices] for indices in self.local_indices)

    def pack(self, params):
        """Strictly pack an already shared iterate; never average unequal values.

        Parameters
        ----------
        params : sequence of numpy.ndarray
            One valid local vector per component. Shared physical amplitudes
            must agree exactly, including across reflected fitting coordinates.
        """
        if len(params) != len(self.layouts):
            raise ValueError("component parameter count does not match mixture")
        result = np.empty(self.n_params, dtype=np.float64)
        seen = np.zeros(self.n_params, dtype=bool)
        for layout, indices, values in zip(
            self.layouts, self.local_indices, params, strict=True
        ):
            values = layout.validate_params(values)
            repeated = seen[indices]
            if np.any(result[indices[repeated]] != values[repeated]):
                raise ValueError("component amplitudes do not share physical values")
            result[indices] = values
            seen[indices] = True
        return result

    def joint_matrix(self, n_logits=0):
        """Linear expansion including the unchanged trailing mixture logits.

        Parameters
        ----------
        n_logits : int, optional
            Number of trailing unconstrained mixture logits to preserve.
        """
        matrix = np.zeros(
            (len(self.expanded_indices) + n_logits, self.n_params + n_logits)
        )
        matrix[: len(self.expanded_indices), : self.n_params] = self.matrix
        if n_logits:
            matrix[-n_logits:, -n_logits:] = np.eye(n_logits)
        return matrix

    def face(self, degrees, active, n_logits=0):
        """Build an exact injection and the corresponding component cone product.

        Faces are immutable and memoized per map.

        Parameters
        ----------
        degrees : sequence of int
            Active curvature degree of each component.
        active : sequence of bool
            Whether the shared lower and upper physical amplitudes are free.
        n_logits : int, optional
            Number of trailing mixture logits included in the face.
        """
        key = (tuple(map(int, degrees)), tuple(map(bool, active)), int(n_logits))
        face = self._faces.get(key)
        if face is None:
            face = _MixtureFace(self, *key)
            self._faces[key] = face
        return face


@dataclass
class _MixtureFace:
    """Exact zero-coordinate face and its coupled component cone bookkeeping.

    Parameters
    ----------
    natural_map : _MixtureNaturalMap
        Joint coordinate map owning the component layouts and physical sides.
    degrees : tuple of int
        Curvature degree retained per component.
    active : tuple of bool
        Activity of the shared lower and upper physical boundary amplitudes.
    n_logits : int, optional
        Number of trailing unconstrained mixture logits.
    """

    natural_map: _MixtureNaturalMap
    degrees: tuple
    active: tuple
    n_logits: int = 0

    def __post_init__(self):
        """Derive the exact injection, cone rows and component dual mappings."""
        mapping = self.natural_map
        if len(self.degrees) != len(mapping.layouts) or len(self.active) != 2:
            raise ValueError("invalid mixture face")
        free = np.ones(mapping.n_params + self.n_logits, dtype=bool)
        representations = []
        for layout, indices, physical, degree in zip(
            mapping.layouts,
            mapping.local_indices,
            mapping.physical_indices,
            self.degrees,
            strict=True,
        ):
            free[
                indices[
                    layout.curvature_slice.start
                    + degree
                    + 1 : layout.curvature_slice.stop
                ]
            ] = False
            local_active = {
                index: bool(on)
                for index, on in zip(physical, self.active, strict=True)
                if index is not None
            }
            representations.append(
                _support_representation(
                    layout,
                    degree,
                    local_active.get(layout.lower_a_index, False),
                    local_active.get(layout.upper_a_index, False),
                )
            )
        for index, active in zip(
            mapping.shared_parameter_indices, self.active, strict=True
        ):
            if index is None:
                if active:
                    raise ValueError("cannot activate an excluded physical boundary")
            elif not active:
                free[index] = False
        self.free_indices = np.flatnonzero(free)
        self.injection = np.eye(free.size)[:, self.free_indices]
        # Free-face column of every component coordinate (-1: fixed at zero)
        # and of every logit, for the compiled mixture objectives.
        position = np.full(free.size, -1, dtype=np.intc)
        position[self.free_indices] = np.arange(self.free_indices.size)
        self.component_columns = tuple(
            position[indices] for indices in mapping.local_indices
        )
        self.logit_columns = position[mapping.n_params :]
        self.component_representations = tuple(representations)
        blocks = []
        rows = []
        duals = []
        row_degrees = []
        self.component_rows = []
        self.block_slices = []
        offset = 0
        for indices, rep in zip(mapping.local_indices, representations, strict=True):
            b = np.zeros((rep.n_rows, free.size))
            b[:, indices] = rep.b_matrix
            b = b[:, self.free_indices]
            keep = np.any(b != 0.0, axis=1)
            for matrices in rep.row_matrices:
                keep |= np.any(matrices != 0.0, axis=(1, 2))
            self.component_rows.append(np.flatnonzero(keep))
            rows.append(b[keep])
            duals.append(rep.reference_dual[keep])
            row_degrees.append(rep.row_degrees[keep])
            local_blocks = tuple(matrices[keep] for matrices in rep.row_matrices)
            self.block_slices.append(slice(offset, offset + len(local_blocks)))
            offset += len(local_blocks)
            blocks.append(local_blocks)
        self.row_offsets = np.concatenate(([0], np.cumsum([len(row) for row in rows])))
        matrices = []
        for k, local_blocks in enumerate(blocks):
            for block in local_blocks:
                padded = np.zeros((self.row_offsets[-1],) + block.shape[1:])
                padded[self.row_offsets[k] : self.row_offsets[k + 1]] = block
                matrices.append(padded)
        self.representation = _ConicRepresentation(
            np.vstack(rows),
            tuple(matrices),
            np.concatenate(duals),
            np.concatenate(row_degrees),
        )

    def expand(self, params):
        """Inject free parameters; absent coordinates are exactly zero.

        Parameters
        ----------
        params : numpy.ndarray, shape (len(free_indices),)
            Reduced parameters on this exact face, including free logits.
        """
        values = np.asarray(params, dtype=np.float64)
        if values.shape != (len(self.free_indices),):
            raise ValueError("invalid free face vector")
        result = np.zeros(self.injection.shape[0], dtype=np.float64)
        result[self.free_indices] = values
        return result

    def restrict(self, params):
        """Restrict a full vector for a proposed exact face.

        Parameters
        ----------
        params : numpy.ndarray
            Full joint shape/amplitude/logit vector in the map's ordering.
        """
        return np.asarray(params, dtype=np.float64)[self.free_indices].copy()

    def component_dual(self, dual, component):
        """Restore eliminated zero equality multipliers for component bookkeeping.

        Parameters
        ----------
        dual : numpy.ndarray
            Joint equality multipliers in the face's concatenated row ordering.
        component : int
            Component whose full local multiplier vector is requested.
        """
        result = np.zeros(self.component_representations[component].n_rows)
        result[self.component_rows[component]] = dual[
            self.row_offsets[component] : self.row_offsets[component + 1]
        ]
        return result
