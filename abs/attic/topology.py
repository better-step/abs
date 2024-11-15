import numpy as np
import h5py


class Topology:
    def __init__(self, part_data):
        self._solids = None
        self._edges = None
        self._faces = None
        self._loops = None
        self._halfedges = None
        self._shells = None
        if not part_data:
            raise ValueError("Valid part data must be provided")
        self._part_data = part_data
        self.process_data()

    def process_data(self):

        self._edges = self._parse_hdf5_data("edges")
        self._faces = self._parse_hdf5_data("faces")
        self._halfedges = self._parse_hdf5_data("halfedges")
        self._loops = self._parse_hdf5_data("loops")
        self._shells = self._parse_hdf5_data("shells")
        self._solids = self._parse_hdf5_data("solids")

        self._create_mappings()

    def _create_mappings(self):
        self.halfEdgeMap = self._create_halfEdge_map()
        self.edgeMap = self._create_edge_map()
        self.loopMap = self._create_loop_map()
        self.faceMap = self._create_face_map()

    def _create_halfEdge_map(self):
        """
        Create a mapping from half-edges to loops.
        """
        halfEdgeMap = {}
        for loop_index, loop in enumerate(self.loops):
            for halfedge_id in loop['halfedges']:
                halfEdgeMapValue = halfEdgeMap.get(halfedge_id, {'loops': set()})
                halfEdgeMapValue['loops'].add(loop_index)
                halfEdgeMap[halfedge_id] = halfEdgeMapValue
        return halfEdgeMap

    def _create_edge_map(self):
        """
        Create a mapping from edges to half-edges.
        """
        edgeMap = {}
        for halfEdge_index, halfEdge in enumerate(self.halfedges):
            edge_id = halfEdge['edge']
            edgeMapValue = edgeMap.get(edge_id, {'halfedges': set()})
            edgeMapValue['halfedges'].add(halfEdge_index)
            edgeMap[edge_id] = edgeMapValue
        return edgeMap

    def _create_loop_map(self):
        """
        Create a mapping from loops to faces.
        """
        loopMap = {}
        for face_index, face in enumerate(self.faces):
            for loop_id in face['loops']:
                loopMapValue = loopMap.get(loop_id, {'faces': set()})
                loopMapValue['faces'].add(face_index)
                loopMap[loop_id] = loopMapValue
        return loopMap

    def _create_face_map(self):
        """
        Create a mapping from faces to shells.
        """
        faceMap = {}
        for shell_index, shell in enumerate(self.shells):
            for face_info in shell['faces']:
                face_id = face_info['face_index']
                faceMapValue = faceMap.get(face_id, {'shells': set()})
                faceMapValue['shells'].add(shell_index)
                faceMap[face_id] = faceMapValue
        return faceMap

    def determine_curve_orientation(self, face, halfedge):
        """
        Determine the orientation of a curve relative to a surface.

        Args:
            face (dict): The face information.
            halfedge (dict): The halfedge information.

        Returns:
            bool: The modified orientation of the curve.
        """
        orientation_wrt_edge = self._halfedges[halfedge]['orientation_wrt_edge']
        if not self._faces[face]['surface_orientation']:
            orientation_wrt_edge = not orientation_wrt_edge
        return orientation_wrt_edge



    @property
    def edges(self):

        if self._edges is None:
            self.process_data()

        return self._edges

    @property
    def faces(self):

        if self._faces is None:
            self.process_data()

        return self._faces

    @property
    def halfedges(self):

        if self._halfedges is None:
            self.process_data()

        return self._halfedges

    @property
    def loops(self):

        if self._loops is None:
            self.process_data()

        return self._loops

    @property
    def shells(self):

        if self._shells is None:
            self.process_data()

        return self._shells

    @property
    def solids(self):

        if self._solids is None:
            self.process_data()

        return self._solids

    def _parse_hdf5_data(self, entity):
        return self._hdf5_to_dict(self._part_data, entity)

    def _hdf5_to_dict(self, hdf5_file, entity):
        def recursive_hdf5_to_dict(group):
            data_dict = {}
            for key, item in group.items():
                if isinstance(item, h5py.Group):
                    data_dict[key] = recursive_hdf5_to_dict(item)
                else:
                    data_dict[key] = item[()]  # Convert dataset to a numpy array
            return data_dict

        data_dict = [(int(k), recursive_hdf5_to_dict(hdf5_file[entity][k])) for k in hdf5_file[entity].keys()]
        data_dict.sort(key=lambda x: x[0])
        return [v[1] for v in data_dict]




    # Advance topological queries
    def find_adjacent_faces(self, face_index):
        """
        Find faces adjacent to a given face.

        Args:
        face_index (int): Index of the face.

        Returns:
        list: List of indices of adjacent faces.
        """
        adjacent_faces = set()
        for loop_id in self.faces[face_index]['loops']:
            for halfedge_id in self.loops[loop_id]['halfedges']:
                edge_id = self.halfedges[halfedge_id]['edge']
                for he in self.edgeMap[edge_id]['halfedges']:
                    loop_ids = self.halfEdgeMap[he]['loops']
                    for loop in loop_ids:
                        adjacent_faces.update(self.loopMap[loop]['faces'])
        adjacent_faces.discard(face_index)  # Remove the original face
        return list(adjacent_faces)

    def find_connected_components(self):
        """
        Identify connected components in the topology.
        """
        visited = set()
        components = []

        def dfs(face_index):
            if face_index in visited:
                return
            visited.add(face_index)
            component.add(face_index)
            for adjacent_face in self.find_adjacent_faces(face_index):
                dfs(adjacent_face)

        for face_index in range(len(self.faces)):
            if face_index not in visited:
                component = set()
                dfs(face_index)
                components.append(component)

        return [list(component) for component in components]
