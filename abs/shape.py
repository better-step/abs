from abs.topology import *
from abs.curve import *
from abs.surface import *
from abs import sampler
from abs.winding_number import find_surface_uv_for_curve

def _get_edges(edge_data):
    return Edge(edge_data)

def _get_faces(face_data):
    return Face(face_data)

def _get_halfedges(halfedge_data):
    return Halfedge(halfedge_data)

def _get_loops(loop_data):
    return Loop(loop_data)



class Shape:
    def __init__(self, geometry_data, topology_data, spacing=0.02):

        self._geometry_data = self._geometry_data(geometry_data)
        self._topology_data = self._topology_data(topology_data)
        spacing *= np.linalg.norm(self._geometry_data.bbox[0][1] - self._geometry_data.bbox[0][0])
        self.bbox = self._geometry_data.bbox
        self.vertices = self._geometry_data.vertices

        self._create_2d_trimming_curves(self._geometry_data.curves2d, self._geometry_data.curves3d, spacing)
        self.Solid = self.Solid(self._topology_data, self._geometry_data, self.trimming_curves_2d)


    def _create_2d_trimming_curves(self, curves_2d, curves_3d, spacing):
        """
        Create 2D trimming curves.
        """
        self.trimming_curves_2d = []
        for shell in self._topology_data.shells:
            self._process_2d_trimming_curves_for_shell(shell, curves_2d, curves_3d, spacing)


    def _process_2d_trimming_curves_for_shell(self, shell_index, curves2d, curves3d, spacing):

        if isinstance(shell_index, Shell):
            shell = shell_index
        else:
            shell = self._topology_data.shells[shell_index]

        for (face_index, _) in shell.faces:
            face = self._topology_data.faces[face_index]
            self.trimming_curves_2d += (face_index-len(self.trimming_curves_2d)+1)* [None]
            self.trimming_curves_2d[face_index] = []

            surface_index = face.surface
            surface = self._geometry_data.surfaces[surface_index]

            for loop_id in face.loops:
                loop = self._topology_data.loops[loop_id]
                for halfedge_index in loop.halfedges:
                    halfedge = self._topology_data.halfedges[halfedge_index]
                    modified_orientation = halfedge.orientation_wrt_edge
                    if not face.surface_orientation:
                        modified_orientation = not modified_orientation

                    curve3d_index = halfedge.edge
                    curve3d = curves3d[curve3d_index]

                    if hasattr(halfedge, 'curve2d'):
                        curve2d_index = halfedge.curve2d
                        current_curve = curves2d[curve2d_index]
                    elif curve3d is not None:
                        current_curve = curve3d
                    else:
                        continue

                    if current_curve.shape_name == 'Line':
                        n_samples = 2
                    else:
                        length = current_curve.get_length()
                        n_samples = int(length / spacing)

                    if hasattr(halfedge, 'curve2d'):
                        curve2d_index = halfedge.curve2d
                        curve2d = curves2d[curve2d_index]
                        _, closest_surface_uv_values_of_curve = sampler.uniform_sample(curve2d, n_samples, 4, 100)
                        if not modified_orientation:
                            closest_surface_uv_values_of_curve = closest_surface_uv_values_of_curve[::-1]
                    else:
                        surface_uv_values, surface_points = sampler.uniform_sample(surface, n_samples*n_samples, 5, 100)


                        # Sample the curve points to get UV values
                        _, curve_points = sampler.uniform_sample(curve3d, n_samples)

                        if not modified_orientation:
                            curve_points = curve_points[::-1]
                        # Calculate the nearest UV values on the surface for the curve points
                        closest_surface_uv_values_of_curve = find_surface_uv_for_curve(surface_points, surface_uv_values, curve_points)

                    self.trimming_curves_2d[face_index].append(closest_surface_uv_values_of_curve)




    class _geometry_data:
        def __init__(self, geometry_data):
            self.curves2d, self.curves3d, self.surfaces, self.bbox, self.vertices = [], [], [], [], []
            self.__init_geometry(geometry_data)

        def __init_geometry(self, data):
            tmp = data.get('2dcurves', {}).values()
            self.curves2d=len(tmp)*[None]
            for curve_data in tmp:
                index, curve = create_curve(curve_data)
                self.curves2d[index] = curve

            tmp = data.get('3dcurves', {}).values()
            self.curves3d=len(tmp)*[None]
            for curve_data in tmp:
                index, curve = create_curve(curve_data)
                self.curves3d[index] = curve

            tmp = data.get('surfaces', {}).values()
            self.surfaces=len(tmp)*[None]
            for surface_data in tmp:
                index, surface = create_surface(surface_data)
                self.surfaces[index] = surface

            self.bbox.append(np.array(data.get('bbox')[:]))

            self.vertices.append(np.array(data.get('vertices')))


    class _topology_data:
        def __init__(self, topology_data):
            self.edges, self.faces, self.halfedges, self.loops, self.shells, self.solids = [], [], [], [], [], []
            self.__init_topology(topology_data)

        def __init_topology(self, data):

            entity_map = {
                'edges': (self.edges, _get_edges),
                'faces': (self.faces, _get_faces),
                'halfedges': (self.halfedges, _get_halfedges),
                'loops': (self.loops, _get_loops),
                'shells': (self.shells, Shell),
                'solids': (self.solids, Solid)
            }

            for entity, (attr_list, constructor) in entity_map.items():
                entity_data = Topology._get_topo_data(data, entity)
                attr_list.extend(constructor(item) for item in entity_data)


    class Solid:
        def __init__(self, topology, geometry, trimming_curves):
            self.edges, self.faces, self.halfedges, self.loops, self.shells, self.solids = [], [], [], [], [], []
            self.__init_solid(topology, geometry, trimming_curves)


        def __init_solid(self, topo, geo, trimming_curves):

            # Loop over edges
            for edge in topo.edges:
                edge.curve3d = geo.curves3d[edge.curve3d]
                self.edges.append(edge)

            # loop over halfedges
            for halfedge in topo.halfedges:
                halfedge.curve2d = geo.curves2d[halfedge.curve2d] if halfedge.curve2d < len(geo.curves2d) else None
                halfedge.edge = self.edges[halfedge.edge]
                if halfedge.mates:
                    halfedge.mates = topo.halfedges[int(halfedge.mates[0])]
                self.halfedges.append(halfedge)

            # Loop over loops
            for loop in topo.loops:
                loop.halfedges = [topo.halfedges[halfedge_id] for halfedge_id in loop.halfedges]
                self.loops.append(loop)

            # Loop over faces
            for idx, face in enumerate(topo.faces):
                face.surface = geo.surfaces[face.surface]
                face.loops = [topo.loops[loop_id] for loop_id in face.loops]
                face.trimming_curves_2d = trimming_curves[idx]
                self.faces.append(face)

            # Loop over shells
            for shell in topo.shells:
                for idx, orientation in enumerate(shell.faces):
                    shell.faces[idx] = (topo.faces[orientation[0]] , orientation[1])
                self.shells.append(shell)

            # loop over solids
            for solid in topo.solids:
                solid.shells = [topo.shells[shell_id] for shell_id in solid.shells]
                self.solids.append(solid)

            # adding the reverse mapping

            # from edges to halfedges
            edgeMap = {}
            for halfEdge in self.halfedges:
                edge = halfEdge.edge
                edgeMapValue = edgeMap.get(edge, {'halfedges': []})
                edgeMapValue['halfedges'].append(halfEdge)
                edgeMap[edge] = edgeMapValue

            for edge in edgeMap:
                edge._halfedges = edgeMap[edge]['halfedges']

            # from halfedges to loops
            halfEdgeMap = {}
            for loop in self.loops:
                for halfedge in loop.halfedges:
                    halfEdgeMapValue = halfEdgeMap.get(halfedge, {'loops': []})
                    halfEdgeMapValue['loops'].append(loop)
                    halfEdgeMap[halfedge] = halfEdgeMapValue

            for halfedge in halfEdgeMap:
                halfedge.loops = halfEdgeMap[halfedge]['loops']

            # from loops to faces
            loopMap = {}
            for face in self.faces:
                for loop in face.loops:
                    loopMapValue = loopMap.get(loop, {'faces': []})
                    loopMapValue['faces'].append(face)
                    loopMap[loop] = loopMapValue

            for loop in loopMap:
                loop.faces = loopMap[loop]['faces']

            # from faces to shells
            faceMap = {}
            for shell in self.shells:
                for face, _ in shell.faces:
                    faceMapValue = faceMap.get(face, {'shells': []})
                    faceMapValue['shells'].append(shell)
                    faceMap[face] = faceMapValue

            for face in faceMap:
                face.shells = faceMap[face]['shells']

            # from shells to solids
            shellMap = {}
            if len(self.solids) > 0:
                for solid in self.solids:
                    for shell in solid.shells:
                        shellMapValue = shellMap.get(shell, {'solids': []})
                        shellMapValue['solids'].append(solid)
                        shellMap[shell] = shellMapValue

                for shell in shellMap:
                    shell.solids = shellMap[shell]['solids']





