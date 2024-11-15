from abs.topology import *
from abs.curve import *
from abs.surface import *
from abs import sampler
from abs.winding_number import winding_number, find_surface_uv_for_curve


def _create_surface(surface_data):
    surface_type = surface_data.get('type')[()].decode('utf-8')
    surface_map = {
        'Plane': Plane,
        'Cylinder': Cylinder,
        'Cone': Cone,
        'Sphere': Sphere,
        'Torus': Torus,
        'BSpline': BSplineSurface
    }
    surface_class = surface_map.get(surface_type)
    if surface_class:
        return surface_class(surface_data)
    else:
        print(f"This surface type: {surface_type}, is currently not supported")
        return None


def _create_curve(curve_data):
    curve_type = curve_data.get('type')[()].decode('utf-8')

    curve_map = {
        'Line': Line,
        'Circle': Circle,
        'Ellipse': Ellipse,
        'BSpline': BSplineCurve
    }
    curve_class = curve_map.get(curve_type)
    if curve_class:
        return curve_class(curve_data)
    else:
        print(f"This curve type: {curve_type}, is currently not supported")
        return None

def _get_edges(edge_data):
    return Edge(edge_data)

def _get_faces(face_data):
    return Face(face_data)

def _get_halfedges(halfedge_data):
    return Halfedge(halfedge_data)

def _get_loops(loop_data):
    return Loop(loop_data)



class Shape:
    def __init__(self, geometry_data, topology_data, spacing=0.1):

        self._geometry_data = self._geometry_data(geometry_data)
        self._topology_data = self._topology_data(topology_data)
        self._create_2d_trimming_curves(self._geometry_data._curves2d, self._geometry_data._curves3d, spacing)
        self.Solid = self.Solid(self._topology_data, self._geometry_data, self._2d_trimming_curves)


    def _create_2d_trimming_curves(self, curves2d, curves3d, spacing):
        """
        Create 2D trimming curves.
        """
        self._2d_trimming_curves = []
        if len(self._topology_data._solids) == 0:
            for shell in self._topology_data._shells:
                self._process_2d_trimming_curves_for_shell(shell, curves2d, curves3d, spacing)
        else:
            for solid in self._topology_data._solids:
                for shell_index in solid._shells:
                    self._process_2d_trimming_curves_for_shell(shell_index, curves2d, curves3d, spacing)


    def _process_2d_trimming_curves_for_shell(self, shell_index, curves2d, curves3d, spacing):

        if isinstance(shell_index, Shell):
            shell = shell_index
        else:
            shell = self._topology_data._shells[shell_index]

        for (face_index, _) in shell._faces:
            face = self._topology_data._faces[face_index]
            self._2d_trimming_curves += (face_index-len(self._2d_trimming_curves)+1)* [None]
            self._2d_trimming_curves[face_index] = []

            surface_index = face._surface
            surface = self._geometry_data._surfaces[surface_index]

            for loop_id in face._loops:
                loop = self._topology_data._loops[loop_id]
                for halfedge_index in loop._halfedges:
                    halfedge = self._topology_data._halfedges[halfedge_index]
                    modified_orientation = halfedge._orientation_wrt_edge
                    if not face._surface_orientation:
                        modified_orientation = not modified_orientation

                    curve3d_index = halfedge._edge
                    curve3d = curves3d[curve3d_index]

                    if hasattr(halfedge, '_2dcurve'):
                        curve2d_index = halfedge._2dcurve
                        current_curve = curves2d[curve2d_index]
                    elif curve3d is not None:
                        current_curve = curve3d
                    else:
                        continue

                    if current_curve._shape_name == 'Line':
                        n_samples = 2
                    else:
                        length = current_curve.length()
                        n_samples = int(length / spacing)

                    if hasattr(halfedge, '_2dcurve'):
                        curve2d_index = halfedge._2dcurve
                        curve2d = curves2d[curve2d_index]
                        _, closest_surface_uv_values_of_curve = sampler.uniform_sample(curve2d, n_samples, 4, 300)
                        if not modified_orientation:
                            closest_surface_uv_values_of_curve = closest_surface_uv_values_of_curve[::-1]
                    else:
                        surface_uv_values, surface_points = sampler.uniform_sample(surface, n_samples*n_samples, 5, 300)


                        # Sample the curve points to get UV values
                        _, curve_points = sampler.uniform_sample(curve3d, n_samples)

                        if not modified_orientation:
                            curve_points = curve_points[::-1]
                        # Calculate the nearest UV values on the surface for the curve points
                        closest_surface_uv_values_of_curve = find_surface_uv_for_curve(surface_points, surface_uv_values, curve_points)

                    self._2d_trimming_curves[face_index].append(closest_surface_uv_values_of_curve)




    class _geometry_data:
        def __init__(self, geometry_data):
            self._curves2d, self._curves3d, self._surfaces, self._bbox = [], [], [], []
            self.__init_geometry(geometry_data)

        def __init_geometry(self, data):
            for curve_data in data.get('2dcurves', {}).values():
                curve = _create_curve(curve_data)
                self._curves2d.append(curve)

            for curve_data in data.get('3dcurves', {}).values():
                curve = _create_curve(curve_data)
                self._curves3d.append(curve)

            for surface_data in data.get('surfaces', {}).values():
                surface = _create_surface(surface_data)
                self._surfaces.append(surface)

            self._bbox.append(np.array(data.get('bbox')[:]))


    class _topology_data:
        def __init__(self, topology_data):
            self._edges, self._faces, self._halfedges, self._loops, self._shells, self._solids = [], [], [], [], [], []
            self.__init_topology(topology_data)

        def __init_topology(self, data):

            entity_map = {
                'edges': (self._edges, _get_edges),
                'faces': (self._faces, _get_faces),
                'halfedges': (self._halfedges, _get_halfedges),
                'loops': (self._loops, _get_loops),
                'shells': (self._shells, Shell),
                'solids': (self._solids, Solid)
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
            for edge in topo._edges:
                edge._3dcurve = geo._curves3d[edge._3dcurve]
                self.edges.append(edge)

            # loop over halfedges
            for halfedge in topo._halfedges:
                halfedge._2dcurve = geo._curves2d[halfedge._2dcurve]
                halfedge._edge = self.edges[halfedge._edge]
                self.halfedges.append(halfedge)

            # Loop over loops
            for loop in topo._loops:
                loop._halfedges = [topo._halfedges[halfedge_id] for halfedge_id in loop._halfedges]
                self.loops.append(loop)

            # Loop over faces
            for idx, face in enumerate(topo._faces):
                face._surface = geo._surfaces[face._surface]
                face._loops = [topo._loops[loop_id] for loop_id in face._loops]
                face._2d_trimming_curves = trimming_curves[idx]
                self.faces.append(face)

            # Loop over shells
            for shell in topo._shells:
                for orientation in shell._faces:
                    shell._faces[orientation[0]] = (topo._faces[orientation[0]] , orientation[1])
                self.shells.append(shell)

            # loop over solids
            for solid in topo._solids:
                solid._shells = [topo._shells[shell_id] for shell_id in solid._shells]
                self.solids.append(solid)

            # adding the reverse mapping

            # from edges to halfedges
            edgeMap = {}
            for halfEdge in self.halfedges:
                edge = halfEdge._edge
                edgeMapValue = edgeMap.get(edge, {'halfedges': []})
                edgeMapValue['halfedges'].append(halfEdge)
                edgeMap[edge] = edgeMapValue

            for edge in edgeMap:
                edge._halfedges = edgeMap[edge]['halfedges']

            # from halfedges to loops
            halfEdgeMap = {}
            for loop in self.loops:
                for halfedge in loop._halfedges:
                    halfEdgeMapValue = halfEdgeMap.get(halfedge, {'loops': []})
                    halfEdgeMapValue['loops'].append(loop)
                    halfEdgeMap[halfedge] = halfEdgeMapValue

            for halfedge in halfEdgeMap:
                halfedge._loops = halfEdgeMap[halfedge]['loops']

            # from loops to faces
            loopMap = {}
            for face in self.faces:
                for loop in face._loops:
                    loopMapValue = loopMap.get(loop, {'faces': []})
                    loopMapValue['faces'].append(face)
                    loopMap[loop] = loopMapValue

            for loop in loopMap:
                loop._faces = loopMap[loop]['faces']

            # from faces to shells
            faceMap = {}
            for shell in self.shells:
                for face, _ in shell._faces:
                    faceMapValue = faceMap.get(face, {'shells': []})
                    faceMapValue['shells'].append(shell)
                    faceMap[face] = faceMapValue

            for face in faceMap:
                face._shells = faceMap[face]['shells']

            # from shells to solids
            shellMap = {}
            if len(self.solids) > 0:
                for solid in self.solids:
                    for shell in solid._shells:
                        shellMapValue = shellMap.get(shell, {'solids': []})
                        shellMapValue['solids'].append(solid)
                        shellMap[shell] = shellMapValue

                for shell in shellMap:
                    shell._solids = shellMap[shell]['solids']





