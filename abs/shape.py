"""
Combines geometry and topology to form complete shapes (parts).
Provides the Shape class that assembles curves, surfaces, and topology into a structured object.
"""
from . import sampler
from .topology import Edge, Face, Halfedge, Loop, Shell, Solid as TopoSolid, Topology
from .curve import create_curve
from .surface import create_surface
from .winding_number import find_surface_uv_for_curve
import numpy as np
# from joblib import Parallel, delayed







class Shape:
    """Represents a geometric part (one part from the CAD model) with geometry and topology assembled."""
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

        def __init_geometry(self, data, version="3.0"):
            if version == "3.0":
                curve2d_index = data.get('2dcurves_index', {})[()]
                curve2d_data = data.get('2dcurves', {})[()]
                self.curves2d = [None] * (len(curve2d_index)-1)

                for i in range(len(curve2d_index)-1):
                    tmp = curve2d_data[curve2d_index[i]:curve2d_index[i+1]]
                    ctype = int(tmp[0])
                    interval = tmp[1:3]

                    if ctype == 0:  # Line
                        curve_data = {'id': i,
                                      'type': 'Line',
                                      'interval': interval,
                                      'location': tmp[3:5],
                                      'direction': tmp[5:7]}
                    elif ctype == 1:  # Circle
                        curve_data = {'id': i,
                                        'type': 'Circle',
                                        'interval': interval,
                                        'location': tmp[3:5],
                                        'x_axis': tmp[5:7],
                                        'y_axis': tmp[7:9],
                                        'radius': tmp[9]}
                    elif ctype == 2:  # Ellipse
                        curve_data = {'id': i,
                                        'type': 'Ellipse',
                                        'interval': interval,
                                        'focus1': tmp[3:5],
                                        'focus2': tmp[5:7],
                                        'x_axis': tmp[7:9],
                                        'y_axis': tmp[9:11],
                                        'maj_radius': tmp[11],
                                        'min_radius': tmp[12]}
                    elif ctype == 3:  # BSplineCurve
                        degree = int(tmp[3])
                        continuity = int(tmp[4])
                        rational = bool(tmp[5])
                        periodic = bool(tmp[6])
                        closed = bool(tmp[7])
                        len_poles = int(tmp[8])
                        pshape = (int(tmp[9]), int(tmp[10]))
                        poles = np.array(tmp[11:11+len_poles]).reshape(pshape)
                        idx = 11 + len_poles
                        len_knots = int(tmp[idx])
                        knots = tmp[idx+1:idx+1+len_knots]
                        idx = idx + 1 + len_knots
                        len_weights = int(tmp[idx])
                        weights = tmp[idx+1:idx+1+len_weights]

                        curve_data = {'id': i,
                                        'type': 'BSpline',
                                        'interval': interval,
                                        'degree': degree,
                                        'continuity': continuity,
                                        'rational': rational,
                                        'periodic': periodic,
                                        'closed': closed,
                                        'poles': poles,
                                        'knots': knots,
                                        'weights': weights}
                    elif ctype == 4:  # Other
                        curve_data = {'id': i,
                                        'type': 'Other',
                                        'interval': interval}
                    else:
                        raise ValueError(f"Unknown curve type: {ctype}")

                    self.curves2d[i] = create_curve(curve_data, False)[1]
                # Parallel(n_jobs=-1, backend="threading")(delayed(process_curve2d)(i) for i in range(len(curve2d_index)-1))
                # for i in range(len(curve2d_index)-1):
                #     process_curve2d(i)

                curve3d_index = data.get('3dcurves_index', {})[()]
                curve3d_data = data.get('3dcurves', {})[()]
                self.curves3d = [None] * (len(curve3d_index)-1)
                for i in range(len(curve3d_index)-1):
                    tmp = curve3d_data[curve3d_index[i]:curve3d_index[i+1]]
                    ctype = int(tmp[0])
                    interval = tmp[1:3]
                    transform = tmp[len(tmp)-12:]  # last 16 values are the transformation matrix
                    transform = np.array(transform).reshape((3,4))

                    if ctype == 0:  # Line
                        curve_data = {'id': i,
                                      'type': 'Line',
                                      'interval': interval,
                                      'location': tmp[3:6],
                                      'direction': tmp[6:9]}
                    elif ctype == 1:  # Circle
                        curve_data = {'id': i,
                                        'type': 'Circle',
                                        'interval': interval,
                                        'location': tmp[3:6],
                                        'x_axis': tmp[6:9],
                                        'y_axis': tmp[9:12],
                                        'z_axis': tmp[12:15],
                                        'radius': tmp[15]}
                    elif ctype == 2:  # Ellipse
                        curve_data = {'id': i,
                                        'type': 'Ellipse',
                                        'interval': interval,
                                        'focus1': tmp[3:6],
                                        'focus2': tmp[6:9],
                                        'x_axis': tmp[9:12],
                                        'y_axis': tmp[12:15],
                                        'z_axis': tmp[15:18],
                                        'maj_radius': tmp[18],
                                        'min_radius': tmp[19]}
                    elif ctype == 3:  # BSplineCurve
                        degree = int(tmp[3])
                        continuity = int(tmp[4])
                        rational = bool(tmp[5])
                        periodic = bool(tmp[6])
                        closed = bool(tmp[7])
                        len_poles = int(tmp[8])
                        pshape = (int(tmp[9]), int(tmp[10]))
                        poles = np.array(tmp[11:11+len_poles]).reshape(pshape)
                        idx = 11 + len_poles
                        len_knots = int(tmp[idx])
                        knots = tmp[idx+1:idx+1+len_knots]
                        idx = idx + 1 + len_knots
                        len_weights = int(tmp[idx])
                        weights = tmp[idx+1:idx+1+len_weights]

                        curve_data = {'id': i,
                                        'type': 'BSpline',
                                        'interval': interval,
                                        'degree': degree,
                                        'continuity': continuity,
                                        'rational': rational,
                                        'periodic': periodic,
                                        'closed': closed,
                                        'poles': poles,
                                        'knots': knots,
                                        'weights': weights}
                    elif ctype == 4:  # Other
                        curve_data = {'id': i,
                                        'type': 'Other',
                                        'interval': interval}
                    else:
                        raise ValueError(f"Unknown curve type: {ctype} for curve {i}")

                    self.curves3d[i] = create_curve(curve_data, False)[1]
                # Parallel(n_jobs=-1, backend="threading")(delayed(process_curve3d)(i) for i in range(len(curve3d_index)-1))
                # for i in range(len(curve3d_index)-1):
                #     process_curve3d(i)

                # TODO
                tmp = data.get('surfaces', {}).values()
                self.surfaces=len(tmp)*[None]
                for surface_data in tmp:
                    index, surface = create_surface(surface_data)
                    self.surfaces[index] = surface
            else:
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

        def __init_topology(self, data, version="3.0"):
            if version == "3.0":
                # Edges
                edge_index = data.get('edge_index', {})[()]
                edge_data = data.get('edges', {})[()]
                self.edges = [None] * (len(edge_index)-1)

                for i in range(len(edge_index)-1):
                    tmp = edge_data[edge_index[i]:edge_index[i+1]]
                    local_edge = {'id': i, '3dcurve': tmp[0], 'start_vertex': tmp[1], 'end_vertex': tmp[2]}
                    self.edges[i] = Edge(local_edge)
                # Parallel(n_jobs=-1, backend="threading")(delayed(process_edge)(i) for i in range(len(edge_index)-1))

                # Halfedges
                halfedge_index = data.get('halfedge_index', {})[()]
                halfedge_data = data.get('halfedges', {})[()]
                self.halfedges = [None]* (len(halfedge_index)-1)

                for i in range(len(halfedge_index)-1):
                    tmp = halfedge_data[halfedge_index[i]:halfedge_index[i+1]]
                    mates = tmp[3:] if len(tmp) > 3 else []
                    local_halfedge = {'id': i, '2dcurve': tmp[0], 'edge': tmp[1], 'orientation_wrt_edge': bool(tmp[2]), 'mates': mates}
                    self.halfedges[i] = Halfedge(local_halfedge)
                # Parallel(n_jobs=-1, backend="threading")(delayed(process_halfedge)(i) for i in range(len(halfedge_index)-1))


                # Loops
                loop_index = data.get('loop_index', {})[()]
                loop_data = data.get('loops', {})[()]
                self.loops = [None] * (len(loop_index)-1)
                for i in range(len(loop_index)-1):
                    tmp = loop_data[loop_index[i]:loop_index[i+1]]
                    local_loop = {'id': i, 'halfedges': tmp}
                    self.loops[i] = Loop(local_loop)
                # Parallel(n_jobs=-1, backend="threading")(delayed(process_loop)(i) for i in range(len(loop_index)-1))

                # Shells
                shell_index = data.get('shell_index', {})[()]
                shell_data = data.get('shells', {})[()]
                self.shells = [None] * (len(shell_index)-1)
                for i in range(len(shell_index)-1):
                    tmp = shell_data[shell_index[i]:shell_index[i+1]]
                    faces = []
                    for j in range(1, len(tmp)-1, 2):
                        faces.append( (tmp[j], bool(tmp[j+1])) )
                    local_shell = {'id': i, 'orientation_wrt_solid': tmp[0], 'faces': faces}
                    self.shells[i] = Shell(local_shell)
                # Parallel(n_jobs=-1, backend="threading")(delayed(process_shell)(i) for i in range(len(shell_index)-1))


                # Solids
                solid_index = data.get('solid_index', {})[()]
                solid_data = data.get('solids', {})[()]
                self.solids = [None] * (len(solid_index)-1)
                for i in range(len(solid_index)-1):
                    tmp = solid_data[solid_index[i]:solid_index[i+1]]
                    local_solid = {'id': i, 'shells': tmp}
                    self.solids[i] = TopoSolid(local_solid)
                # Parallel(n_jobs=-1, backend="threading")(delayed(process_solid)(i) for i in range(len(solid_index)-1))

                # Faces
                face_index = data.get('face_index', {})[()]
                face_data = data.get('faces', {})[()]
                self.faces = [None] * (len(face_index)-1)
                for i in range(len(face_index)-1):
                    tmp = face_data[face_index[i]:face_index[i+1]]

                    exact_domain = tmp[:4]
                    has_singularities = bool(tmp[4])
                    nr_singularities = tmp[5]
                    # singularities = tmp[6:6+nr_singularities]
                    outer_loop = tmp[6]
                    surface = tmp[7]
                    surface_orientation = bool(tmp[8])
                    loops = np.array(tmp[9:]).astype(np.int64)


                    local_face = {'id': i,
                                    'exact_domain': exact_domain,
                                    'has_singularities': has_singularities,
                                    'nr_singularities': nr_singularities,
                                    'outer_loop': outer_loop,
                                'surface': surface,
                                'singularities': [],
                                'surface_orientation': surface_orientation,
                                'loops': loops}
                    self.faces[i] = Face(local_face)
                # Parallel(n_jobs=-1, backend="threading")(delayed(process_face)(i) for i in range(len(face_index)-1))
            else:
                entity_map = {
                    'edges': (self.edges, _get_edges),
                    'faces': (self.faces, _get_faces),
                    'halfedges': (self.halfedges, _get_halfedges),
                    'loops': (self.loops, _get_loops),
                    'shells': (self.shells, Shell),
                    'solids': (self.solids, TopoSolid)
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





# Helper functions to construct topology objects
def _get_edges(edge_data): return Edge(edge_data)
def _get_faces(face_data): return Face(face_data)
def _get_halfedges(halfedge_data): return Halfedge(halfedge_data)
def _get_loops(loop_data): return Loop(loop_data)
