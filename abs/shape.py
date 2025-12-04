"""
Combines geometry and topology to form complete shapes (parts).
Provides the Shape class that assembles curves, surfaces, and topology into a structured object.
"""
from . import sampler
from .topology import Edge, Face, Halfedge, Loop, Shell, Solid as TopoSolid, Topology
from .curve import Circle, Ellipse, Line, BSplineCurve, Other
from .surface import *
from .winding_number import find_surface_uv_for_curve
import numpy as np
# from joblib import Parallel, delayed







class Shape:
    """Represents a geometric part (one part from the CAD model) with geometry and topology assembled."""
    def __init__(self, geometry_data, topology_data, version, spacing=0.02):

        self._geometry_data = self._geometry_data(geometry_data, version)
        self._topology_data = self._topology_data(topology_data, version)
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
        def __init__(self, geometry_data, version):
            self.curves2d, self.curves3d, self.surfaces, self.bbox, self.vertices = [], [], [], [], []
            self.__init_geometry(geometry_data, version)

        def __init_geometry(self, data, version):
            if version == "3.0":
                curve2d_index = data.get('2dcurves_index', {})[()]
                curve2d_data = data.get('2dcurves', {})[()]
                self.curves2d = [None] * (len(curve2d_index)-1)

                for i in range(len(curve2d_index)-1):
                    tmp = curve2d_data[curve2d_index[i]:curve2d_index[i+1]]
                    ctype = int(tmp[0])
                    interval = tmp[1:3]

                    if ctype == 0:  # Line
                        self.curves2d[i] = Line(None, interval=interval,
                                                location=np.array(tmp[3:5]),
                                                direction=np.array(tmp[5:7]))

                    elif ctype == 1:  # Circle
                        self.curves2d[i] = Circle(None,
                                                interval=interval,
                                                location=np.array(tmp[3:5]),
                                                x_axis=np.array(tmp[5:7]),
                                                y_axis=np.array(tmp[7:9]),
                                                radius=tmp[9])
                    elif ctype == 2:  # Ellipse
                        self.curves2d[i] = Ellipse(None,
                                                interval=interval,
                                                focus1=np.array(tmp[3:5]),
                                                focus2=np.array(tmp[5:7]),
                                                x_axis=np.array(tmp[7:9]),
                                                y_axis=np.array(tmp[9:11]),
                                                maj_radius=tmp[11],
                                                min_radius=tmp[12])
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

                        self.curves2d[i] = BSplineCurve(None,
                                                        interval=interval,
                                                        degree=degree,
                                                        continuity=continuity,
                                                        rational=rational,
                                                        periodic=periodic,
                                                        closed=closed,
                                                        poles=poles,
                                                        knots=knots,
                                                        weights=weights)
                    elif ctype == 4:  # Other
                        self.curves2d[i] = Other(None, interval=interval)
                    else:
                        raise ValueError(f"Unknown curve type: {ctype}")

                del curve2d_index
                del curve2d_data

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
                        self.curves3d[i] = Line(None,
                                                interval=interval,
                                                location=tmp[3:6],
                                                direction=tmp[6:9],
                                                transform=transform)
                    elif ctype == 1:  # Circle
                        self.curves3d[i] = Circle(None,
                                                 interval=interval,
                                                 location=tmp[3:6],
                                                 x_axis=tmp[6:9],
                                                 y_axis=tmp[9:12],
                                                 z_axis=tmp[12:15],
                                                 radius=tmp[15],
                                                 transform=transform)
                    elif ctype == 2:  # Ellipse
                        self.curves3d[i] = Ellipse(None,
                                                  interval=interval,
                                                  focus1=tmp[3:6],
                                                  focus2=tmp[6:9],
                                                  x_axis=tmp[9:12],
                                                  y_axis=tmp[12:15],
                                                  z_axis=tmp[15:18],
                                                  maj_radius=tmp[18],
                                                  min_radius=tmp[19],
                                                   transform=transform)
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

                        self.curves3d[i] = BSplineCurve(None,
                                                       interval=interval,
                                                       degree=degree,
                                                       continuity=continuity,
                                                       rational=rational,
                                                       periodic=periodic,
                                                       closed=closed,
                                                       poles=poles,
                                                       knots=knots,
                                                       weights=weights,
                                                        transform=transform)
                    elif ctype == 4:  # Other
                        self.curves3d[i] = Other(None, interval=interval, transform=transform)
                    else:
                        raise ValueError(f"Unknown curve type: {ctype} for curve {i}")

                del curve3d_index
                del curve3d_data

                surface_index = data.get('surfaces_index', {})[()]
                surface_data = data.get('surfaces', {})[()]
                self.surfaces = [None] * (len(surface_index) - 1)
                surface_type_map = {
                    0: 'Plane',
                    1: 'Cylinder',
                    2: 'Cone',
                    3: 'Sphere',
                    4: 'Torus',
                    5: 'BSpline',
                    6: 'Extrusion',
                    7: 'Revolution',
                    8: 'Offset',
                    9: 'Other'
                }

                for i in range(len(surface_index) - 1):
                    tmp = surface_data[surface_index[i]:surface_index[i + 1]]
                    transform = np.array(tmp[-12:]).reshape((3, 4))
                    payload = tmp[:-12]
                    stype = int(payload[0])

                    trim_domain = np.array(payload[1:5]).reshape((2, 2))
                    idx = 5

                    if stype == 0:  # Plane
                        self.surfaces[i] = Plane(None,
                                                trim_domain=trim_domain,
                                                transform=transform,
                                                location=np.array(payload[idx:idx + 3]),
                                                coefficients=np.array(payload[idx + 3:idx + 7]),
                                                x_axis=np.array(payload[idx + 7:idx + 10]),
                                                y_axis=np.array(payload[idx + 10:idx + 13]),
                                                z_axis=np.array(payload[idx + 13:idx + 16]))

                    elif stype == 1:  # Cylinder
                        self.surfaces[i] = Cylinder(None,
                                                   trim_domain=trim_domain,
                                                   transform=transform,
                                                   location=np.array(payload[idx:idx + 3]),
                                                   radius=payload[idx + 3],
                                                   coefficients=np.array(payload[idx + 4:idx + 8]),
                                                   x_axis=np.array(payload[idx + 8:idx + 11]),
                                                   y_axis=np.array(payload[idx + 11:idx + 14]),
                                                   z_axis=np.array(payload[idx + 14:idx + 17]))

                    elif stype == 2:  # Cone
                        self.surfaces[i] = Cone(None,
                                                trim_domain=trim_domain,
                                                transform=transform,
                                                location=np.array(payload[idx:idx + 3]),
                                                radius=payload[idx + 3],
                                                coefficients=np.array(payload[idx + 4:-13]),
                                                apex=np.array(payload[-13:-10]),
                                                angle=payload[-10],
                                                x_axis=np.array(payload[-9:-6]),
                                                y_axis=np.array(payload[-6:-3]),
                                                z_axis=np.array(payload[-3:]))

                    elif stype == 3:  # Sphere
                        self.surfaces[i] = Sphere(None,
                                                 trim_domain=trim_domain,
                                                 transform=transform,
                                                 location=np.array(payload[idx:idx + 3]),
                                                 radius=payload[idx + 3],
                                                 coefficients=np.array(payload[idx + 4:idx + 8]),
                                                 x_axis=np.array(payload[idx + 8:idx + 11]),
                                                 y_axis=np.array(payload[idx + 11:idx + 14]),
                                                 z_axis=np.array(payload[idx + 14:idx + 17]))

                    elif stype == 4:  # Torus
                        self.surfaces[i] = Torus(None,
                                                trim_domain=trim_domain,
                                                transform=transform,
                                                location=np.array(payload[idx:idx + 3]),
                                                max_radius=payload[idx + 3],
                                                min_radius=payload[idx + 4],
                                                x_axis=np.array(payload[idx + 5:idx + 8]),
                                                y_axis=np.array(payload[idx + 8:idx + 11]),
                                                z_axis=np.array(payload[idx + 11:idx + 14]))

                    elif stype == 5:  # BSplineSurface
                        u_degree, v_degree, continuity, u_rational, v_rational, u_periodic, v_periodic, u_closed, v_closed, is_trimmed, face_domain_len = payload[idx:idx + 11]
                        idx += 11
                        face_domain_len = int(face_domain_len)
                        face_domain = np.array(payload[idx:idx + face_domain_len])
                        idx += face_domain_len

                        len_poles = int(payload[idx])
                        pshape = [int(val) for val in payload[idx + 1:idx + 4]]
                        idx += 4
                        poles = np.array(payload[idx:idx + len_poles]).reshape(pshape)
                        idx += len_poles

                        len_uknots = int(payload[idx])
                        idx += 1
                        u_knots = np.array(payload[idx:idx + len_uknots])
                        idx += len_uknots

                        len_vknots = int(payload[idx])
                        idx += 1
                        v_knots = np.array(payload[idx:idx + len_vknots])
                        idx += len_vknots

                        len_weights = int(payload[idx])
                        wshape = [int(val) for val in payload[idx + 1:idx + 3]]
                        idx += 3
                        weights = np.array(payload[idx:idx + len_weights]).reshape(wshape)
                        # weights_dict = {str(j): weights[j] for j in range(weights.shape[0])}

                        self.surfaces[i] = BSplineSurface(None,
                                                         trim_domain=trim_domain,
                                                         transform=transform,
                                                         u_degree=int(u_degree),
                                                         v_degree=int(v_degree),
                                                         continuity=int(continuity),
                                                         u_rational=bool(u_rational),
                                                         v_rational=bool(v_rational),
                                                         u_periodic=bool(u_periodic),
                                                         v_periodic=bool(v_periodic),
                                                         u_closed=bool(u_closed),
                                                         v_closed=bool(v_closed),
                                                         is_trimmed=bool(is_trimmed),
                                                         face_domain=face_domain,
                                                         poles=poles,
                                                         u_knots=u_knots,
                                                         v_knots=v_knots,
                                                         weights=weights)

                    elif stype == 6:  # Extrusion
                        self.surfaces[i] = Extrusion(None,
                                                     trim_domain=trim_domain,
                                                     transform=transform,
                                                     direction=np.array(payload[idx:idx + 3]),
                                                     curve=self.curves3d_data[int(payload[idx + 3])])

                    elif stype == 7:  # Revolution
                        self.surfaces[i] = Revolution(None,
                                                     trim_domain=trim_domain,
                                                     transform=transform,
                                                     location=np.array(payload[idx:idx + 3]),
                                                     z_axis=np.array(payload[idx + 3:idx + 6]),
                                                     curve=self.curves3d_data[int(payload[idx + 6])])

                    elif stype == 8:  # Offset
                        self.surfaces[i] = Offset(None,
                                                 trim_domain=trim_domain,
                                                 transform=transform,
                                                 value=np.array(payload[idx]),
                                                 surface=self.surfaces[int(payload[idx + 1])])


                    elif stype == 9:  # Other
                        pass

                del surface_index
                del surface_data
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
        def __init__(self, topology_data, version="3.0"):
            self.edges, self.faces, self.halfedges, self.loops, self.shells, self.solids = [], [], [], [], [], []
            self.__init_topology(topology_data, version)

        def __init_topology(self, data, version):
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
                del edge_index
                del edge_data

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
                del halfedge_index
                del halfedge_data

                # Loops
                loop_index = data.get('loop_index', {})[()]
                loop_data = data.get('loops', {})[()]
                self.loops = [None] * (len(loop_index)-1)
                for i in range(len(loop_index)-1):
                    tmp = loop_data[loop_index[i]:loop_index[i+1]]
                    local_loop = {'id': i, 'halfedges': tmp}
                    self.loops[i] = Loop(local_loop)
                # Parallel(n_jobs=-1, backend="threading")(delayed(process_loop)(i) for i in range(len(loop_index)-1))
                del loop_index
                del loop_data

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
                del shell_index
                del shell_data

                # Solids
                solid_index = data.get('solid_index', {})[()]
                solid_data = data.get('solids', {})[()]
                self.solids = [None] * (len(solid_index)-1)
                for i in range(len(solid_index)-1):
                    tmp = solid_data[solid_index[i]:solid_index[i+1]]
                    local_solid = {'id': i, 'shells': tmp}
                    self.solids[i] = TopoSolid(local_solid)
                # Parallel(n_jobs=-1, backend="threading")(delayed(process_solid)(i) for i in range(len(solid_index)-1))
                del solid_index
                del solid_data

                # Faces
                face_index = data.get('face_index', {})[()]
                face_data = data.get('faces', {})[()]
                self.faces = [None] * (len(face_index)-1)
                for i in range(len(face_index)-1):
                    tmp = face_data[face_index[i]:face_index[i+1]]

                    exact_domain = tmp[:4]
                    has_singularities = bool(tmp[4])
                    nr_singularities = tmp[5]
                    # singularities = tmp[6:6+nr_singularities] #TODO new format for singularities
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
                del face_index
                del face_data
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


        def __eq__(self, other):
            if not isinstance(other, self.__class__):
                return NotImplemented

            def summarize_edges(edges):
                return {
                    e.id: (
                        int(e.start_vertex),
                        int(e.end_vertex),
                        e.curve3d,
                    )
                    for e in edges
                }

            def summarize_halfedges(halfedges):
                summary = {}
                for h in halfedges:
                    mate = getattr(h, "mates", None)
                    mate_id = mate.id if mate is not None else None

                    summary[h.id] = (
                        bool(h.orientation_wrt_edge),
                        h.edge.id,
                        mate_id,
                        h.curve2d,
                    )
                return summary

            def summarize_loops(loops):

                return {
                    loop.id: tuple(h.id for h in loop.halfedges)
                    for loop in loops
                }

            def summarize_faces(faces):
                summary = {}
                for f in faces:
                    exact_domain = tuple(np.asarray(f.exact_domain).ravel().tolist())
                    loop_ids = tuple(loop.id for loop in f.loops)

                    sing_raw = getattr(f, "singularities", None)
                    sing_summary = None
                    if isinstance(sing_raw, dict):
                        sing_summary = {}
                        for key, inner in sing_raw.items():
                            norm_inner = {}
                            for k2, v in inner.items():
                                if isinstance(v, (np.ndarray, list, tuple)):
                                    norm_inner[k2] = np.asarray(v).ravel().tolist()
                                else:
                                    norm_inner[k2] = v
                                sing_summary[key] = norm_inner

                    summary[f.id] = (
                        exact_domain,
                        bool(getattr(f, "has_singularities", False)),
                        int(getattr(f, "nr_singularities", 0)),
                        int(f.outer_loop),
                        bool(f.surface_orientation),
                        loop_ids,
                        # sing_summary,
                        f.surface,
                    )
                return summary


            def summarize_shells(shells):
                summary = {}
                for s in shells:
                    faces = tuple((face.id, bool(orient)) for (face, orient) in s.faces)
                    summary[s.id] = (
                        bool(s.orientation_wrt_solid),
                        faces,
                    )
                return summary

            def summarize_solids(solids):
                summary = {}
                for s in solids:
                    shell_ids = tuple(shell.id for shell in s.shells)
                    summary[s.id] = shell_ids
                return summary

            return (
                summarize_edges(self.edges) == summarize_edges(other.edges)
                and summarize_halfedges(self.halfedges) == summarize_halfedges(other.halfedges)
                and summarize_loops(self.loops) == summarize_loops(other.loops)
                and summarize_faces(self.faces) == summarize_faces(other.faces)
                and summarize_shells(self.shells) == summarize_shells(other.shells)
                and summarize_solids(self.solids) == summarize_solids(other.solids)
            )



def _get_edges(edge_data): return Edge(edge_data)
def _get_faces(face_data): return Face(face_data)
def _get_halfedges(halfedge_data): return Halfedge(halfedge_data)
def _get_loops(loop_data): return Loop(loop_data)
