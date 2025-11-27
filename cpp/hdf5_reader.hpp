#include <cstdlib>

#include <memory>
#include <string>
#include <nanospline/PatchBase.h>
#include <nanospline/CurveBase.h>

template <int N>
class CurveND
{
public:
	std::shared_ptr<nanospline::CurveBase<double, N>> curve;
	Eigen::Matrix<double, 3, 4, Eigen::RowMajor> transform;
	Eigen::Vector2d interval;
	std::string type;
};

class Patch
{
public:
	std::shared_ptr<nanospline::PatchBase<double, 3>> patch;
	Eigen::Matrix<double, 3, 4, Eigen::RowMajor> transform;
	Eigen::Matrix2d domain;
	std::string type;
};

using Mesh = std::pair<Eigen::Matrix<double, Eigen::Dynamic, 3>, Eigen::Matrix<int64_t, Eigen::Dynamic, 3>>;

class Edge
{
public:
	int64_t start_vertex;
	int64_t end_vertex;
	int64_t curve_id;
};

class Face
{
public:
	std::array<double, 4> exact_domain;
	bool has_singularities;
	std::vector<int64_t> loops;
	int64_t nr_singularities;
	int64_t outer_loop;
	int64_t surface;
	bool surface_orientation;
	// TODO add singularities
};

class HalfEdge
{
public:
	int64_t curve_id;
	int64_t edge;
	std::vector<int64_t> mates;
	bool orientation;
};

class Loop
{
public:
	std::vector<int64_t> half_edges;
};

class Shell
{
public:
	std::vector<int64_t> faces;
	bool orientation;
};

class Solid
{
public:
	std::vector<int64_t> shells;
};

class Part
{
public:
	Eigen::Matrix<double, Eigen::Dynamic, 3> vertices;
	std::vector<Mesh> meshes;
	Eigen::Matrix<double, 2, 3> bbox;

	std::vector<CurveND<2>> curves2d;
	std::vector<CurveND<3>> curves3d;
	std::vector<Patch> surfaces;

	std::vector<Edge> edges;
	std::vector<Face> faces;
	std::vector<HalfEdge> half_edges;
	std::vector<Loop> loops;
	std::vector<Shell> shells;
	std::vector<Solid> solids;
};

void read_parts(const std::string &path, std::vector<Part> &part_list);
