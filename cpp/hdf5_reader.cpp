#include <cstdlib>

#include "hdf5_reader.hpp"

#include <h5pp/h5pp.h>

#include <nanospline/BSpline.h>
#include <nanospline/BSplinePatch.h>
#include <nanospline/Circle.h>
#include <nanospline/Cone.h>
#include <nanospline/CurveBase.h>
#include <nanospline/Cylinder.h>
#include <nanospline/Ellipse.h>
#include <nanospline/ExtrusionPatch.h>
#include <nanospline/Line.h>
#include <nanospline/NURBSPatch.h>
#include <nanospline/OffsetPatch.h>
#include <nanospline/PatchBase.h>
#include <nanospline/Plane.h>
#include <nanospline/RevolutionPatch.h>
#include <nanospline/Sphere.h>
#include <nanospline/Torus.h>

template <int N>
void read_curve(const h5pp::File &file,
				const std::string &curve_path,
				std::vector<CurveND<N>> &ret_curves,
				int curve_id)
{
	std::array<double, 2> interval = file.readDataset<std::array<double, 2>>(curve_path + "/interval");
	ret_curves[curve_id].interval(0) = interval[0];
	ret_curves[curve_id].interval(1) = interval[1];

	Eigen::Matrix<double, 3, 4, Eigen::RowMajor> &transform = ret_curves[curve_id].transform;
	if (!file.linkExists(curve_path + "/transform"))
	{
		transform.setZero();
		transform(0, 0) = 1;
		transform(1, 1) = 1;
		transform(2, 2) = 1;
	}
	else
	{
		std::array<double, 12> transform_vec = file.readDataset<std::array<double, 12>>(curve_path + "/transform");
		transform = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>::Map(transform_vec.data());
	}
	std::string type = file.readDataset<std::string>(curve_path + "/type");
	ret_curves[curve_id].type = type;

	if (type == "Circle")
	{
		double radius = file.readDataset<double>(curve_path + "/radius");

		std::array<double, N> location = file.readDataset<std::array<double, N>>(curve_path + "/location");
		Eigen::Matrix<double, 1, N> center = Eigen::Matrix<double, 1, N>::Map(location.data());

		std::array<double, N> xframe = file.readDataset<std::array<double, N>>(curve_path + "/x_axis");
		std::array<double, N> yframe = file.readDataset<std::array<double, N>>(curve_path + "/y_axis");
		Eigen::Matrix<double, 2, N> frame;
		frame.row(0) = Eigen::Matrix<double, 1, N>::Map(xframe.data());
		frame.row(1) = Eigen::Matrix<double, 1, N>::Map(yframe.data());

		auto curve = std::make_shared<nanospline::Circle<double, N>>();
		curve->set_center(center);
		curve->set_radius(radius);
		curve->set_frame(frame);
		curve->set_domain_lower_bound(interval[0]);
		curve->set_domain_upper_bound(interval[1]);
		curve->initialize();
		ret_curves[curve_id].curve = curve;
	}
	else if (type == "Line")
	{
		std::array<double, N> location = file.readDataset<std::array<double, N>>(curve_path + "/location");
		std::array<double, N> direction = file.readDataset<std::array<double, N>>(curve_path + "/direction");
		Eigen::Matrix<double, 1, N> start = Eigen::Matrix<double, 1, N>::Map(location.data());
		Eigen::Matrix<double, 1, N> dir = Eigen::Matrix<double, 1, N>::Map(direction.data());

		auto curve = std::make_shared<nanospline::Line<double, N>>();
		curve->set_domain_lower_bound(interval[0]);
		curve->set_domain_upper_bound(interval[1]);
		curve->set_location(start);
		curve->set_direction(dir);
		curve->initialize();
		ret_curves[curve_id].curve = curve;
	}
	else if (type == "BSpline")
	{
		std::vector<double> knots_vector = file.readDataset<std::vector<double>>(curve_path + "/knots");
		std::vector<double> control_points_vector = file.readDataset<std::vector<double>>(curve_path + "/poles");
		Eigen::Matrix<double, Eigen::Dynamic, N, Eigen::RowMajor> control_points = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, N, Eigen::RowMajor>>(control_points_vector.data(), control_points_vector.size() / N, N);
		Eigen::Matrix<double, Eigen::Dynamic, 1> knots = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>>(knots_vector.data(), knots_vector.size(), 1);
		int64_t degree = file.readDataset<int64_t>(curve_path + "/degree");
		bool rational = file.readDataset<bool>(curve_path + "/rational");
		if (rational)
		{
			std::vector<double> weights_vector = file.readDataset<std::vector<double>>(curve_path + "/weights");
			Eigen::Matrix<double, Eigen::Dynamic, 1> weights = Eigen::Matrix<double, Eigen::Dynamic, 1>::Map(weights_vector.data(), weights_vector.size(), 1);

			auto curve = std::make_shared<nanospline::NURBS<double, N, -1>>();
			curve->set_weights(weights);
			curve->set_control_points(control_points);
			curve->set_knots(knots);
			curve->initialize();
			ret_curves[curve_id].curve = curve;
		}
		else
		{
			auto curve = std::make_shared<nanospline::BSpline<double, N, -1>>();
			curve->set_control_points(control_points);
			curve->set_knots(knots);
			curve->initialize();
			ret_curves[curve_id].curve = curve;
		}
	}
	else if (type == "Ellipse")
	{
		std::array<double, N> focus1 = file.readDataset<std::array<double, N>>(curve_path + "/focus1");
		Eigen::Matrix<double, 1, N> f1 = Eigen::Matrix<double, 1, N>::Map(focus1.data());
		std::array<double, N> focus2 = file.readDataset<std::array<double, N>>(curve_path + "/focus2");
		Eigen::Matrix<double, 1, N> f2 = Eigen::Matrix<double, 1, N>::Map(focus2.data());
		Eigen::Matrix<double, 1, N> center = (f1 + f2) / 2.;

		double maj_radius = file.readDataset<double>(curve_path + "/maj_radius");
		double min_radius = file.readDataset<double>(curve_path + "/min_radius");
		std::array<double, N> xframe = file.readDataset<std::array<double, N>>(curve_path + "/x_axis");
		std::array<double, N> yframe = file.readDataset<std::array<double, N>>(curve_path + "/y_axis");

		Eigen::Matrix<double, 2, N> frame;
		frame.row(0) = Eigen::Matrix<double, 1, N>::Map(xframe.data());
		frame.row(1) = Eigen::Matrix<double, 1, N>::Map(yframe.data());

		auto curve = std::make_shared<nanospline::Ellipse<double, N>>();
		curve->set_center(center);
		curve->set_major_radius(maj_radius);
		curve->set_minor_radius(min_radius);
		curve->set_frame(frame);
		curve->set_domain_lower_bound(interval[0]);
		curve->set_domain_upper_bound(interval[1]);
		curve->initialize();
		ret_curves[curve_id].curve = curve;
	}
	else if (type == "Other")
	{
		ret_curves[curve_id].curve = nullptr;
	}
	else
	{
		throw std::runtime_error("Unknown curve type");
	}
}

void read_surface(const h5pp::File &file,
				  const std::string &surface_path,
				  std::vector<Patch> &ret_patches,
				  int patch_id)
{
	Eigen::Matrix<double, 3, 4, Eigen::RowMajor> &transform = ret_patches[patch_id].transform;
	if (!file.linkExists(surface_path + "/transform"))
	{
		transform.setZero();
		transform(0, 0) = 1;
		transform(1, 1) = 1;
		transform(2, 2) = 1;
	}
	else
	{
		std::array<double, 12> transform_vec = file.readDataset<std::array<double, 12>>(surface_path + "/transform");
		transform = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>::Map(transform_vec.data());
	}
	std::string type = file.readDataset<std::string>(surface_path + "/type");
	ret_patches[patch_id].type = type;

	std::array<double, 4> trimmed_domain = file.readDataset<std::array<double, 4>>(surface_path + "/trim_domain");
	ret_patches[patch_id].domain(0, 0) = trimmed_domain[0];
	ret_patches[patch_id].domain(0, 1) = trimmed_domain[1];
	ret_patches[patch_id].domain(1, 0) = trimmed_domain[2];
	ret_patches[patch_id].domain(1, 1) = trimmed_domain[3];

	if (type == "Plane")
	{
		std::array<double, 3> location = file.readDataset<std::array<double, 3>>(surface_path + "/location");
		std::array<double, 3> x_axis = file.readDataset<std::array<double, 3>>(surface_path + "/x_axis");
		std::array<double, 3> y_axis = file.readDataset<std::array<double, 3>>(surface_path + "/y_axis");
		Eigen::Matrix<double, 1, 3> loc = Eigen::Matrix<double, 1, 3>::Map(location.data());
		Eigen::Matrix<double, 1, 3> x = Eigen::Matrix<double, 1, 3>::Map(x_axis.data());
		Eigen::Matrix<double, 1, 3> y = Eigen::Matrix<double, 1, 3>::Map(y_axis.data());
		Eigen::Matrix<double, 2, 3> frame;
		frame.row(0) = x;
		frame.row(1) = y;

		auto patch = std::make_shared<nanospline::Plane<double, 3>>();
		patch->set_location(loc);
		patch->set_frame(frame);
		patch->set_u_lower_bound(trimmed_domain[0]);
		patch->set_u_upper_bound(trimmed_domain[1]);
		patch->set_v_lower_bound(trimmed_domain[2]);
		patch->set_v_upper_bound(trimmed_domain[3]);
		patch->initialize();
		ret_patches[patch_id].patch = patch;
	}
	else if (type == "BSpline")
	{
		std::vector<double> u_knots = file.readDataset<std::vector<double>>(surface_path + "/u_knots");
		std::vector<double> v_knots = file.readDataset<std::vector<double>>(surface_path + "/v_knots");
		int64_t degree_u = file.readDataset<int64_t>(surface_path + "/u_degree");
		int64_t degree_v = file.readDataset<int64_t>(surface_path + "/v_degree");
		bool u_periodic = file.readDataset<bool>(surface_path + "/u_closed");
		bool v_periodic = file.readDataset<bool>(surface_path + "/v_closed");
		std::vector<double> poles = file.readDataset<std::vector<double>>(surface_path + "/poles");
		auto shape = file.getDatasetDimensions(surface_path + "/poles");
		Eigen::Matrix<double, Eigen::Dynamic, 3> grid(shape[0] * shape[1], shape[2]);
		grid.setZero();
		for (int i = 0; i < shape[0]; i++)
		{
			for (int j = 0; j < shape[1]; j++)
			{
				grid.row(i * (shape[1]) + j) = Eigen::Matrix<double, 1, 3>::Map(
					poles.data() + 3 * (i * (shape[1]) + j));
			}
		}
		bool u_rational = file.readDataset<bool>(surface_path + "/u_rational");
		bool v_rational = file.readDataset<bool>(surface_path + "/v_rational");

		Eigen::Matrix<double, Eigen::Dynamic, 1> u_knots_eigen = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>>(u_knots.data(), u_knots.size(), 1);
		Eigen::Matrix<double, Eigen::Dynamic, 1> v_knots_eigen = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>>(v_knots.data(), v_knots.size(), 1);

		if (u_rational || v_rational)
		{
			auto weights = file.findDatasets("", surface_path + "/weights", -1, 0);
			std::vector<double> all_weights;

			for (const auto &weight : weights)
			{
				std::string weight_path = surface_path + "/weights/" + weight;
				std::vector<double> weight_vector = file.readDataset<std::vector<double>>(weight_path);
				all_weights.insert(all_weights.end(), weight_vector.begin(), weight_vector.end());
			}

			// Convert to Eigen::Matrix and set weights
			Eigen::Matrix<double, Eigen::Dynamic, 1> weights_matrix = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>>(all_weights.data(), all_weights.size());

			auto patch = std::make_shared<nanospline::NURBSPatch<double, 3, -1, -1>>();
			patch->set_knots_u(u_knots_eigen);
			patch->set_knots_v(v_knots_eigen);
			patch->swap_control_grid(grid);
			patch->set_periodic_u(u_periodic);
			patch->set_periodic_v(v_periodic);
			patch->set_degree_u(static_cast<int>(degree_u));
			patch->set_degree_v(static_cast<int>(degree_v));
			// patch->set_u_lower_bound(trimmed_domain[0]);
			// patch->set_u_upper_bound(trimmed_domain[1]);
			// patch->set_v_lower_bound(trimmed_domain[2]);
			// patch->set_v_upper_bound(trimmed_domain[3]);
			patch->set_weights(weights_matrix);
			patch->initialize();
			ret_patches[patch_id].patch = patch;
		}
		else
		{
			auto patch = std::make_shared<nanospline::BSplinePatch<double, 3, -1, -1>>();
			patch->set_degree_u(static_cast<int>(degree_u));
			patch->set_degree_v(static_cast<int>(degree_v));
			patch->set_knots_u(u_knots_eigen);
			patch->set_knots_v(v_knots_eigen);
			patch->swap_control_grid(grid);
			// patch->set_u_lower_bound(trimmed_domain[0]);
			// patch->set_u_upper_bound(trimmed_domain[1]);
			// patch->set_v_lower_bound(trimmed_domain[2]);
			// patch->set_v_upper_bound(trimmed_domain[3]);
			patch->set_periodic_u(u_periodic);
			patch->set_periodic_v(v_periodic);
			patch->initialize();
			ret_patches[patch_id].patch = patch;
		}
	}
	else if (type == "Torus")
	{
		std::array<double, 3> location = file.readDataset<std::array<double, 3>>(surface_path + "/location");
		std::array<double, 3> x_axis = file.readDataset<std::array<double, 3>>(surface_path + "/x_axis");
		std::array<double, 3> y_axis = file.readDataset<std::array<double, 3>>(surface_path + "/y_axis");
		std::array<double, 3> z_axis = file.readDataset<std::array<double, 3>>(surface_path + "/z_axis");

		Eigen::Matrix<double, 1, 3> loc = Eigen::Matrix<double, 1, 3>::Map(location.data());
		Eigen::Matrix<double, 1, 3> x = Eigen::Matrix<double, 1, 3>::Map(x_axis.data());
		Eigen::Matrix<double, 1, 3> y = Eigen::Matrix<double, 1, 3>::Map(y_axis.data());
		Eigen::Matrix<double, 1, 3> z = Eigen::Matrix<double, 1, 3>::Map(z_axis.data());
		Eigen::Matrix<double, 3, 3> frame;
		frame.row(0) = x;
		frame.row(1) = y;
		frame.row(2) = z;

		double min_radius = file.readDataset<double>(surface_path + "/min_radius");
		double major_radius = file.readDataset<double>(surface_path + "/max_radius");

		auto patch = std::make_shared<nanospline::Torus<double, 3>>();
		patch->set_location(loc);
		patch->set_frame(frame);
		patch->set_major_radius(major_radius);
		patch->set_minor_radius(min_radius);
		patch->set_u_lower_bound(trimmed_domain[0]);
		patch->set_u_upper_bound(trimmed_domain[1]);
		patch->set_v_lower_bound(trimmed_domain[2]);
		patch->set_v_upper_bound(trimmed_domain[3]);
		patch->initialize();
		// // set periodicity
		// if (abs(abs(trimmed_domain[1] - trimmed_domain[0]) - 2 * M_PI) < 1e-4) {
		//   torus.set_periodic_u(true);
		// } else {
		//   torus.set_periodic_u(false);
		// }
		// if (abs(abs(trimmed_domain[3] - trimmed_domain[2]) - 2 * M_PI) < 1e-4) {
		//   torus.set_periodic_v(true);
		// } else {
		//   torus.set_periodic_v(false);
		// }
		ret_patches[patch_id].patch = patch;
	}
	else if (type == "Cylinder")
	{
		std::array<double, 3> location = file.readDataset<std::array<double, 3>>(surface_path + "/location");
		std::array<double, 3> x_axis = file.readDataset<std::array<double, 3>>(surface_path + "/x_axis");
		std::array<double, 3> y_axis = file.readDataset<std::array<double, 3>>(surface_path + "/y_axis");
		std::array<double, 3> z_axis = file.readDataset<std::array<double, 3>>(surface_path + "/z_axis");
		Eigen::Matrix<double, 1, 3> loc = Eigen::Matrix<double, 1, 3>::Map(location.data());
		Eigen::Matrix<double, 1, 3> x = Eigen::Matrix<double, 1, 3>::Map(x_axis.data());
		Eigen::Matrix<double, 1, 3> y = Eigen::Matrix<double, 1, 3>::Map(y_axis.data());
		Eigen::Matrix<double, 1, 3> z = Eigen::Matrix<double, 1, 3>::Map(z_axis.data());
		Eigen::Matrix<double, 3, 3> frame;
		frame.row(0) = x;
		frame.row(1) = y;
		frame.row(2) = z;

		double radius = file.readDataset<double>(surface_path + "/radius");

		auto patch = std::make_shared<nanospline::Cylinder<double, 3>>();
		patch->set_location(loc);
		patch->set_frame(frame);
		patch->set_radius(radius);
		patch->set_u_lower_bound(trimmed_domain[0]);
		patch->set_u_upper_bound(trimmed_domain[1]);
		patch->set_v_lower_bound(trimmed_domain[2]);
		patch->set_v_upper_bound(trimmed_domain[3]);
		patch->initialize();
		// set periodicity
		// use trim domain to determine periodicity. But note this doesn't decide
		// topology. TThe final topology will still be decided by tracing
		// check if the u domain covers full 2pi
		// if (abs(abs(trimmed_domain[1] - trimmed_domain[0]) - 2 * M_PI) < 1e-4) {
		//   cylinder.set_periodic_u(true);
		// } else {
		//   cylinder.set_periodic_u(false);
		// }
		// cylinder.set_periodic_v(false);
		ret_patches[patch_id].patch = patch;
	}
	else if (type == "Cone")
	{
		std::array<double, 3> location = file.readDataset<std::array<double, 3>>(surface_path + "/location");
		std::array<double, 3> x_axis = file.readDataset<std::array<double, 3>>(surface_path + "/x_axis");
		std::array<double, 3> y_axis = file.readDataset<std::array<double, 3>>(surface_path + "/y_axis");
		std::array<double, 3> z_axis = file.readDataset<std::array<double, 3>>(surface_path + "/z_axis");

		Eigen::Matrix<double, 1, 3> loc = Eigen::Matrix<double, 1, 3>::Map(location.data());
		Eigen::Matrix<double, 1, 3> x = Eigen::Matrix<double, 1, 3>::Map(x_axis.data());
		Eigen::Matrix<double, 1, 3> y = Eigen::Matrix<double, 1, 3>::Map(y_axis.data());
		Eigen::Matrix<double, 1, 3> z = Eigen::Matrix<double, 1, 3>::Map(z_axis.data());
		Eigen::Matrix<double, 3, 3> frame;
		frame.row(0) = x;
		frame.row(1) = y;
		frame.row(2) = z;

		double radius = file.readDataset<double>(surface_path + "/radius");
		double angle = file.readDataset<double>(surface_path + "/angle");

		auto patch = std::make_shared<nanospline::Cone<double, 3>>();
		patch->set_location(loc);
		patch->set_frame(frame);
		patch->set_radius(radius);
		patch->set_angle(angle);
		patch->set_u_lower_bound(trimmed_domain[0]);
		patch->set_u_upper_bound(trimmed_domain[1]);
		patch->set_v_lower_bound(trimmed_domain[2]);
		patch->set_v_upper_bound(trimmed_domain[3]);
		patch->initialize();
		// set periodicity
		// if (abs(abs(trimmed_domain[1] - trimmed_domain[0]) - 2 * M_PI) < 1e-4) {
		//   patch->set_periodic_u(true);
		// } else {
		//   patch->set_periodic_u(false);
		// }
		// patch->set_periodic_v(false);

		ret_patches[patch_id].patch = patch;
	}
	else if (type == "Sphere")
	{
		std::array<double, 3> location = file.readDataset<std::array<double, 3>>(surface_path + "/location");
		std::array<double, 3> x_axis = file.readDataset<std::array<double, 3>>(surface_path + "/x_axis");
		std::array<double, 3> y_axis = file.readDataset<std::array<double, 3>>(surface_path + "/y_axis");

		Eigen::Matrix<double, 1, 3> loc = Eigen::Matrix<double, 1, 3>::Map(location.data());
		Eigen::Matrix<double, 1, 3> x = Eigen::Matrix<double, 1, 3>::Map(x_axis.data());
		Eigen::Matrix<double, 1, 3> y = Eigen::Matrix<double, 1, 3>::Map(y_axis.data());
		Eigen::Matrix<double, 1, 3> z; // TODO= x.cross(y).normalized();
		Eigen::Matrix<double, 3, 3> frame;
		frame.row(0) = x;
		frame.row(1) = y;
		frame.row(2) = z;

		double radius = file.readDataset<double>(surface_path + "/radius");

		auto patch = std::make_shared<nanospline::Sphere<double, 3>>();
		patch->set_location(loc);
		patch->set_frame(frame);
		patch->set_radius(radius);
		patch->set_u_lower_bound(trimmed_domain[0]);
		patch->set_u_upper_bound(trimmed_domain[1]);
		patch->set_v_lower_bound(trimmed_domain[2]);
		patch->set_v_upper_bound(trimmed_domain[3]);
		patch->initialize();
		// set periodicity
		// if (abs(abs(trimmed_domain[1] - trimmed_domain[0]) - 2 * M_PI) < 1e-4) {
		//   patch->set_periodic_u(true);
		// } else {
		//   patch->set_periodic_u(false);
		// }
		// patch->set_periodic_v(false);
		ret_patches[patch_id].patch = patch;
	}
	else if (type == "Revolution")
	{
		std::array<double, 3> location = file.readDataset<std::array<double, 3>>(surface_path + "/location");
		std::array<double, 3> z_axis = file.readDataset<std::array<double, 3>>(surface_path + "/z_axis");

		// auto curve_path = surface_path + "/curve/";
		// std::vector<T> interval =
		//     file.readDataset<std::vector<T>>(curve_path + "/interval");
		// std::vector<T> transform_vec =
		//     file.readDataset<std::vector<T>>(curve_path + "/transform");
		// Eigen::Matrix<T, 3, 4, Eigen::RowMajor> transform =
		//     Eigen::Matrix<T, 3, 4, Eigen::RowMajor>::Map(transform_vec.data());
		// auto feature_edge = read_curve<T, 3>(file, curve_path);
		// auto patch = std::make_unique<nanospline::RevolutionPatch<T, 3>>();
		// patch->set_profile(std::move(feature_edge));
		// patch->initialize();
		// return patch;
	}
	else if (type == "Extrusion")
	{
		std::array<double, 3> direction_vec =
			file.readDataset<std::array<double, 3>>(surface_path + "/direction");
		Eigen::Matrix<double, 1, 3> direction =
			Eigen::Matrix<double, 1, 3>::Map(direction_vec.data());
		// auto curve_path = surface_path + "/curve/";
		// std::vector<T> interval =
		//     file.readDataset<std::vector<T>>(curve_path + "/interval");

		// auto feature_edge = read_curve<T, 3>(file, curve_path);

		// auto patch = std::make_unique<nanospline::ExtrusionPatch<T, 3>>();
		// patch->set_profile(std::move(feature_edge));

		// patch->set_direction(direction);
		// patch->initialize();
		// return patch;
	}
	else if (type == "Offset")
	{
		// auto base_surface_path = surface_path + "/surface/";
		// auto base_surface = read_surface<T>(file, base_surface_path);
		// auto offset = file.readDataset<T>(surface_path + "/value");
		// auto offset_surface = std::make_shared<nanospline::OffsetPatch<T, 3>>();
		// offset_surface->set_base_surface(std::move(base_surface));
		// offset_surface->set_offset(offset);
		// offset_surface->initialize();
		// patch = offset_surface;
	}
	else
	{
		throw std::runtime_error("Unknown surface type");
	}
}

void read_parts(const std::string &path, std::vector<Part> &part_list)
{
	h5pp::File hdf5(path, h5pp::FileAccess::READONLY, 6);

	auto parts = hdf5.findGroups("part_", "parts", -1, 0);

	for (const auto &part : parts)
	{
		part_list.emplace_back();
		Part &current_part = part_list.back();
		// Geometry
		{
			{
				auto cruves2d = hdf5.findGroups("", "parts/" + part + "/geometry/2dcurves", -1, 0);
				current_part.curves2d.resize(cruves2d.size());
				for (const auto &curve2d : cruves2d)
				{
					int curve_id = std::stoi(curve2d);
					read_curve<2>(hdf5, "parts/" + part + "/geometry/2dcurves/" + curve2d, current_part.curves2d, curve_id);
				}
			}
			{
				auto cruves3d = hdf5.findGroups("", "parts/" + part + "/geometry/3dcurves", -1, 0);
				current_part.curves3d.resize(cruves3d.size());
				for (const auto &curve3d : cruves3d)
				{
					int curve_id = std::stoi(curve3d);
					read_curve<3>(hdf5, "parts/" + part + "/geometry/3dcurves/" + curve3d, current_part.curves3d, curve_id);
				}
			}
			{
				auto surfaces = hdf5.findGroups("", "parts/" + part + "/geometry/surfaces", -1, 0);
				current_part.surfaces.resize(surfaces.size());
				for (const auto &surface : surfaces)
				{
					int surface_id = std::stoi(surface);
					read_surface(hdf5, "parts/" + part + "/geometry/surfaces/" + surface, current_part.surfaces, surface_id);
				}
			}

			const auto vertices = hdf5.readDataset<std::vector<double>>("parts/" + part + "/geometry/vertices");
			current_part.vertices = Eigen::Matrix<double, Eigen::Dynamic, 3>::Map(vertices.data(), vertices.size() / 3, 3);
			const auto bbox = hdf5.readDataset<std::array<double, 6>>("parts/" + part + "/geometry/bbox");
			current_part.bbox = Eigen::Matrix<double, 2, 3>::Map(bbox.data());
		}

		// meshes
		{
			auto meshes = hdf5.findGroups("", "parts/" + part + "/mesh/", -1, 0);
			current_part.meshes.resize(meshes.size());
			for (const auto &mesh : meshes)
			{
				const int mesh_id = std::stoi(mesh);

				const auto points = hdf5.readDataset<std::vector<double>>("parts/" + part + "/mesh/" + mesh + "/points");
				current_part.meshes[mesh_id].first = Eigen::Matrix<double, Eigen::Dynamic, 3>::Map(points.data(), points.size() / 3, 3);

				const auto triangles = hdf5.readDataset<std::vector<int64_t>>("parts/" + part + "/mesh/" + mesh + "/triangle");
				current_part.meshes[mesh_id].second = Eigen::Matrix<int64_t, Eigen::Dynamic, 3>::Map(triangles.data(), triangles.size() / 3, 3);
			}
		}

		// Topology
		{
			{
				auto edges = hdf5.findGroups("", "parts/" + part + "/topology/edges", -1, 0);
				current_part.edges.resize(edges.size());
				for (const auto &e : edges)
				{
					const int edge_id = std::stoi(e);
					const std::string edge_path = "parts/" + part + "/topology/edges/" + e;
					current_part.edges[edge_id].start_vertex = hdf5.readDataset<int64_t>(edge_path + "/start_vertex");
					current_part.edges[edge_id].end_vertex = hdf5.readDataset<int64_t>(edge_path + "/end_vertex");
					current_part.edges[edge_id].curve_id = hdf5.readDataset<int64_t>(edge_path + "/3dcurve");
				}
			}

			{
				auto faces = hdf5.findGroups("", "parts/" + part + "/topology/faces", -1, 0);
				current_part.faces.resize(faces.size());
				for (const auto &e : faces)
				{
					const int face_id = std::stoi(e);
					const std::string face_path = "parts/" + part + "/topology/faces/" + e;
					current_part.faces[face_id].exact_domain = hdf5.readDataset<std::array<double, 4>>(face_path + "/exact_domain");
					current_part.faces[face_id].has_singularities = hdf5.readDataset<int64_t>(face_path + "/has_singularities");
					current_part.faces[face_id].loops = hdf5.readDataset<std::vector<int64_t>>(face_path + "/loops");
					current_part.faces[face_id].nr_singularities = hdf5.readDataset<int64_t>(face_path + "/nr_singularities");
					current_part.faces[face_id].outer_loop = hdf5.readDataset<int64_t>(face_path + "/outer_loop");
					current_part.faces[face_id].surface = hdf5.readDataset<int64_t>(face_path + "/surface");
					current_part.faces[face_id].surface_orientation = hdf5.readDataset<bool>(face_path + "/surface_orientation");
				}
			}

			{
				auto halfedges = hdf5.findGroups("", "parts/" + part + "/topology/halfedges", -1, 0);
				current_part.half_edges.resize(halfedges.size());
				for (const auto &e : halfedges)
				{
					const int edge_id = std::stoi(e);
					const std::string edge_path = "parts/" + part + "/topology/halfedges/" + e;
					current_part.half_edges[edge_id].curve_id = hdf5.readDataset<int64_t>(edge_path + "/2dcurve");
					current_part.half_edges[edge_id].edge = hdf5.readDataset<int64_t>(edge_path + "/edge");
					try
					{
						current_part.half_edges[edge_id].mates = hdf5.readDataset<std::vector<int64_t>>(edge_path + "/mates");
					}
					catch (std::runtime_error &)
					{
						current_part.half_edges[edge_id].mates = {};
					}
					current_part.half_edges[edge_id].orientation = hdf5.readDataset<bool>(edge_path + "/orientation_wrt_edge");
				}
			}

			{
				auto loops = hdf5.findGroups("", "parts/" + part + "/topology/loops", -1, 0);
				current_part.loops.resize(loops.size());
				for (const auto &e : loops)
				{
					const int edge_id = std::stoi(e);
					const std::string edge_path = "parts/" + part + "/topology/loops/" + e;
					current_part.loops[edge_id].half_edges = hdf5.readDataset<std::vector<int64_t>>(edge_path + "/halfedges");
				}
			}

			{
				auto shells = hdf5.findGroups("", "parts/" + part + "/topology/shells", -1, 0);
				current_part.shells.resize(shells.size());
				for (const auto &e : shells)
				{
					const int edge_id = std::stoi(e);
					const std::string edge_path = "parts/" + part + "/topology/shells/" + e;
					// current_part.shells[edge_id].faces = hdf5.readDataset<std::vector<int64_t>>(edge_path + "/faces");
					current_part.shells[edge_id].orientation = hdf5.readDataset<bool>(edge_path + "/orientation_wrt_solid");
				}
			}

			{
				auto solids = hdf5.findGroups("", "parts/" + part + "/topology/solids", -1, 0);
				current_part.solids.resize(solids.size());
				for (const auto &e : solids)
				{
					const int edge_id = std::stoi(e);
					const std::string edge_path = "parts/" + part + "/topology/solids/" + e;
					current_part.solids[edge_id].shells = hdf5.readDataset<std::vector<int64_t>>(edge_path + "/shells");
				}
			}
		}
	}
}
