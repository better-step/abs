#include <Eigen/Core>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nanospline/BSplinePatch.h>
#include <nanospline/NURBSPatch.h>

#include <algorithm>
#include <random>
#include <unordered_map>
#include <vector>

namespace
{

	// This file is part of libigl, a simple c++ geometry processing library.
	//
	// Copyright (C) 2020 Alec Jacobson <alecjacobson@gmail.com>
	//
	// This Source Code Form is subject to the terms of the Mozilla Public License
	// v. 2.0. If a copy of the MPL was not distributed with this file, You can
	// obtain one at http://mozilla.org/MPL/2.0/.

	inline void
	sortrows(const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> &X,
			 const bool ascending,
			 Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> &Y,
			 Eigen::VectorXi &IX)
	{
		// This is already 2x faster than matlab's builtin `sortrows`. I have tried
		// implementing a "multiple-pass" sort on each column, but see no performance
		// improvement.
		using namespace std;
		using namespace Eigen;
		// Resize output
		const size_t num_rows = X.rows();
		const size_t num_cols = X.cols();
		Y.resize(num_rows, num_cols);
		IX.resize(num_rows, 1);
		for (int i = 0; i < num_rows; i++)
		{
			IX(i) = i;
		}
		if (ascending)
		{
			auto index_less_than = [&X, num_cols](size_t i, size_t j) {
				for (size_t c = 0; c < num_cols; c++)
				{
					if (X.coeff(i, c) < X.coeff(j, c))
						return true;
					else if (X.coeff(j, c) < X.coeff(i, c))
						return false;
				}
				return false;
			};
			std::sort(IX.data(), IX.data() + IX.size(), index_less_than);
		}
		else
		{
			auto index_greater_than = [&X, num_cols](size_t i, size_t j) {
				for (size_t c = 0; c < num_cols; c++)
				{
					if (X.coeff(i, c) > X.coeff(j, c))
						return true;
					else if (X.coeff(j, c) > X.coeff(i, c))
						return false;
				}
				return false;
			};
			std::sort(IX.data(), IX.data() + IX.size(), index_greater_than);
		}
		for (size_t j = 0; j < num_cols; j++)
		{
			for (int i = 0; i < num_rows; i++)
			{
				Y(i, j) = X(IX(i), j);
			}
		}
	}

	using DEFAULT_URBG = std::mt19937;
	inline DEFAULT_URBG generate_default_urbg()
	{
		return DEFAULT_URBG(std::rand());
	}

	inline int64_t blue_noise_key(
		const int64_t w, // pass by copy --> int64_t so that multiplication is OK
		const int64_t x, // pass by copy --> int64_t so that multiplication is OK
		const int64_t y, // pass by copy --> int64_t so that multiplication is OK
		const int64_t z) // pass by copy --> int64_t so that multiplication is OK
	{
		return x + w * (y + w * z);
	}

	inline bool blue_noise_far_enough(
		const Eigen::MatrixXd &X,
		const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> &Xs,
		const std::unordered_map<int64_t, int> &S, const double rr, const int w,
		const int i)
	{
		const int xi = Xs(i, 0);
		const int yi = Xs(i, 1);
		const int zi = Xs(i, 2);
		int g = 2; // ceil(r/s)
		for (int x = std::max(xi - g, 0); x <= std::min(xi + g, w - 1); x++)
			for (int y = std::max(yi - g, 0); y <= std::min(yi + g, w - 1); y++)
				for (int z = std::max(zi - g, 0); z <= std::min(zi + g, w - 1); z++)
				{
					if (x != xi || y != yi || z != zi)
					{
						const int64_t nk = blue_noise_key(w, x, y, z);
						// have already selected from this cell
						const auto Siter = S.find(nk);
						if (Siter != S.end() && Siter->second >= 0)
						{
							const int ni = Siter->second;
							// too close
							if ((X.row(i) - X.row(ni)).squaredNorm() < rr)
							{
								return false;
							}
						}
					}
				}
		return true;
	}

	inline bool
	activate(const Eigen::MatrixXd &X,
			 const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> &Xs,
			 const double rr, const int i, const int w, const int64_t nk,
			 std::unordered_map<int64_t, std::vector<int>> &M,
			 std::unordered_map<int64_t, int> &S, std::vector<int> &active)
	{
		assert(M.count(nk));
		auto &Mvec = M.find(nk)->second;
		auto miter = Mvec.begin();
		while (miter != Mvec.end())
		{
			const int mi = *miter;
			// mi is our candidate sample. Is it far enough from all existing
			// samples?
			if (i >= 0 && (X.row(i) - X.row(mi)).squaredNorm() > 4. * rr)
			{
				// too far skip (reject)
				miter++;
			}
			else if (blue_noise_far_enough(X, Xs, S, rr, w, mi))
			{
				active.push_back(mi);
				S.find(nk)->second = mi;
				// printf("  found %d\n",mi);
				return true;
			}
			else
			{
				// remove forever (instead of incrementing we swap and eat from the
				// back)
				// std::swap(*miter,Mvec.back());
				*miter = Mvec.back();
				bool was_last = (std::next(miter) == Mvec.end());
				Mvec.pop_back();
				if (was_last)
				{
					// popping from the vector can invalidate the iterator, if it was
					// pointing to the last element that was popped. Alternatively,
					// one could use indices directly...
					miter = Mvec.end();
				}
			}
		}
		return false;
	}

	inline bool
	step(const Eigen::MatrixXd &X,
		 const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> &Xs,
		 const double rr, const int w, DEFAULT_URBG &urbg,
		 std::unordered_map<int64_t, std::vector<int>> &M,
		 std::unordered_map<int64_t, int> &S, std::vector<int> &active,
		 std::vector<int> &collected)
	{
		// considered.clear();
		if (active.size() == 0)
			return false;
		// random entry
		std::uniform_int_distribution<> dis(0, active.size() - 1);
		const int e = dis(urbg);
		const int i = active[e];
		// printf("%d\n",i);
		const int xi = Xs(i, 0);
		const int yi = Xs(i, 1);
		const int zi = Xs(i, 2);
		// printf("%d %d %d - %g %g %g\n",xi,yi,zi,X(i,0),X(i,1),X(i,2));
		//  cell indices of neighbors
		int g = 2; // ceil(r/s)
		std::vector<int64_t> N;
		const auto est_size = 1 + g * 1;
		N.reserve(est_size * est_size * est_size - 1);
		for (int x = std::max(xi - g, 0); x <= std::min(xi + g, w - 1); x++)
			for (int y = std::max(yi - g, 0); y <= std::min(yi + g, w - 1); y++)
				for (int z = std::max(zi - g, 0); z <= std::min(zi + g, w - 1); z++)
				{
					if (x != xi || y != yi || z != zi)
					{
						// printf("  %d %d %d\n",x,y,z);
						const int64_t nk = blue_noise_key(w, x, y, z);
						// haven't yet selected from this cell?
						const auto Siter = S.find(nk);
						if (Siter != S.end() && Siter->second < 0)
						{
							assert(M.find(nk) != M.end());
							N.emplace_back(nk);
						}
					}
				}
		// printf("  --------\n");
		// randomize order: this might be a little paranoid...
		std::shuffle(std::begin(N), std::end(N), urbg);
		bool found = false;
		for (const int64_t &nk : N)
		{
			assert(M.find(nk) != M.end());
			if (activate(X, Xs, rr, i, w, nk, M, S, active))
			{
				found = true;
				break;
			}
		}
		if (!found)
		{
			// remove i from active list
			// https://stackoverflow.com/a/60765833/148668
			collected.push_back(i);
			// printf("  before: "); for(const int j : active) { printf("%d ",j); }
			// printf("\n");
			std::swap(active[e], active.back());
			// printf("  after : "); for(const int j : active) { printf("%d ",j); }
			// printf("\n");
			active.pop_back();
			// printf("  removed %d\n",i);
		}
		// printf("  active: "); for(const int j : active) { printf("%d ",j); }
		// printf("\n");
		return true;
	}

	void blue_noise_downsample(const Eigen::MatrixXd &X, const double r,
							   Eigen::VectorXi &XI, DEFAULT_URBG &&urbg)
	{
		assert(X.cols() == 3 && "Only 3D embeddings allowed");

		// minimum radius
		const double min_r = r;
		// cell size based on 3D distance
		// It works reasonably well (but is probably biased to use s=2*r/√3 here and
		// g=1 in the outer loop below.
		//
		// One thing to try would be to store a list in S (rather than a single point)
		// or equivalently a mask over M and just use M as a generic spatial hash
		// (with arbitrary size) and then tune its size (being careful to make g a
		// function of r and s; and removing the `if S=-1 checks`)
		const double s = r / sqrt(3.0);

		// Make a uniform random sampling with 30*expected_number_of_points.
		const int nx = X.rows();

		// Rescale so that s = 1
		Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> Xs =
			((X.rowwise() - X.colwise().minCoeff()) / s).template cast<int>();
		const int w = Xs.maxCoeff() + 1;
		Eigen::VectorXi SortIdx;
		Eigen::MatrixXd Xsorted;
		{
			sortrows(decltype(Xs)(Xs), true, Xs, SortIdx);
			Xsorted = X(SortIdx, Eigen::all).eval();
		}
		// Initialization
		std::unordered_map<int64_t, std::vector<int>> M;
		std::unordered_map<int64_t, int> S;
		// attempted to seed
		// std::unordered_map<int64_t, int> A;
		// Q: Too many?
		// A: Seems to help though.
		M.reserve(Xs.rows());
		S.reserve(Xs.rows());
		for (int i = 0; i < Xs.rows(); i++)
		{
			int64_t k = blue_noise_key(w, Xs(i, 0), Xs(i, 1), Xs(i, 2));
			const auto Miter = M.find(k);
			if (Miter == M.end())
			{
				M.insert({k, {i}});
			}
			else
			{
				Miter->second.push_back(i);
			}
			S.emplace(k, -1);
			// A.emplace(k, false);
		}

		std::vector<int> active;
		active.reserve(nx);
		// precompute r²
		// Q: is this necessary?
		const double rr = r * r;
		std::vector<int> collected;
		collected.reserve(nx);

		auto Mouter = M.begin();
		// Just take the first point as the initial seed
		const auto initialize = [&]() -> bool {
			while (true)
			{
				if (Mouter == M.end())
				{
					return false;
				}
				const int64_t k = Mouter->first;
				// Haven't placed in this cell yet
				if (S[k] < 0)
				{
					if (activate(Xsorted, Xs, rr, -1, w, k, M, S, active))
					{
						return true;
					}
				}
				Mouter++;
			}
			assert(false && "should not be reachable.");
		};

		// important if mesh contains many connected components
		while (initialize())
		{
			while (active.size() > 0)
			{
				step(Xsorted, Xs, rr, w, urbg, M, S, active, collected);
			}
		}

		{
			const int n = collected.size();
			XI.resize(n);
			for (int i = 0; i < n; i++)
			{
				const int c = collected[i];
				XI(i) = SortIdx[c];
				//   P.row(i) = X.row(c).template cast<typename DerivedP::Scalar>();
			}
		}

	} // namespace

	// Downsample a point set so that samples are approximately evenly spaced.

	// Args:
	//     v: \#v by 3 array of vertex positions
	//     radius: desired separation between points.
	//     target_num_samples: If set to a positive value, iterate to generate
	//     points as close to this target as possible (determined by
	//     sample_num_tolerance). random_seed: A random seed used to generate the
	//     samples. Passing in 0 will use the current time. (0 by default).
	//     sample_num_tolerance: If you requested a target number of samples, by
	//     passsing num_samples > 0, then this function will return between (1 -
	//     sample_num_tolerance) * num_samples and (1 + sample_num_tolerance) *
	//     num_samples. Setting a very small value for this parameter will increase
	//     convergence time. (0.04 by default).

	// Returns:
	//     p_idx : A (m,) shaped array of indices into v where m is the number of
	//     Poisson-disk samples
	Eigen::VectorXi poisson_disk_downsample(const Eigen::MatrixXd &v,
											int target_num_samples,
											uint64_t random_seed,
											double sample_num_tolerance)
	{

		if (target_num_samples <= 0)
			throw pybind11::value_error(
				"Cannot have both num_samples <= 0 and radius <= 0");

		if (sample_num_tolerance > 1.0 || sample_num_tolerance <= 0.0)
			throw pybind11::value_error("sample_num_tolerance must be in (0, 1]");

		if (random_seed != 0)
			srand(random_seed);

		Eigen::VectorXi ret_i;

		if (target_num_samples >= v.rows())
		{
			ret_i.resize(v.rows());
			for (int i = 0; i < ret_i.size(); ++i)
			{
				ret_i(i) = i;
			}
			return ret_i;
		}

		const size_t num_samples_min =
			int(double(target_num_samples) * (1 - sample_num_tolerance));
		const size_t num_samples_max =
			int(double(target_num_samples) * (1 + sample_num_tolerance));

		const Eigen::Vector3d bmin = v.colwise().minCoeff();
		const Eigen::Vector3d bmax = v.colwise().maxCoeff();
		const double bbsize = (bmax - bmin).norm();
		double range_min_rad = bbsize / 50.0;
		double range_max_rad = bbsize / 50.0;
		size_t range_min_rad_num = -1;
		size_t range_max_rad_num = -1;

		do
		{
			ret_i.conservativeResize(0);
			range_min_rad /= 2.0;
			blue_noise_downsample(v, range_min_rad, ret_i, generate_default_urbg());
			range_min_rad_num = ret_i.size();
		} while (range_min_rad_num < target_num_samples);

		do
		{
			ret_i.conservativeResize(0);
			range_max_rad *= 2.0;
			blue_noise_downsample(v, range_max_rad, ret_i, generate_default_urbg());
			range_max_rad_num = ret_i.size();
		} while (range_max_rad_num > target_num_samples);

		double current_rad = range_max_rad;
		int iter_count = 0;
		while (iter_count < 20 && (ret_i.size() < num_samples_min || ret_i.size() > num_samples_max))
		{
			iter_count += 1;
			ret_i.conservativeResize(0);
			current_rad = (range_min_rad + range_max_rad) / 2.0;
			blue_noise_downsample(v, current_rad, ret_i, generate_default_urbg());
			if (ret_i.size() > target_num_samples)
			{
				range_min_rad = current_rad;
				range_min_rad_num = ret_i.size();
			}
			if (ret_i.size() < target_num_samples)
			{
				range_max_rad = current_rad;
				range_max_rad_num = ret_i.size();
			}
		}

		return ret_i;
	}
} // namespace

class BSpline
{
public:
	using Vect1 = Eigen::Matrix<double, Eigen::Dynamic, 1>;
	using Vect2 = Eigen::Matrix<double, Eigen::Dynamic, 2>;
	using Vect3 = Eigen::Matrix<double, Eigen::Dynamic, 3>;

	BSpline(int degree_u, int degree_v,
			bool u_rational, bool v_rational,
			const Vect1 &u_knots, const Vect1 &v_knots,
			const Vect3 &grid,
			const Vect1 &weights,
			bool u_periodic, bool v_periodic)
	{
		if (u_rational || v_rational)
		{
			if (degree_u == 3 && degree_v == 3)
			{
				auto patch = std::make_unique<nanospline::NURBSPatch<double, 3, 3, 3>>();
				patch->set_knots_u(u_knots);
				patch->set_knots_v(v_knots);
				patch->set_control_grid(grid);
				patch->set_periodic_u(u_periodic);
				patch->set_periodic_v(v_periodic);
				patch->set_degree_u(degree_u);
				patch->set_degree_v(degree_v);
				patch->set_weights(weights);
				patch->initialize();

				this->patch = std::move(patch);
			}
			else
			{
				auto patch = std::make_unique<nanospline::NURBSPatch<double, 3, -1, -1>>();
				patch->set_knots_u(u_knots);
				patch->set_knots_v(v_knots);
				patch->set_control_grid(grid);
				patch->set_periodic_u(u_periodic);
				patch->set_periodic_v(v_periodic);
				patch->set_degree_u(degree_u);
				patch->set_degree_v(degree_v);
				patch->set_weights(weights);
				patch->initialize();

				this->patch = std::move(patch);
			}
		}
		else
		{
			if (degree_u == 3 && degree_v == 3)
			{
				auto patch = std::make_unique<nanospline::BSplinePatch<double, 3, 3, 3>>();
				patch->set_degree_u(degree_u);
				patch->set_degree_v(degree_v);
				patch->set_knots_u(u_knots);
				patch->set_knots_v(v_knots);
				patch->set_control_grid(grid);
				patch->set_periodic_u(u_periodic);
				patch->set_periodic_v(v_periodic);
				patch->initialize();

				this->patch = std::move(patch);
			}
			else
			{
				auto patch = std::make_unique<nanospline::BSplinePatch<double, 3, -1, -1>>();
				patch->set_degree_u(degree_u);
				patch->set_degree_v(degree_v);
				patch->set_knots_u(u_knots);
				patch->set_knots_v(v_knots);
				patch->set_control_grid(grid);
				patch->set_periodic_u(u_periodic);
				patch->set_periodic_v(v_periodic);
				patch->initialize();

				this->patch = std::move(patch);
			}
		}
	}

	Vect3 sample(const Vect2 &params) const
	{
		Vect3 samples(params.rows(), 3);
		for (int i = 0; i < params.rows(); ++i)
		{
			samples.row(i) = patch->evaluate(params(i, 0), params(i, 1));
		}
		return samples;
	}

	std::tuple<Vect3, Vect3> first_derivative(const Vect2 &params) const
	{
		Vect3 du(params.rows(), 3);
		Vect3 dv(params.rows(), 3);

		for (int i = 0; i < params.rows(); ++i)
		{
			du.row(i) = patch->evaluate_derivative_u(params(i, 0), params(i, 1));
			dv.row(i) = patch->evaluate_derivative_v(params(i, 0), params(i, 1));
		}
		return std::make_tuple(du, dv);
	}

	std::tuple<Vect3, Vect3, Vect3> second_derivative(const Vect2 &params) const
	{
		Vect3 duu(params.rows(), 3);
		Vect3 dvv(params.rows(), 3);
		Vect3 duv(params.rows(), 3);
		for (int i = 0; i < params.rows(); ++i)
		{
			duu.row(i) = patch->evaluate_2nd_derivative_uu(params(i, 0), params(i, 1));
			dvv.row(i) = patch->evaluate_2nd_derivative_vv(params(i, 0), params(i, 1));
			duv.row(i) = patch->evaluate_2nd_derivative_uv(params(i, 0), params(i, 1));
		}
		return std::make_tuple(duu, dvv, duv);
	}

private:
	std::unique_ptr<nanospline::PatchBase<double, 3>> patch;
};

PYBIND11_MODULE(abspy, m)
{
	namespace py = pybind11;

	m.doc() = "Poisson disk downsample";

	m.def(
		"poisson_disk_downsample",
		[](const Eigen::MatrixXd &v, int target_num_samples, uint64_t random_seed,
		   double sample_num_tolerance) {
			return poisson_disk_downsample(v, target_num_samples, random_seed,
										   sample_num_tolerance);
		},
		"A (m,) shaped array of indices into v where m is the number of "
		"Poisson-disk samples",
		py::arg("v"), py::arg("target_num_samples"), py::arg("random_seed") = 0,
		py::arg("sample_num_tolerance") = 0.04);

	py::class_<BSpline>(m, "BSpline")
		.def(py::init<int, int, bool, bool,
					  const BSpline::Vect1 &,
					  const BSpline::Vect1 &,
					  const BSpline::Vect3 &,
					  const BSpline::Vect1 &,
					  bool, bool>(),
			 py::arg("degree_u"), py::arg("degree_v"),
			 py::arg("u_rational"), py::arg("v_rational"),
			 py::arg("u_knots"), py::arg("v_knots"),
			 py::arg("grid"),
			 py::arg("weights"),
			 py::arg("u_periodic"), py::arg("v_periodic"))
		.def("sample", &BSpline::sample, "Sample points on the surface given parameter values", py::arg("params"))
		.def("first_derivative", &BSpline::first_derivative, "Sample points on the derivative surface given parameter values", py::arg("params"))
		.def("second_derivative", &BSpline::second_derivative, "Sample points on the second derivative surface given parameter values", py::arg("params"));
}
