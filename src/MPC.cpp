#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"

using CppAD::AD;

// I choose N to be fairly low to avoid convergence problems with the linear equation solver. The higher the N value, the
// higher the unknown variables that will need to be solved. Also, it's not useful to have a high N since the environment
// could change within a short period of time.
//
// dt was set to 0.25 to give the model enough flexible to adjust actuators quickly but also high enough to model some
// of the latency period. By setting dt > 2 * the latency, the model will partially compensate for the state of the car
// during the latency period.
//
// See below for more unsuccessful attempts at modeling latency.
//
// I tried higher values such as N=25 and lower values of dt = 0.1. The higher N values cause convergence problems with
// the linear system solver - even with constraints in place, the solver would diverge and produce non-optimal solution.
// So instead I kept the N low along with the speed to maintain good performance.
// I found these number below to be adequate for modeling turns at a reasonable speed (< 40 mph).
size_t N = 5;
double dt = 0.25;
int latency_steps = 0;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// Both the reference cross track and orientation errors are 0.
// The reference velocity is set to 30 mph.
double ref_v = 30;

// The solver takes all the state variables and actuator
// variables in a singular vector. Thus, we should to establish
// when one variable starts and another ends to make our lifes easier.
size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
	double result = 0.0;
	for (int i = 0; i < coeffs.size(); i++) {
		result += coeffs[i] * pow(x, i);
	}
	return result;
}

AD<double> polyevalAD(Eigen::VectorXd coeffs, AD<double> x) {
	AD<double> result = 0.0;
	for (int i = 0; i < coeffs.size(); i++) {
		result += coeffs[i] * CppAD::pow(x, i);
	}
	return result;
}

// Generate the derivative coefficients of a polynomial.
Eigen::VectorXd polyderv(Eigen::VectorXd coeffs) {
	Eigen::VectorXd dCoeffs(coeffs.size() - 1);
	for (int i = 1; i < coeffs.size(); i++) {
		dCoeffs(i-1) = coeffs[i] * i;
	}
	return dCoeffs;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
	assert(xvals.size() == yvals.size());
	assert(order >= 1 && order <= xvals.size() - 1);
	Eigen::MatrixXd A(xvals.size(), order + 1);

	for (int i = 0; i < xvals.size(); i++) {
		A(i, 0) = 1.0;
	}

	for (int j = 0; j < xvals.size(); j++) {
		for (int i = 0; i < order; i++) {
			A(j, i + 1) = A(j, i) * xvals(j);
		}
	}

	auto Q = A.householderQr();
	auto result = Q.solve(yvals);
	return result;
}

class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
	Eigen::VectorXd deriv;
  FG_eval(Eigen::VectorXd coeffs) {
		this->coeffs = coeffs;
		this->deriv = polyderv(coeffs);
	  std::cout << "Polynomial coeffs: " << this->coeffs << std::endl;
	  std::cout << "Polynomial deriv: " << this->deriv << std::endl;
	}

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    // The cost is stored is the first element of `fg`.
    // Any additions to the cost should be added to `fg[0]`.
    fg[0] = 0;

    // The part of the cost based on the reference state.
    for (int t = 0; t < N; t++) {
      fg[0] += CppAD::pow(vars[cte_start + t], 2);
      fg[0] += CppAD::pow(vars[epsi_start + t], 2);
      fg[0] += CppAD::pow(vars[v_start + t] - ref_v, 2);
    }

    // Minimize the use of actuators.
    for (int t = 0; t < N - 1; t++) {
      fg[0] += 10 * CppAD::pow(vars[delta_start + t], 2);
      fg[0] += CppAD::pow(vars[a_start + t], 2);
    }

    // Minimize the value gap between sequential actuations.
    for (int t = 0; t < N - 2; t++) {
      fg[0] += 10 * CppAD::pow((vars[delta_start + t + 1] - vars[delta_start + t]), 2);
      fg[0] += 10 * CppAD::pow((vars[a_start + t + 1] - vars[a_start + t]), 2);
    }

    //
    // Setup Constraints
    //
    // NOTE: In this section you'll setup the model constraints.

    // Initial constraints
    //
    // We add 1 to each of the starting indices due to cost being located at
    // index 0 of `fg`.
    // This bumps up the position of all the other values.
    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    // The rest of the constraints
    for (int t = 1; t < N; t++) {
      // The state at time t+1 .
      AD<double> x1 = vars[x_start + t];
      AD<double> y1 = vars[y_start + t];
      AD<double> psi1 = vars[psi_start + t];
      AD<double> v1 = vars[v_start + t];
      AD<double> cte1 = vars[cte_start + t];
      AD<double> epsi1 = vars[epsi_start + t];

      // The state at time t.
      AD<double> x0 = vars[x_start + t - 1];
      AD<double> y0 = vars[y_start + t - 1];
      AD<double> psi0 = vars[psi_start + t - 1];
      AD<double> v0 = vars[v_start + t - 1];
      AD<double> cte0 = vars[cte_start + t - 1];
      AD<double> epsi0 = vars[epsi_start + t - 1];

      // Only consider the actuation at time t.
      AD<double> delta0 = vars[delta_start + t - 1];
      AD<double> a0 = vars[a_start + t - 1];

      AD<double> f0 = polyevalAD(this->coeffs, x0);
      AD<double> psides0 = polyevalAD(this->deriv, x0);

      // Here are the update equations for the model.
	    // These equations account for the change in position, orienation, velocity, cte and epsi after each timestep.

      // x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
      // y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
      // psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
      // v_[t+1] = v[t] + a[t] * dt
      // cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
      // epsi[t+1] = psi[t] - psides[t] + v[t] * delta[t] / Lf * dt
      fg[1 + x_start + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[1 + y_start + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[1 + psi_start + t] = psi1 - (psi0 + v0 * delta0 / Lf * dt);
      fg[1 + v_start + t] = v1 - (v0 + a0 * dt);
      fg[1 + cte_start + t] =
              cte1 - ((f0 - y0)); //I found the cte change term here to cause unnecesary shifts in the model, so I eliminated it.
      fg[1 + epsi_start + t] =
              epsi1 - ((psi0 - psides0) + v0 * delta0 / Lf * dt);
    }
  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

Solution MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  double x = state[0];
  double y = state[1];
  double psi = state[2];
  double v = state[3];
  double cte = state[4];
  double epsi = state[5];
	double steering = state[6];
	double throttle = state[7];

  // number of independent variables
  // N timesteps == N - 1 actuations
  size_t n_vars = N * 6 + (N - 1) * 2;
  // Number of constraints
  size_t n_constraints = N * 6;

  // Initial value of the independent variables.
  // Should be 0 except for the initial values.
  Dvector vars(n_vars);
  for (i = 0; i < n_vars; i++) {
    vars[i] = 0.0;
  }
  // Set the initial variable values

	// My model uses these variables:
	// x, y for position
	// psi for orientation angle
	// v for velocity
	// cte for the cross track error
	// epsi for the orientation angle error
	// delta for the orientation angle change per time step
	// a for the change in velocity per time step

	vars[x_start] = x;
	vars[y_start] = y;
	vars[psi_start] = psi;
	vars[v_start] = v;
	vars[cte_start] = cte;
	vars[epsi_start] = epsi;
	vars[delta_start] = steering;
	vars[a_start] = throttle;

	// I attempted to model latency by constraining the initial state by X number of time steps (depending on the dt setting)/
	// Unforunately this didn't appear to work properly. I choose a simpler approach by making the dt value larger than the latency period.
	// Having a high dt will not work well for high speeds, so I would have to get the code below to work to attempt high
	// speed MPC.
	auto derivCoeffs = polyderv(coeffs);
	for (i = 0; i < latency_steps; i++) {
		vars[x_start] += vars[v_start] * cos(vars[psi_start]) * dt;
		vars[y_start] += vars[v_start] * sin(vars[psi_start]) * dt;
		vars[psi_start] += vars[v_start] * steering / Lf * dt;
		vars[v_start] += throttle * dt;
		vars[cte_start] = polyeval(coeffs, vars[x_start]) - vars[y_start];
		vars[epsi_start] = (psi - polyeval(derivCoeffs, vars[x_start])) + vars[v_start] * steering / Lf * dt;
	}

  // Lower and upper limits for x
  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);

  // Set all non-actuators upper and lowerlimits
  // to the max negative and positive values.
  for (i = x_start; i < psi_start; i++) {
    vars_lowerbound[i] = -1000;
    vars_upperbound[i] = 1000;
  }
	for (i = psi_start; i < v_start; i++) {
		vars_lowerbound[i] = deg2rad(-180);
		vars_upperbound[i] = deg2rad(180);
	}
	for (i = v_start; i < cte_start; i++) {
		vars_lowerbound[i] = -100;
		vars_upperbound[i] = 100;
	}
	for (i = cte_start; i < epsi_start; i++) {
		vars_lowerbound[i] = -100;
		vars_upperbound[i] = 100;
	}
	for (i = epsi_start; i < delta_start; i++) {
		vars_lowerbound[i] = deg2rad(-180);
		vars_upperbound[i] = deg2rad(180);
	}

  // The upper and lower limits of delta are set to -25 and 25
  // degrees (values in radians).
  // NOTE: Feel free to change this to something else.
  for (i = delta_start; i < a_start; i++) {
    vars_lowerbound[i] = -deg2rad(18);
    vars_upperbound[i] = deg2rad(18);
  }

  // Acceleration/decceleration upper and lower limits.
  // NOTE: Feel free to change this to something else.
  for (i = a_start; i < n_vars; i++) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = 1.0;
  }

  // Lower and upper limits for constraints
  // All of these should be 0 except the initial
  // state indices.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }
  constraints_lowerbound[x_start] = x;
  constraints_lowerbound[y_start] = y;
  constraints_lowerbound[psi_start] = psi;
  constraints_lowerbound[v_start] = v;
  constraints_lowerbound[cte_start] = cte;
  constraints_lowerbound[epsi_start] = epsi;

  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = cte;
  constraints_upperbound[epsi_start] = epsi;

  // Object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
	options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.25\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  bool ok = solution.status == CppAD::ipopt::solve_result<Dvector>::success;

	// Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  // TODO: Return the first actuator values. The variables can be accessed with
  // `solution.x[i]`.
  //
  // {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
  // creates a 2 element double vector.

	Solution retval;
	retval.status = ok;
	retval.steering = solution.x[delta_start];
	retval.throttle = solution.x[a_start];

	for (i = 0; i < N; i++) {
		retval.x.push_back(solution.x[x_start + i]);
		retval.y.push_back(solution.x[y_start + i]);
	}

	if (retval.steering > deg2rad(15) || retval.steering < deg2rad(-15)) {
		printf("SHARP TURN!!!\n");
	}

  return retval;
}
