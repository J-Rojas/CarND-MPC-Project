#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];
	        double throttle = j[1]["throttle"];
	        double steering = j[1]["steering_angle"];

          /*
          * TODO: Calculate steering angle and throttle using MPC.
          *
          * Both are in between [-1, 1].
          *
          */
          double steer_value = 0;
          double throttle_value = 0.1;
          double n_waypoints = 100;
	        int n_poly_degree = 3;
          double dist_per_waypoint = 5;
          json msgJson;

          //Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line

	        // the waypoints are transformed wrt to the vehicle's coordinate system
          for (int i = 0; i < ptsx.size(); i++) {
            auto x = ptsx[i] - px;
            auto y = ptsy[i] - py;
            ptsx[i] = cos(-psi) * x - sin(-psi) * y;
            ptsy[i] = sin(-psi) * x + cos(-psi) * y;
          }

          Eigen::VectorXd xIntp = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(ptsx.data(), ptsx.size());
          Eigen::VectorXd yIntp = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(ptsy.data(), ptsy.size());
          Eigen::VectorXd polyWaypoints = polyfit(xIntp, yIntp, n_poly_degree);

          for (int i = 0; i < n_waypoints; i++) {
            auto x = (i - (n_waypoints / 2)) * dist_per_waypoint;
            next_x_vals.push_back(x);
            next_y_vals.push_back(polyeval(polyWaypoints, x));
          }

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

	        // the initial cte is the value of f(0) for the waypoint polynomial equation
	        auto cte = polyeval(polyWaypoints, 0);
          auto deriv = polyderv(polyWaypoints);

          //std::cout << "Poly waypoints: " << polyWaypoints << std::endl;
          //std::cout << "Poly deriv: " << deriv << std::endl;

	        // the initial epsi is the arctan(f'(0)) for the waypoint polynomial equation
	        auto epsi = atan(polyeval(deriv, 0));

	        Eigen::VectorXd state(8);
          //the waypoints have been converted to local vehicle coordinates so
          // x = 0, y = 0, and psi is 0 because it represents the current vehicle heading
	        state << 0, 0, 0, v, cte, epsi, steering, throttle;

          std::cout << "State: " << state << std::endl;

	        auto solution = mpc.Solve(state, polyWaypoints);

          //Display the MPC predicted trajectory
          msgJson["mpc_x"] = solution.x;
          msgJson["mpc_y"] = solution.y;

	        // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
	        // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
	        if (solution.status) {
		        steer_value = solution.steering / deg2rad(25);
		        throttle_value = solution.throttle;
		        std::cout << "steer_value: " << steer_value << std::endl;
		        std::cout << "throttle_value: " << throttle_value << std::endl;
	        } else {
		        steer_value = 0;
		        throttle_value = 0;
		        std::cout << "invalid solution " << std::endl;
	        }


	        msgJson["steering_angle"] = steer_value * -1; //steering is inverted
	        msgJson["throttle"] = throttle_value;

	        auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
