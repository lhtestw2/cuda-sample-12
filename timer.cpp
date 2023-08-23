#include "timer.hpp"

#include <chrono>
#include <memory>
#include <sstream>
#include <string>


#include <unordered_map>
double duration_cast(const std::chrono::duration<double> &dur,
                     const Timer::Unit &unit) {
  switch (unit) {
  case Timer::Unit::kSecond:
    return std::chrono::duration_cast<std::chrono::seconds>(dur).count();
    break;
  case Timer::Unit::kMilliSecond:
    return std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    break;
  case Timer::Unit::kMicroSecond:
    return std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
    break;
  default:
    break;
  }
  return 0;
}
std::string duration_unit_2_str(Timer::Unit unit) {
  switch (unit) {
  case Timer::Unit::kSecond:
    return "sec";
    break;
  case Timer::Unit::kMilliSecond:
    return "ms";
    break;
  case Timer::Unit::kMicroSecond:
    return "us";
    break;
  default:
    break;
  }
  return std::string();
}

class Timer::Impl {
public:
  Impl() = default;
  std::unordered_map<std::string,
                     std::pair<decltype(std::chrono::steady_clock::now()),
                               std::chrono::duration<double>>>
      recorder;
  std::vector<std::string> log_order;
};

Timer::Timer() : impl_(std::make_unique<Timer::Impl>()) {}

Timer::~Timer() = default;
Timer &Timer::Global() {
  static Timer global_timer;
  return global_timer;
}

void Timer::start(const std::string &name) {
  if (impl_->recorder.find(name) == impl_->recorder.end()) {
    impl_->recorder[name] = {std::chrono::steady_clock::now(),
                             std::chrono::duration<double>::zero()};
    impl_->log_order.emplace_back(name);
  }
}

void Timer::log(const std::string &name) {
  if (impl_->recorder.find(name) != impl_->recorder.end()) {
    auto cur = std::chrono::steady_clock::now();
    auto &cur_pair = impl_->recorder[name];
    cur_pair.second += cur - cur_pair.first;
    cur_pair.first = cur;
  }
}

std::string Timer::to_string(const Timer::Unit &unit) const {
  std::ostringstream oss;
  const std::string unit_str = duration_unit_2_str(unit);
  const auto max_name_len = 40;

  for (const auto &log_name : impl_->log_order) {
    const auto &duration = impl_->recorder.at(log_name).second;
    std::string print_name = log_name;
    if (print_name.size() > max_name_len) {
      print_name.resize(max_name_len - strlen("..."));
      print_name.append("...");
    } else {
      print_name.append(max_name_len - print_name.size(), ' ');
    }
    oss << "[" << print_name << "] cost: " << duration_cast(duration, unit)
        << " " << unit_str << "\n";
  }
  return oss.str();
}
