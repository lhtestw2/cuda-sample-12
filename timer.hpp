#pragma once

#include <memory>
#include <string>

class Timer {
public:
  enum Unit {
    kSecond,
    kMilliSecond,
    kMicroSecond,
  };

  static Timer &Global();

  ~Timer();
  void start(const std::string &name);
  void log(const std::string &name);
  [[nodiscard]] std::string to_string(const Unit &unit = Unit::kSecond) const;

private:
  Timer();
  class Impl;
  std::unique_ptr<Impl> impl_;
};
