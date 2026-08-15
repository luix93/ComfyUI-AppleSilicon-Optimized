#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace asfp8 {

// Strict source stamper shared by runtime-compiled Metal extensions. Tokens
// use the package's existing @NAME@ convention; misspellings fail before the
// much slower newLibraryWithSource call.
class MetalTemplate {
 public:
  explicit MetalTemplate(const char* source) : source_(source) {}

  MetalTemplate& set(const std::string& name, const std::string& value) {
    const std::string token = "@" + name + "@";
    size_t position = source_.find(token);
    if (position == std::string::npos)
      throw std::runtime_error("unused Metal template token " + token);
    do {
      source_.replace(position, token.size(), value);
      position = source_.find(token, position + value.size());
    } while (position != std::string::npos);
    return *this;
  }

  std::string finish() && {
    const size_t start = source_.find('@');
    if (start != std::string::npos) {
      const size_t stop = source_.find('@', start + 1);
      throw std::runtime_error(
          "unresolved Metal template token " +
          source_.substr(start, stop == std::string::npos
              ? std::string::npos : stop - start + 1));
    }
    return std::move(source_);
  }

 private:
  std::string source_;
};

}  // namespace asfp8
