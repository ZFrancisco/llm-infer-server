#include "../third_party/httplib/httplib.h"
#include "../third_party/json/json.hpp"
#include <chrono>
#include <thread>
#include <sstream>
#include <vector>

using json = nlohmann::json;

static std::vector<std::string> split_words(const std::string& s) {
  std::istringstream iss(s);
  std::vector<std::string> out;
  std::string w;
  while (iss >> w) out.push_back(w);
  return out;
}

static std::string sse(const json& j) {
  return "data: " + j.dump() + "\n\n";
}

int main() {
  httplib::Server svr;

  svr.Post("/v1/generate", [](const httplib::Request& req, httplib::Response& res) {
    json body = json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.contains("prompt")) {
      res.status = 400;
      res.set_content(R"({"error":"bad json or missing prompt"})", "application/json");
      return;
    }

    std::string prompt = body["prompt"].get<std::string>();
    int max_tokens = body.value("max_tokens", 32);
    bool stream = body.value("stream", true);

    // Non-streaming fallback (optional)
    if (!stream) {
      res.set_content(json({{"response", "You sent: " + prompt}}).dump(), "application/json");
      return;
    }

    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");

    res.set_chunked_content_provider(
      "text/event-stream",
      [prompt, max_tokens](size_t, httplib::DataSink& sink) {
        std::string fake = "Mock response to: " + prompt;
        auto toks = split_words(fake);

        int emitted = 0;
        for (const auto& t : toks) {
          if (emitted++ >= max_tokens) break;

          std::string msg = sse(json({{"token", t}}));
          sink.write(msg.data(), msg.size());

          std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }

        std::string done = sse(json({{"done", true}, {"tokens_out", emitted}}));
        sink.write(done.data(), done.size());
        sink.done();
        return true;
      },
      [](bool) {}
    );
  });

  svr.listen("0.0.0.0", 8080);
}
