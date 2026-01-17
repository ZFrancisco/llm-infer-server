// main.cpp
#include "../third_party/httplib/httplib.h"
#include "../third_party/json/json.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

//observability
int queue_depth = 0;
int time_to_first_token = 0;
int tokens_per_second = 0;
int active_streams = 0;

// -functs
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

// job struct
struct Job {
  std::string prompt;
  int max_tokens = 32;

  //token state
  std::vector<std::string> toks;
  size_t idx = 0;
  int tokens_out = 0;
  bool initialized = false;  

  std::mutex m;
  std::condition_variable cv;
  std::deque<std::string> out; // SSE-formatted chunks
  bool done = false;
};

// backend
class Backend {
public:
  virtual void generate(const std::string& prompt,
                        int max_tokens,
                        const std::function<void(const std::string&)>& on_token,
                        const std::function<void(int /*tokens_out*/)> &on_done) = 0;
  virtual ~Backend() = default;
};

class MockBackend : public Backend {
public:
  void generate(const std::string& prompt,
                int max_tokens,
                const std::function<void(const std::string&)>& on_token,
                const std::function<void(int)>& on_done) override {
    auto toks = split_words(prompt);
    int emitted = 0;
    for (const auto& t : toks) {
      if (emitted >= max_tokens) break;
      on_token(t);
      emitted++;
      std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
    on_done(emitted);
  }
};

// global job_q
static std::queue<std::shared_ptr<Job>> job_queue;
static std::mutex job_queue_m;
static std::condition_variable job_queue_cv;

static std::atomic<bool> running{true};

// --- worker thread: pop jobs, call backend, push SSE chunks into job->out ---
static void worker_loop(Backend& backend) {
  while (running.load()) {
    std::shared_ptr<Job> job;

    // wait for a job
    {
      std::unique_lock<std::mutex> lk(job_queue_m);
      job_queue_cv.wait(lk, [] { return !job_queue.empty() || !running.load(); });
      if (!running.load()) break;

      job = job_queue.front();
      job_queue.pop();
    }

    // generate tokens fake-asyncly
    backend.generate(
      job->prompt,
      job->max_tokens,

      [job](const std::string& tok) {
        std::string msg = sse(json({{"token", tok}}));
        {
          std::lock_guard<std::mutex> lk(job->m);
          job->out.push_back(std::move(msg));
        }
        job->cv.notify_one();
      },

      [job](int tokens_out) {
        std::string done_msg = sse(json({{"done", true}, {"tokens_out", tokens_out}}));
        {
          std::lock_guard<std::mutex> lk(job->m);
          job->out.push_back(std::move(done_msg));
          job->done = true;
        }
        job->cv.notify_all();
      }
    );
  }
}

int main() {
  httplib::Server svr;

  MockBackend backend;
  std::thread worker([&] { worker_loop(backend); });

  svr.Post("/v1/generate", [](const httplib::Request& req, httplib::Response& res) {
    json body = json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.contains("prompt") || !body["prompt"].is_string()) {
      res.status = 400;
      res.set_content(R"({"error":"bad json or missing prompt"})", "application/json");
      return;
    }

    const std::string prompt = body["prompt"].get<std::string>();
    const int max_tokens = body.value("max_tokens", 32);
    const bool stream = body.value("stream", true);

    if (!stream) {
      // simple non-streaming echo (no queue)
      res.set_content(json({{"response", "You sent: " + prompt}}).dump(), "application/json");
      return;
    }

    // create + enqueue job
    auto job = std::make_shared<Job>();
    job->prompt = prompt;
    job->max_tokens = max_tokens;

    {
      std::lock_guard<std::mutex> lk(job_queue_m);
      job_queue.push(job);
    }
    job_queue_cv.notify_one();

    // SSE headers
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");

    // stream from job->out
    res.set_chunked_content_provider(
      "text/event-stream",
      [job](size_t, httplib::DataSink& sink) {
        while (true) {
          std::deque<std::string> to_send;
          bool done = false;

          // wait for output or done
          {
            std::unique_lock<std::mutex> lk(job->m);
            job->cv.wait(lk, [&] { return !job->out.empty() || job->done; });

            // move all available messages out in one shot
            while (!job->out.empty()) {
              to_send.push_back(std::move(job->out.front()));
              job->out.pop_front();
            }
            done = job->done;
          }

          // send everything we grabbed
          for (auto& msg : to_send) {
            sink.write(msg.data(), msg.size());
          }

          // if done and nothing left to send, close stream
          if (done) {
            sink.done();
            return true;
          }
        }
      },
      [](bool) {}
    );
  });

  svr.Get("/metrics", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(
      "queue_depth " + std::to_string(queue_depth) + "\n" +
      "time_to_first_token " + std::to_string(time_to_first_token) + "\n" +
      "tokens_per_second " + std::to_string(tokens_per_second) + "\n" +
      "active_streams " + std::to_string(active_streams) + "\n",
      "text/plain"
    );
  });

  svr.listen("0.0.0.0", 8080);

  // shutdown (listen blocks; this is mainly for completeness)
  running.store(false);
  job_queue_cv.notify_all();
  if (worker.joinable()) worker.join();
  return 0;
}
