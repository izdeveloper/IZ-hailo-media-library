#include <nlohmann/json.hpp>
#include <cxxopts/cxxopts.hpp>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include "pipeline/pipeline_factory.hpp"
#include "common/httplib/httplib_utils.hpp"
#include "common/logger_macros.hpp"
#include "common/common.hpp"
#include "resources/common/repository.hpp"
#include "media_library/signal_utils.hpp"
#include "media_library/media_library_types.hpp"

#define DEFAULT_CONFIGS_PATH "/etc/imaging/cfg/medialib_configs/"
#define APPEND_CONFIG_PATH(path) DEFAULT_CONFIGS_PATH path
#define DEFAULT_MEDIALIB_CONFIG_PATH APPEND_CONFIG_PATH("webserver_medialib_config.json")

static std::atomic<bool> g_webserver_stopping{false};


/*void start_day_night_auto_switcher()
{
    std::thread([]() {
        // Initial boot delay to let media pipeline and WebRTC stabilize
        for (int i = 0; i < 15 && !g_webserver_stopping; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        httplib::Client cli("127.0.0.1", 8000);
        cli.set_connection_timeout(5);
        cli.set_read_timeout(5);

        bool is_night_mode = false;

        const uint32_t NIGHT_THRESHOLD_TIME   = 15000; // Trigger night mode (>15ms)
        const uint32_t DAY_THRESHOLD_TIME     = 11000;  // Trigger day mode (<8ms)
        const uint32_t NIGHT_INTEGRATION_TIME = 11000; // 12ms capped night exposure
        const uint32_t NIGHT_GAIN             = 4;  // Boosted gain for YOLO

        const int CHECK_INTERVAL_MINUTES = 10;

        while (!g_webserver_stopping) {
            try {
                if (!is_night_mode) {
                    // Daytime Check: AE is already Auto; just query current exposure time
                    auto res = cli.Get("/isp/auto_exposure");
                    if (res && res->status == 200) {
                        auto j = nlohmann::json::parse(res->body);
                        uint32_t integration_time = j.value("integration_time", 0);

                        if (integration_time > NIGHT_THRESHOLD_TIME) {
                            WEBSERVER_LOG_INFO("Auto-Switcher: Low light ({}), applying manual night cap", integration_time);
                            
                            nlohmann::json ae_night_body = {
                                {"enabled", false},
                                {"integration_time", NIGHT_INTEGRATION_TIME},
                                {"gain", NIGHT_GAIN},
                                {"backlight", 0}
                            };
                            cli.Post("/isp/auto_exposure", ae_night_body.dump(), "application/json");
                            is_night_mode = true;
                        }
                    }
                } else {
                    // Morning Probing: Enable Auto AE to measure ambient light
                    nlohmann::json probe_body = {{"enabled", true}, {"gain", 0}, {"integration_time", 0}, {"backlight", 0}};
                    cli.Post("/isp/auto_exposure", probe_body.dump(), "application/json");

                    // Wait 4 seconds for hardware 3A algorithm and frame buffers to converge smoothly
                    for (int i = 0; i < 4 && !g_webserver_stopping; ++i) {
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                    }

                    auto res = cli.Get("/isp/auto_exposure");
                    if (res && res->status == 200) {
                        auto j = nlohmann::json::parse(res->body);
                        uint32_t integration_time = j.value("integration_time", 0);

                        if (integration_time < DAY_THRESHOLD_TIME) {
                            WEBSERVER_LOG_INFO("Auto-Switcher: Daylight ({}), returning to Auto AE", integration_time);
                            is_night_mode = false;
                        } else {
                            // Still dark: re-apply manual night exposure cap
                            WEBSERVER_LOG_INFO("Auto-Switcher: Still dark ({}), re-applying night cap", integration_time);
                            nlohmann::json ae_night_body = {
                                {"enabled", false},
                                {"integration_time", NIGHT_INTEGRATION_TIME},
                                {"gain", NIGHT_GAIN},
                                {"backlight", 0}
                            };
                            cli.Post("/isp/auto_exposure", ae_night_body.dump(), "application/json");
                        }
                    }
                }
            } catch (const std::exception &e) {
                WEBSERVER_LOG_ERROR("Auto-Switcher exception: {}", e.what());
            }

            // Sleep 30 minutes in 1-second chunks so the thread exits cleanly on application shutdown
            for (int i = 0; i < CHECK_INTERVAL_MINUTES * 60 && !g_webserver_stopping; ++i) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
    }).detach();
}*/

static std::string patch_vendor_configs_in_memory(const std::string& original_config_path) {
    try {
        std::string tmp_dir = "/tmp/hailo_patched_configs/";
        std::filesystem::create_directories(tmp_dir);

        std::ifstream in_main(original_config_path);
        if (!in_main.is_open()) return original_config_path;
        
        nlohmann::json main_json;
        in_main >> main_json;
        in_main.close();

        if (main_json.contains("profiles")) {
            for (auto& profile : main_json["profiles"]) {
                std::string orig_profile_path = profile["config_file"];
                
                std::ifstream in_prof(orig_profile_path);
                if (in_prof.is_open()) {
                    nlohmann::json prof_json;
                    in_prof >> prof_json;
                    in_prof.close();

                    // --- NEW LOGIC: Follow the path to the application_settings file ---
                    if (prof_json.contains("application_settings") && prof_json["application_settings"].is_string()) {
                        std::string app_settings_path = prof_json["application_settings"];
                        std::ifstream in_app(app_settings_path);
                        
                        if (in_app.is_open()) {
                            nlohmann::json app_json;
                            in_app >> app_json;
                            in_app.close();

                            // Override the buffer pools inside the actual app settings JSON
                            if (app_json.contains("application_input_streams") && 
                                app_json["application_input_streams"].contains("resolutions")) {
                                for (auto& res : app_json["application_input_streams"]["resolutions"]) {
                                    res["pool_max_buffers"] = 35; // Force 35 buffers for all streams
                                }
                            }

                            // Save the patched application_settings file to /tmp/
                            std::string prof_name_str = profile["name"];
                            std::string new_app_path = tmp_dir + prof_name_str + "_application_settings.json";
                            std::ofstream out_app(new_app_path);
                            out_app << app_json.dump(4);
                            out_app.close();

                            // Update the profile JSON to point to our new patched app settings file
                            prof_json["application_settings"] = new_app_path;
                        }
                    }

                    // Save the patched profile to /tmp/
                    std::string new_prof_name = std::filesystem::path(orig_profile_path).filename();
                    std::string new_prof_path = tmp_dir + new_prof_name;
                    std::ofstream out_prof(new_prof_path);
                    out_prof << prof_json.dump(4);
                    out_prof.close();

                    // Update the main config to point to our new patched profile
                    profile["config_file"] = new_prof_path;
                }
            }
        }

        // Save the patched main config to /tmp/
        std::string new_main_path = tmp_dir + "webserver_medialib_config.json";
        std::ofstream out_main(new_main_path);
        out_main << main_json.dump(4);
        out_main.close();

        std::cout << "Successfully patched vendor buffer pools to 35 in /tmp/" << std::endl;
        return new_main_path; // Return the path to the patched config
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to patch configs: " << e.what() << std::endl;
        return original_config_path; // Fallback to original if something fails
    }
}

bool is_device_in_use(const std::string &device_path)
{
    struct stat dev_stat;
    if (stat(device_path.c_str(), &dev_stat) != 0)
        return false;

    pid_t self = getpid();
    std::error_code ec;

    for (const auto &proc : std::filesystem::directory_iterator("/proc", ec))
    {
        const auto &path = proc.path();
        const auto name = path.filename().string();

        if (name.empty() || !std::isdigit(name[0]))
            continue;

        if (std::stoi(name) == self)
            continue;

        std::string comm;
        if (std::ifstream(path / "comm") >> comm && comm.rfind("isp_med", 0) == 0)
            continue;

        for (const auto &fd : std::filesystem::directory_iterator(path / "fd", ec))
        {
            struct stat fd_stat;
            if (stat(fd.path().c_str(), &fd_stat) == 0 && S_ISCHR(fd_stat.st_mode) &&
                fd_stat.st_rdev == dev_stat.st_rdev)
                return true;
        }
    }
    return false;
}
void flags_init(int argc, char *argv[], std::string &medialib_config_path)
{
    try
    {
        cxxopts::Options options(argv[0], "Webserver application");
        options.add_options()("config", "Media library configuration path",
                              cxxopts::value<std::string>()->default_value(DEFAULT_MEDIALIB_CONFIG_PATH))(
            "h,help", "Print usage");

        auto result = options.parse(argc, argv);

        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        std::string config_path = result["config"].as<std::string>();
        WEBSERVER_LOG_INFO("Using medialib config path: {}", config_path);
        medialib_config_path = config_path;
    }
    catch (const cxxopts::OptionException &e)
    {
        WEBSERVER_LOG_ERROR("Error parsing options: {}", e.what());
        std::cout << "Error parsing options: " << e.what() << std::endl;
        std::cout << "Use --help to see valid options" << std::endl;
        exit(1);
    }
}

int main(int argc, char *argv[])
{
    WEBSERVER_LOG_INFO("Starting webserver");

    if (is_device_in_use("/dev/video0"))
    {
        WEBSERVER_LOG_ERROR("/dev/video0 is already in use by another process");
        return 1;
    }

    setenv("HAILO_MEDIA_LIB_SKIP_METADATA_CONFIG_VALIDATION", "1", 1);
    setenv("MEDIALIB_USE_DIV_FRAMERATE_LOGIC", "1", 1);
    setenv("MEDIALIB_FD_DUP", "1", 1);

    // Disable persistence so analytics config gets updated when switching pipelines
    application_analytics_config_t::is_persistent = false;

    std::string medialib_config_path = "";
    Architecture arch = get_hailo_architecture();
    flags_init(argc, argv, medialib_config_path);

    medialib_config_path = patch_vendor_configs_in_memory(medialib_config_path);

    std::unique_ptr<HTTPServer> svr = HTTPServer::create();
    // register error handler
    svr->set_exception_handler([](const auto &req, auto &res, std::exception_ptr ep) {
        auto fmt = "Error 500: %s";
        char buf[BUFSIZ];
        try
        {
            std::rethrow_exception(ep);
        }
        catch (std::exception &e)
        {
            snprintf(buf, sizeof(buf), fmt, e.what());
            WEBSERVER_LOG_ERROR("{}", buf);
        }
        catch (...)
        { // See the following NOTE
            snprintf(buf, sizeof(buf), fmt, "Unknown Exception");
            WEBSERVER_LOG_ERROR("Unknown Excpetion");
        }
        res.set_content(buf, "text/html");
        res.status = 500;
    });

    // Create pipeline factory instead of direct pipeline creation
    WebserverResourceRepository resources =
        webserver::resources::ResourceRepository::create(*svr.get(), medialib_config_path);

    auto pipeline_factory = std::make_unique<webserver::pipeline::PipelineFactory>(*resources, arch, pipeline_t::LPR);

    signal_utils::SignalHandler signal_handler;
    signal_handler.register_signal_handler([&pipeline_factory, &resources, &svr](int signal) {
        // Use an atomic flag to make sure the shutdown sequence is executed only once,
        // even if multiple signals are delivered or multiple threads enter this handler.
        bool expected = false;

        // On the first call:
        //   - g_webserver_stopping is false  -> compare_exchange_strong() returns true
        //   - g_webserver_stopping is set to true and the shutdown flow continues.
        // On any subsequent call:
        //   - g_webserver_stopping is already true
        //   - compare_exchange_strong() returns false and we exit the handler immediately.
        if (!g_webserver_stopping.compare_exchange_strong(expected, true))
        {
            return;
        }
        std::cout << "Received signal " << signal << ", stopping pipeline..." << std::endl;

        pipeline_factory = nullptr;
        resources = nullptr;
        svr = nullptr;

        std::cout << "Pipeline stopped, exiting now" << std::endl;
        // Use _Exit(0) to terminate the process immediately without running static
        // destructors, flushing stdio buffers, or calling any atexit/quick_exit handlers.
        _Exit(0);
    });

    pipeline_factory->get_current_pipeline()->start();

    WEBSERVER_LOG_INFO("Webserver started");

    //start_day_night_auto_switcher();

    svr->listen("0.0.0.0", 8000);
}
