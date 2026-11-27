// Observation terms for Hiking-convention perceptive policies (G1_Perceptive_WMAC).
//
// * hiking_torso_ang_vel / hiking_torso_projected_gravity: the policy measures
//   angular velocity and gravity in the torso_link frame. They are computed here
//   from the pelvis IMU in LowState and the waist encoders (Unitree convention),
//   using Unitree's waist chain pelvis -> Rz(yaw) -> Rx(roll) -> Ry(pitch) -> torso.
//   The pelvis IMU alone, or the secondary IMU (on waist_roll_link), makes this
//   policy fall.
// * hiking_depth_ring: 37 frames of [depth / max_depth, valid] (36 x 64), oldest
//   first, read from a PointCloud2 topic carrying image-plane depth in metres
//   (one float32 field per pixel; inf/nan/<=0 = no return). The ring is refilled
//   with the newest frame on reset and advanced by one frame per policy step.
//   Missing returns stay at 1e6 m because the policy blurs before clipping.
#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/ros2/PointCloud2_.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <atomic>
#include <limits>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>

namespace hiking
{

inline std::vector<int> waist_ids(const YAML::Node& params)
{
    std::vector<int> ids{12, 13, 14};
    if (params && params["waist_joint_ids"]) ids = params["waist_joint_ids"].as<std::vector<int>>();
    if (ids.size() != 3) throw std::runtime_error("waist_joint_ids must list waist yaw, roll and pitch");
    return ids;
}

// Torso orientation and torso-frame angular velocity from the pelvis IMU.
inline void torso_from_pelvis(const Eigen::Quaternionf& pelvis, const Eigen::Vector3f& pelvis_gyro,
                              const Eigen::Vector3f& q_waist, const Eigen::Vector3f& dq_waist,
                              Eigen::Quaternionf& torso, Eigen::Vector3f& torso_gyro)
{
    const Eigen::Quaternionf rz(Eigen::AngleAxisf(q_waist[0], Eigen::Vector3f::UnitZ()));
    const Eigen::Quaternionf rx(Eigen::AngleAxisf(q_waist[1], Eigen::Vector3f::UnitX()));
    const Eigen::Quaternionf ry(Eigen::AngleAxisf(q_waist[2], Eigen::Vector3f::UnitY()));
    const Eigen::Quaternionf relative = rz * rx * ry;
    torso = pelvis * relative;
    torso_gyro = relative.conjugate() * pelvis_gyro
               + (rx * ry).conjugate() * Eigen::Vector3f(0.f, 0.f, dq_waist[0])
               + ry.conjugate() * Eigen::Vector3f(dq_waist[1], 0.f, 0.f)
               + Eigen::Vector3f(0.f, dq_waist[2], 0.f);
}

inline void torso_state(isaaclab::ManagerBasedRLEnv* env, const YAML::Node& params,
                        Eigen::Quaternionf& torso, Eigen::Vector3f& gyro)
{
    auto& data = env->robot->data;
    const auto ids = waist_ids(params);
    Eigen::Vector3f q, dq;
    for (int i = 0; i < 3; ++i) { q[i] = data.joint_pos[ids[i]]; dq[i] = data.joint_vel[ids[i]]; }
    torso_from_pelvis(data.root_quat_w, data.root_ang_vel_b, q, dq, torso, gyro);
}

// Latest depth image from the simulator or camera driver.
class DepthSource
{
public:
    static DepthSource& instance(const std::string& topic, int height, int width)
    {
        static std::unique_ptr<DepthSource> source;
        static std::mutex creation;
        std::lock_guard<std::mutex> lock(creation);
        if (!source) source.reset(new DepthSource(topic, height, width));
        if (source->topic_ != topic || source->height_ != height || source->width_ != width)
            throw std::runtime_error("hiking_depth_ring: one depth topic/shape per controller");
        return *source;
    }

    // Copies the newest image (metres); returns false if none has arrived yet.
    bool latest(std::vector<float>& depth)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (frame_.empty()) return false;
        depth = frame_;
        return true;
    }

    // Test hook: inject a frame without DDS.
    void inject(const std::vector<float>& depth)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        frame_ = depth;
    }

private:
    DepthSource(std::string topic, int height, int width)
    : topic_(std::move(topic)), height_(height), width_(width)
    {
        if (std::getenv("HIKING_DEPTH_NO_DDS")) return;   // unit tests inject frames directly
        subscriber_ = std::make_shared<unitree::robot::ChannelSubscriber<sensor_msgs::msg::dds_::PointCloud2_>>(topic_);
        subscriber_->InitChannel([this](const void* message) {
            const auto& cloud = *static_cast<const sensor_msgs::msg::dds_::PointCloud2_*>(message);
            const size_t count = static_cast<size_t>(height_) * width_;
            if (cloud.height() != static_cast<uint32_t>(height_) || cloud.width() != static_cast<uint32_t>(width_)
                || cloud.point_step() != 4 || cloud.data().size() < count * 4) {
                if (!warned_shape_.exchange(true))
                    spdlog::warn("hiking_depth_ring: ignoring {}x{} depth cloud (expected {}x{} float32)",
                                 cloud.height(), cloud.width(), height_, width_);
                return;
            }
            std::vector<float> depth(count);
            std::memcpy(depth.data(), cloud.data().data(), count * sizeof(float));
            std::lock_guard<std::mutex> lock(mutex_);
            frame_.swap(depth);
            if (!received_.exchange(true)) spdlog::info("hiking_depth_ring: first depth image received on {}", topic_);
        }, 1);
        spdlog::info("hiking_depth_ring: subscribed to {}", topic_);
    }

    std::string topic_;
    int height_, width_;
    std::mutex mutex_;
    std::vector<float> frame_;
    std::atomic<bool> warned_shape_{false};
    std::atomic<bool> received_{false};
    std::shared_ptr<unitree::robot::ChannelSubscriber<sensor_msgs::msg::dds_::PointCloud2_>> subscriber_;
};

}  // namespace hiking

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(hiking_torso_ang_vel)
{
    Eigen::Quaternionf torso;
    Eigen::Vector3f gyro;
    hiking::torso_state(env, params, torso, gyro);
    return {gyro[0], gyro[1], gyro[2]};
}

REGISTER_OBSERVATION(hiking_torso_projected_gravity)
{
    Eigen::Quaternionf torso;
    Eigen::Vector3f gyro;
    hiking::torso_state(env, params, torso, gyro);
    const Eigen::Vector3f gravity = torso.conjugate() * env->robot->data.GRAVITY_VEC_W;
    return {gravity[0], gravity[1], gravity[2]};
}

REGISTER_OBSERVATION(hiking_depth_ring)
{
    const std::string topic = params["topic"].as<std::string>("rt/hiking/depth");
    const int ring = params["ring_length"].as<int>(37);
    const int height = params["height"].as<int>(36);
    const int width = params["width"].as<int>(64);
    const float max_depth = params["max_depth"].as<float>(2.5f);
    const size_t pixels = static_cast<size_t>(height) * width;
    const size_t frame_size = 2 * pixels;
    static std::vector<float> buffer;      // [ring][2][height][width], oldest first
    static bool warned_missing = false, warned_running_blind = false;

    std::vector<float> depth;
    std::vector<float> frame(frame_size);
    if (!hiking::DepthSource::instance(topic, height, width).latest(depth)) {
        if (!warned_missing) {
            spdlog::warn("hiking_depth_ring: no depth image on {} yet; the policy sees no terrain", topic);
            warned_missing = true;
        }
        if (env->episode_length > 0 && !warned_running_blind) {
            spdlog::error("hiking_depth_ring: the policy is running without depth images ({})", topic);
            warned_running_blind = true;
        }
        depth.assign(pixels, std::numeric_limits<float>::infinity());
    }
    for (size_t i = 0; i < pixels; ++i) {
        const float d = depth[i];
        const bool hit = std::isfinite(d) && d > 0.f;
        frame[i] = (hit ? std::min(d, 1e6f) : 1e6f) / max_depth;
        frame[pixels + i] = (hit && d <= max_depth) ? 1.f : 0.f;
    }
    if (env->episode_length == 0 || buffer.size() != ring * frame_size) {
        buffer.resize(ring * frame_size);
        for (int r = 0; r < ring; ++r) std::copy(frame.begin(), frame.end(), buffer.begin() + r * frame_size);
    } else {
        std::move(buffer.begin() + frame_size, buffer.end(), buffer.begin());
        std::copy(frame.begin(), frame.end(), buffer.end() - frame_size);
    }
    return buffer;
}

}  // namespace mdp
}  // namespace isaaclab
