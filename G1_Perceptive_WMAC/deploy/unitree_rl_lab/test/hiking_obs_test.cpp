// Runs unitree_rl_lab's ObservationManager/ActionManager on a recorded input
// sequence with the Hiking deploy.yaml and writes every observation vector, so
// check_hiking_obs.py can compare it with the training-side reference.
//
// inputs.bin: int32 N, int32 pixels, then per step float32
//   pelvis_quat_wxyz[4], pelvis_gyro[3], q_unitree[29], dq_unitree[29], action[29], depth_m[pixels]
// obs.bin:    int32 N, int32 width, then N x width float32
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/observations/hiking_observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <fstream>
#include <iostream>

class RecordedRobot : public isaaclab::Articulation
{
public:
    void update() override {}
};

int main(int argc, char** argv)
{
    if (argc != 4) {
        std::cerr << "usage: hiking_obs_test deploy.yaml inputs.bin obs.bin\n";
        return 2;
    }
    setenv("HIKING_DEPTH_NO_DDS", "1", 1);
    std::ifstream in(argv[2], std::ios::binary);
    int32_t steps = 0, pixels = 0;
    in.read(reinterpret_cast<char*>(&steps), 4);
    in.read(reinterpret_cast<char*>(&pixels), 4);
    const int record = 4 + 3 + 29 * 3 + pixels;
    std::vector<std::vector<float>> rows(steps, std::vector<float>(record));
    for (auto& row : rows) in.read(reinterpret_cast<char*>(row.data()), record * 4);
    if (!in) { std::cerr << "truncated inputs\n"; return 1; }

    unitree::common::UnitreeJoystick joystick;
    unitree::common::REMOTE_DATA_RX key{};
    key.RF_RX.ly = 0.6f; key.RF_RX.rx = -0.3f;
    auto robot = std::make_shared<RecordedRobot>();
    robot->data.joystick = &joystick;
    robot->data.root_quat_w = Eigen::Quaternionf::Identity();
    robot->data.root_ang_vel_b.setZero();

    auto apply = [&](const std::vector<float>& row) {
        auto& d = robot->data;
        d.root_quat_w = Eigen::Quaternionf(row[0], row[1], row[2], row[3]);
        d.root_ang_vel_b = Eigen::Vector3f(row[4], row[5], row[6]);
        for (int j = 0; j < 29; ++j) { d.joint_pos[j] = row[7 + j]; d.joint_vel[j] = row[36 + j]; }
        std::vector<float> depth(row.begin() + 94, row.end());
        hiking::DepthSource::instance("rt/hiking/depth", 36, 64).inject(depth);
        joystick.extract(key);
    };

    auto cfg = YAML::LoadFile(argv[1]);
    robot->data.joint_pos.resize(29);
    robot->data.joint_vel.resize(29);
    apply(rows[0]);
    isaaclab::ManagerBasedRLEnv env(cfg, robot);
    env.reset();
    std::ofstream out(argv[3], std::ios::binary);
    int32_t width = 0;
    for (int k = 0; k < steps; ++k) {
        apply(rows[k]);
        env.episode_length += 1;                 // ManagerBasedRLEnv::step() without the policy
        auto obs = env.observation_manager->compute().at("obs");
        if (k == 0) {
            width = static_cast<int32_t>(obs.size());
            out.write(reinterpret_cast<char*>(&steps), 4);
            out.write(reinterpret_cast<char*>(&width), 4);
        }
        out.write(reinterpret_cast<char*>(obs.data()), obs.size() * 4);
        std::vector<float> action(rows[k].begin() + 65, rows[k].begin() + 94);
        env.action_manager->process_action(action);
    }
    auto targets = env.action_manager->processed_actions();
    std::cout << "steps " << steps << " observation width " << width << " last target[12..14] "
              << targets[12] << " " << targets[13] << " " << targets[14] << "\n";
    return 0;
}
