// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "LinearInterpolator.h"

class State_FixStand : public FSMState
{
public:
    State_FixStand(int state, std::string state_string = "FixStand") 
    : FSMState(state, state_string) 
    {
        // A FixStand-type state may carry its own pose/gains (e.g. HikingStand);
        // otherwise it uses the FixStand section.
        auto cfg = param::config["FSM"][state_string];
        if(!cfg["qs"]) cfg = param::config["FSM"]["FixStand"];
        ts_ = cfg["ts"].as<std::vector<float>>();
        qs_ = cfg["qs"].as<std::vector<std::vector<float>>>();
        kp_ = cfg["kp"].as<std::vector<float>>();
        kd_ = cfg["kd"].as<std::vector<float>>();
        assert(ts_.size() == qs_.size());
    }

    void enter()
    {
        // set gain
        const auto& kp = kp_;
        const auto& kd = kd_;
        for(int i(0); i < kp.size(); ++i)
        {
            auto & motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = kp[i];
            motor.kd() = kd[i];
            motor.dq() = motor.tau() = 0;
        }


        // set initial position
        std::vector<float> q0;
        for(int i(0); i < kp.size(); ++i) {
            q0.push_back(lowcmd->msg_.motor_cmd()[i].q());
        }
        qs_[0] = q0;
        t0_ = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3;
    }

    void run()
    {
        float t = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3 - t0_;
        auto q = linear_interpolate(t, ts_, qs_);
        
        for(int i(0); i < q.size(); ++i) {
            lowcmd->msg_.motor_cmd()[i].q() = q[i];
        }
    }

private:
    double t0_;
    std::vector<float> ts_;
    std::vector<std::vector<float>> qs_;
    std::vector<float> kp_, kd_;
};

REGISTER_FSM(State_FixStand)