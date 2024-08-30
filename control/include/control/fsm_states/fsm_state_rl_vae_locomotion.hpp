#ifndef FSM_STATE_RL_VAE_LOCOMOTION_HPP_
#define FSM_STATE_RL_VAE_LOCOMOTION_HPP_

#include <cstring>
#include <experimental/filesystem>
#include <sys/timerfd.h>

#include "controllers/leg_controller.hpp"
#include "controllers/state_estimator_container.hpp"
#include "utilities/control_utilities.hpp"
#include "header/lcm_type/rl_vae_action_lcmt.hpp"
#include "header/lcm_type/rl_vae_obs_lcmt.hpp"
#include "cpu_mlp.hpp"
#include "fsm_state.hpp"
#include "utilities/toolkit.hpp"

static const int kNumObs = 45;

/**
 * @brief FSM state for common periodic gaits using reinforcement learning.
 *
 */
template < typename T > class FsmStateRlVaeLocomotion : public FsmState< T > {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FsmStateRlVaeLocomotion( ControlFsmData< T >* control_fsm_data );

    // Behavior to be carried out when entering a state
    void OnEnter();

    // Run the normal behavior for the state
    void Run();

    // Checks for any transition triggers
    FsmStateName CheckTransition();

    // Manages state specific transitions
    TransitionData< T > Transition();

    // Behavior to be carried out when exiting a state
    void OnExit();

    // Set user parameters
    void SetUserParameters();

    // Remap joint pos/vel for real robot (different in isaac gym)
    void Remap( Eigen::Matrix< float, 12, 1 >& joint_data );

    // Use thread to inference rl model
    void RlModelInferenceThread();
    void CreateRlModelInferenceThread();
    void StopRlModelInference();

    void UpdateAction();
    void UpdateCommand();
    void UpdateHistory();
    void UpdateObservation();
    void UpdateProjectedGravity( Eigen::Vector4f& robot_quat, Eigen::Vector3f& gravity_vec );

private:
    void HandleActionLcm( const lcm::ReceiveBuffer* rbuf, const std::string& chan, const rl_vae_action_lcmt* msg );
    void HandleLcm();

    // Keep track of the control iterations
    int iter_ = 0;

    // Observation
    Eigen::Matrix< float, 3, 1 >  projected_gravity_;
    Eigen::Matrix< float, 3, 1 > command_;
    Eigen::Matrix< float, 12, 1 > joint_pos_;
    Eigen::Matrix< float, 12, 1 > joint_vel_;
    Eigen::Matrix< float, 12, 1 > prev_action_;
    Eigen::Matrix< float, 4, 1 >  clock_inputs_;

    Eigen::Matrix< float, kNumObs, 1 >                                                          obs_;
    Eigen::Matrix< float, 12, 1 >                                                                         action_;
    Eigen::Matrix< float, 12, 1 >                                                                         default_joint_pos_;
    Eigen::Matrix< int, 12, 1 >                                                                           motor_remap_;
    Eigen::Vector3f                                                                                       gravity_vec_;
    Eigen::Vector4f                                                                                       robot_quat_;  // (w, x, y, z)
    Eigen::Vector3f                                                                                       base_ang_vel_;
    Eigen::Matrix< float, 4, 3 >                                                                          gaits_;
    Eigen::Matrix< float, 12, 1 >                                                                         des_joint_pos_;
    int                                                                                                   gait_id_;
    float                                                                                                 gait_indices_;

    // Motor Kp/Kd
    Eigen::Matrix< float, 3, 3 > kp_mat_;
    Eigen::Matrix< float, 3, 3 > kd_mat_;

    // Scale
    Eigen::Matrix< float, 3, 1 >  command_scale_;
    float                         base_ang_vel_scale_;
    float                         joint_pos_scale_;
    float                         joint_vel_scale_;
    float                         action_scale_;
    float                         abad_scale_reduction_;

    // Command range
    float min_lin_vel_x_;    // [m/s]
    float max_lin_vel_x_;    // [m/s]
    float min_lin_vel_y_;    // [m/s]
    float max_lin_vel_y_;    // [m/s]
    float min_ang_vel_yaw_;  // [rad/s]
    float max_ang_vel_yaw_;  // [rad/s]

    // Velocity offset
    float lin_vel_x_offset_;    // [m/s]
    float lin_vel_y_offset_;    // [m/s]
    float ang_vel_yaw_offset_;  // [rad/s]

    // Acceleration limit
    float min_acc_lin_vel_x_;
    float max_acc_lin_vel_x_;
    float min_acc_lin_vel_y_;
    float max_acc_lin_vel_y_;
    float min_acc_ang_vel_yaw_;
    float max_acc_ang_vel_yaw_;

    // Current velocity (from command)
    float cur_lin_vel_x_;
    float cur_lin_vel_y_;
    float cur_ang_vel_yaw_;

    float stop_update_cmd_;  // stop update cmd when transition to other fsm

    double control_dt_;  // policy control period

    std::string actor_path_;      // rl model parameters path

    cyberdog::cpu_mlp::MlpFullyConnected< float, kNumObs, 12, cyberdog::cpu_mlp::ActivationType::elu > actor_;

    bool         rl_model_inference_running_;
    std::thread* rl_model_inference_thread_;
    bool         first_run_ = true;
    std::mutex   joint_command_mutex_;
    std::mutex   receive_action_mutex_;

    lcm::LCM                      rl_lcm_;
    rl_vae_obs_lcmt               rl_vae_obs_lcm_;
    Eigen::Matrix< float, 12, 1 > received_action_;
    std::thread                   rl_lcm_thread_;
};

#endif  // FSM_STATE_RL_LOCOMOTION_HPP_