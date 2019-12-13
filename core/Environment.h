#ifndef __MASS_ENVIRONMENT_H__
#define __MASS_ENVIRONMENT_H__
#include "dart/dart.hpp"
#include "Character.h"
#include "Muscle.h"
#include <tuple>

namespace MASS
{
    enum SAMPLING_TYPE{
        UNIFORM,
        NORMAL,
        ADAPTIVE
    };

struct MuscleTuple
{
	Eigen::VectorXd JtA;
	Eigen::MatrixXd L;
	Eigen::VectorXd b;
	Eigen::VectorXd tau_des;
};
class Environment
{
public:
	Environment();
    Environment(int _index);

	void SetUseMuscle(bool use_muscle){mUseMuscle = use_muscle;}
	void SetControlHz(int con_hz) {mControlHz = con_hz;}
	void SetSimulationHz(int sim_hz) {mSimulationHz = sim_hz;}

	void SetCharacter(Character* character) {mCharacter = character;}
	void SetGround(const dart::dynamics::SkeletonPtr& ground) {mGround = ground;}

	void SetRewardParameters(double w_q,double w_v,double w_ee,double w_com){this->w_q = w_q;this->w_v = w_v;this->w_ee = w_ee;this->w_com = w_com;}
	void Initialize();
	void Initialize(const std::string& meta_file,bool load_obj = false);
public:
	void Step();
	void Reset(bool RSI = true);
	bool IsEndOfEpisode();
	Eigen::VectorXd GetState();
	void SetAction(const Eigen::VectorXd& a);
	double GetReward();

	Eigen::VectorXd GetDesiredTorques();
	Eigen::VectorXd GetMuscleTorques();

	const dart::simulation::WorldPtr& GetWorld(){return mWorld;}
	Character* GetCharacter(){return mCharacter;}
	const dart::dynamics::SkeletonPtr& GetGround(){return mGround;}
	int GetControlHz(){return mControlHz;}
	int GetSimulationHz(){return mSimulationHz;}
	int GetNumTotalRelatedDofs(){return mCurrentMuscleTuple.JtA.rows();}
	std::vector<MuscleTuple>& GetMuscleTuples(){return mMuscleTuples;};
	int GetNumState(){return mNumState;}
	int GetNumAction(){return mNumActiveDof;}
	int GetNumSteps(){return mSimulationHz/mControlHz;}
	
	const Eigen::VectorXd& GetActivationLevels(){return mActivationLevels;}
	const Eigen::VectorXd& GetAverageActivationLevels(){return mAverageActivationLevels;}
	void SetActivationLevels(const Eigen::VectorXd& a){mActivationLevels = a;}
	bool GetUseMuscle(){return mUseMuscle;}
	bool GetUseAdaptiveSampling(){return sample_param_type==MASS::ADAPTIVE;}

    // push experiments
    bool HasCrouchVariation(){return crouch_angle_set.size() > 1;}
	void SampleWalkingParams();
    void SampleWalkingParamsFromMarginalSampled();
    void SamplePushParams();
	static int GetMarginalStateNum(){return 4;}
	void SetMarginalSampled(const std::vector<Eigen::VectorXd> &_marginal_samples, const std::vector<double> &marginal_cumulative_probs);
	std::tuple<int, double, double> GetWalkingParams();
	std::tuple<double, double, double> GetNormalizedWalkingParams();
	void SetWalkingParams(int _crouch_angle, double _stride_length, double _walk_speed);
	void SetPushParams(int _push_step, double _push_duration, double _push_force_mean, double _push_force_std, double _push_start_timing_mean, double _push_start_timing_std);
    void PrintPushParamsSampled();
	void PrintWalkingParams();
	void PrintWalkingParamsSampled();

	void SetSampleStrategy(int flag);

	void SetPushEnable(bool flag){push_enable = flag;}

	double GetMechanicalWork(){return this->mechanicalWork;}

private:
	dart::simulation::WorldPtr mWorld;
	int mControlHz,mSimulationHz;
	bool mUseMuscle;
	Character* mCharacter;
	dart::dynamics::SkeletonPtr mGround;
	Eigen::VectorXd mAction;
	Eigen::VectorXd mTargetPositions,mTargetVelocities;

	double world_start_time;

	int mNumState;
	int mNumActiveDof;
	int mRootJointDof;

	Eigen::VectorXd mActivationLevels;
	Eigen::VectorXd mAverageActivationLevels;
	Eigen::VectorXd mDesiredTorque;
	std::vector<MuscleTuple> mMuscleTuples;
	MuscleTuple mCurrentMuscleTuple;
	int mSimCount;
	int mRandomSampleIndex;

	double w_q,w_v,w_ee,w_com;


	std::default_random_engine generator;

    int index;

    double phase; // for adaptive sampling
    int crouch_angle;
    double stride_length;
    double walk_speed;

    int crouch_angle_index;
    std::vector<int> crouch_angle_set;
    std::vector<double> stride_length_mean_vec;
    std::vector<double> stride_length_var_vec;
    std::vector<double> walk_speed_mean_vec;
    std::vector<double> walk_speed_var_vec;
    std::vector<double> stride_speed_covar_vec;
    int sample_param_type;

    bool walking_param_change;


    bool marginal_set;
    std::vector<Eigen::VectorXd> marginal_samples;
    std::vector<double> marginal_cumulative_probs;

    WalkFSM walk_fsm;

    bool push_enable;
    int push_step;
    double push_duration;
    double push_force;
    double push_force_mean;
    double push_force_std;
    double push_start_timing;
    double push_start_timing_mean;
    double push_start_timing_std;

    double push_start_time;
    bool push_ready;

    double mechanicalWork;

    double muscle_maxforce_ratio;
};
};

#endif