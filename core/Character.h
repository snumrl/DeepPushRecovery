#ifndef __MASS_CHARACTER_H__
#define __MASS_CHARACTER_H__
#include "dart/dart.hpp"
#include "BVH.h"

namespace MASS
{
class Muscle;
class Character
{
public:
	Character();
    Character(int _index);

	void LoadSkeleton(const std::string& path, bool create_obj = false, double _height_scale=1., double _mass_scale=1.);
	void LoadMuscles(const std::string& path);
	void LoadBVH(const std::string& path,bool cyclic=true);

	void GenerateBvhForPushExp(long crouch_angle, double stride_length, double walk_speed, double scale=1.);
    void GenerateBvhForPushExp_old(long crouch_angle, double stride_length, double walk_speed);

	void Reset();	
	void SetPDParameters(double kp, double kv);
	void AddEndEffector(const std::string& body_name){mEndEffectors.push_back(mSkeleton->getBodyNode(body_name));}
	Eigen::VectorXd GetSPDForces(const Eigen::VectorXd& p_desired);

	Eigen::VectorXd GetTargetPositions(double t,double dt);
	std::pair<Eigen::VectorXd,Eigen::VectorXd> GetTargetPosAndVel(double t,double dt);

	const dart::dynamics::SkeletonPtr& GetSkeleton(){return mSkeleton;}
	const std::vector<Muscle*>& GetMuscles() {return mMuscles;}
	const std::vector<dart::dynamics::BodyNode*>& GetEndEffectors(){return mEndEffectors;}
	BVH* GetBVH(){return mBVH;}
public:
    int index;
	dart::dynamics::SkeletonPtr mSkeleton;
	BVH* mBVH;
	Eigen::Isometry3d mTc;
	double height_scale;
	double mass_scale;

	std::vector<Muscle*> mMuscles;
	std::vector<dart::dynamics::BodyNode*> mEndEffectors;

	Eigen::VectorXd mKp, mKv;

};
};

#endif
