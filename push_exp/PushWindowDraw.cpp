//
// Created by trif on 08/12/2019.
//

#include "PushWindow.h"
#include <iostream>

using namespace MASS;
using namespace dart;
using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart::gui;

void
PushWindow::
draw()
{
    GLfloat matrix[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
    Eigen::Matrix3d A;
    Eigen::Vector3d b;
    A<<matrix[0],matrix[4],matrix[8],
            matrix[1],matrix[5],matrix[9],
            matrix[2],matrix[6],matrix[10];
    b<<matrix[12],matrix[13],matrix[14];
    mViewMatrix.linear() = A;
    mViewMatrix.translation() = b;

    auto ground = mEnv->GetGround();
    double y = ground->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape*>(ground->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;

    DrawGround(y);
    DrawTrajectory();
    DrawWalkingDir();
    DrawDetourPerp();
    DrawPush();
    DrawMuscles(mEnv->GetCharacter()->GetMuscles());
    DrawSkeleton(mEnv->GetCharacter()->GetSkeleton());

    // Eigen::Quaterniond q = mTrackBall.getCurrQuat();
    // q.x() = 0.0;
    // q.z() = 0.0;
    // q.normalize();
    // mTrackBall.setQuaternion(q);
    SetFocusing();
}

void
PushWindow::
SetFocusing()
{
    if(mFocus)
    {
        mTrans = -mEnv->GetWorld()->getSkeleton("Human")->getRootBodyNode()->getCOM();
        mTrans[1] -= 0.3;

        mTrans *=1000.0;

    }
}

void
PushWindow::
DrawEntity(const Entity* entity)
{
    if (!entity)
        return;
    const auto& bn = dynamic_cast<const BodyNode*>(entity);
    if(bn)
    {
        DrawBodyNode(bn);
        return;
    }

    const auto& sf = dynamic_cast<const ShapeFrame*>(entity);
    if(sf)
    {
        DrawShapeFrame(sf);
        return;
    }
}
void
PushWindow::
DrawBodyNode(const BodyNode* bn)
{
    if(!bn)
        return;
    if(!mRI)
        return;

    mRI->pushMatrix();
    mRI->transform(bn->getRelativeTransform());

    auto sns = bn->getShapeNodesWith<VisualAspect>();
    for(const auto& sn : sns)
        DrawShapeFrame(sn);

    for(const auto& et : bn->getChildEntities())
        DrawEntity(et);

    mRI->popMatrix();

}
void
PushWindow::
DrawSkeleton(const SkeletonPtr& skel)
{
    DrawBodyNode(skel->getRootBodyNode());
}
void
PushWindow::
DrawShapeFrame(const ShapeFrame* sf)
{
    if(!sf)
        return;

    if(!mRI)
        return;

    const auto& va = sf->getVisualAspect();

    if(!va || va->isHidden())
        return;

    mRI->pushMatrix();
    mRI->transform(sf->getRelativeTransform());

    DrawShape(sf->getShape().get(),va->getRGBA());
    mRI->popMatrix();
}
void
PushWindow::
DrawShape(const Shape* shape,const Eigen::Vector4d& color)
{
    if(!shape)
        return;
    if(!mRI)
        return;

    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST);
    mRI->setPenColor(color);
    if(mDrawOBJ == false)
    {
        if (shape->is<SphereShape>())
        {
            const auto* sphere = static_cast<const SphereShape*>(shape);
            mRI->drawSphere(sphere->getRadius());
        }
        else if (shape->is<BoxShape>())
        {
            const auto* box = static_cast<const BoxShape*>(shape);
            mRI->drawCube(box->getSize());
        }
        else if (shape->is<CapsuleShape>())
        {
            const auto* capsule = static_cast<const CapsuleShape*>(shape);
            mRI->drawCapsule(capsule->getRadius(), capsule->getHeight());
        }
    }
    else
    {
        if (shape->is<MeshShape>())
        {
            const auto& mesh = static_cast<const MeshShape*>(shape);
            glDisable(GL_COLOR_MATERIAL);
            mRI->drawMesh(mesh->getScale(), mesh->getMesh());
            float y = mEnv->GetGround()->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape*>(mEnv->GetGround()->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
            this->DrawShadow(mesh->getScale(), mesh->getMesh(),y);
        }

    }

    glDisable(GL_COLOR_MATERIAL);
}
void
PushWindow::
DrawMuscles(const std::vector<Muscle*>& muscles)
{
    int count =0;
    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);

    for(auto muscle : muscles)
    {
        auto aps = muscle->GetAnchors();
        bool lower_body = true;
        double a = muscle->activation;
        // Eigen::Vector3d color(0.7*(3.0*a),0.2,0.7*(1.0-3.0*a));
        Eigen::Vector4d color(0.4+(2.0*a),0.4,0.4,1.0);//0.7*(1.0-3.0*a));
        // glColor3f(1.0,0.0,0.362);
        // glColor3f(0.0,0.0,0.0);
        mRI->setPenColor(color);
        for(int i=0;i<aps.size();i++)
        {
            Eigen::Vector3d p = aps[i]->GetPoint();
            mRI->pushMatrix();
            mRI->translate(p);
            mRI->drawSphere(0.005*sqrt(muscle->f0/1000.0));
            mRI->popMatrix();
        }

        for(int i=0;i<aps.size()-1;i++)
        {
            Eigen::Vector3d p = aps[i]->GetPoint();
            Eigen::Vector3d p1 = aps[i+1]->GetPoint();

            Eigen::Vector3d u(0,0,1);
            Eigen::Vector3d v = p-p1;
            Eigen::Vector3d mid = 0.5*(p+p1);
            double len = v.norm();
            v /= len;
            Eigen::Isometry3d T;
            T.setIdentity();
            Eigen::Vector3d axis = u.cross(v);
            axis.normalize();
            double angle = acos(u.dot(v));
            Eigen::Matrix3d w_bracket = Eigen::Matrix3d::Zero();
            w_bracket(0, 1) = -axis(2);
            w_bracket(1, 0) =  axis(2);
            w_bracket(0, 2) =  axis(1);
            w_bracket(2, 0) = -axis(1);
            w_bracket(1, 2) = -axis(0);
            w_bracket(2, 1) =  axis(0);


            Eigen::Matrix3d R = Eigen::Matrix3d::Identity()+(sin(angle))*w_bracket+(1.0-cos(angle))*w_bracket*w_bracket;
            T.linear() = R;
            T.translation() = mid;
            mRI->pushMatrix();
            mRI->transform(T);
            mRI->drawCylinder(0.005*sqrt(muscle->f0/1000.0),len);
            mRI->popMatrix();
        }

    }
    glEnable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
}
void
PushWindow::
DrawShadow(const Eigen::Vector3d& scale, const aiScene* mesh,double y)
{
    glDisable(GL_LIGHTING);
    glPushMatrix();
    glScalef(scale[0],scale[1],scale[2]);
    GLfloat matrix[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
    Eigen::Matrix3d A;
    Eigen::Vector3d b;
    A<<matrix[0],matrix[4],matrix[8],
            matrix[1],matrix[5],matrix[9],
            matrix[2],matrix[6],matrix[10];
    b<<matrix[12],matrix[13],matrix[14];

    Eigen::Affine3d M;
    M.linear() = A;
    M.translation() = b;
    M = (mViewMatrix.inverse()) * M;

    glPushMatrix();
    glLoadIdentity();
    glMultMatrixd(mViewMatrix.data());
    DrawAiMesh(mesh,mesh->mRootNode,M,y);
    glPopMatrix();
    glPopMatrix();
    glEnable(GL_LIGHTING);
}
void
PushWindow::
DrawAiMesh(const struct aiScene *sc, const struct aiNode* nd,const Eigen::Affine3d& M,double y)
{
    unsigned int i;
    unsigned int n = 0, t;
    Eigen::Vector3d v;
    Eigen::Vector3d dir(0.4,0,-0.4);
    glColor3f(0.3,0.3,0.3);

    // update transform

    // draw all meshes assigned to this node
    for (; n < nd->mNumMeshes; ++n) {
        const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];

        for (t = 0; t < mesh->mNumFaces; ++t) {
            const struct aiFace* face = &mesh->mFaces[t];
            GLenum face_mode;

            switch(face->mNumIndices) {
                case 1: face_mode = GL_POINTS; break;
                case 2: face_mode = GL_LINES; break;
                case 3: face_mode = GL_TRIANGLES; break;
                default: face_mode = GL_POLYGON; break;
            }
            glBegin(face_mode);
            for (i = 0; i < face->mNumIndices; i++)
            {
                int index = face->mIndices[i];

                v[0] = (&mesh->mVertices[index].x)[0];
                v[1] = (&mesh->mVertices[index].x)[1];
                v[2] = (&mesh->mVertices[index].x)[2];
                v = M*v;
                double h = v[1]-y;

                v += h*dir;

                v[1] = y+0.001;
                glVertex3f(v[0],v[1],v[2]);
            }
            glEnd();
        }

    }

    // draw all children
    for (n = 0; n < nd->mNumChildren; ++n) {
        DrawAiMesh(sc, nd->mChildren[n],M,y);
    }

}

void
PushWindow::
DrawTrajectory()
{
    glDisable(GL_LIGHTING);
    if (mRootTrajectory.size() > 1){
        glColor3f(0., 1., 0.);
        glBegin(GL_LINE_STRIP);
        for (auto &point : mRootTrajectory)
            glVertex3f(point[0], point[1], point[2]);
        glEnd();
    }
    glEnable(GL_LIGHTING);
}

void
PushWindow::
DrawWalkingDir()
{
    if (info_root_pos.size() == 2){
        Eigen::Vector3d walking_point0, walking_point1;
        walking_point0 = info_root_pos[0] - 20.*walking_dir;
        walking_point0[1] = 0.;
        walking_point1 = info_root_pos[1] + 20.*walking_dir;
        walking_point1[1] = 0.;
        glDisable(GL_LIGHTING);
        glColor3f(0., 0., 1.);
        glBegin(GL_LINES);
        glVertex3f(walking_point0[0], info_root_pos[1][1], walking_point0[2]);
        glVertex3f(walking_point1[0], info_root_pos[1][1], walking_point1[2]);
        glEnd();
        glEnable(GL_LIGHTING);
    }
}

void
PushWindow::
DrawDetourPerp()
{
    if (GetSimulationTime() > push_start_time){
        glDisable(GL_LIGHTING);
        glColor3f(1., 0., 0.);
        glBegin(GL_LINES);
        glVertex3f(max_detour_root_pos[0], max_detour_root_pos[1], max_detour_root_pos[2]);
        glVertex3f(max_detour_on_line[0], max_detour_on_line[1], max_detour_on_line[2]);
        glEnd();
        glEnable(GL_LIGHTING);
    }
}

void PushWindow::DrawPush() {
    if ( GetSimulationTime() > push_start_time && GetSimulationTime() < push_end_time) {
        Eigen::Vector3d pushed_body_pos = this->GetBodyPosition("ArmL");
        Eigen::Vector3d push_start_pos = pushed_body_pos - push_force_vec / 100.;
        glDisable(GL_LIGHTING);
        glColor3f(1., 0., 1.);
        glBegin(GL_LINES);
        glVertex3f(pushed_body_pos[0], pushed_body_pos[1], pushed_body_pos[2]);
        glVertex3f(push_start_pos[0], push_start_pos[1], push_start_pos[2]);
        glEnd();
        glEnable(GL_LIGHTING);
    }
}

void
PushWindow::
DrawGround(double y)
{
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    glDisable(GL_LIGHTING);
    double width = 0.005;
    int count = 0;
    glBegin(GL_QUADS);
    for(double x = -100.0;x<100.01;x+=1.0)
    {
        for(double z = -100.0;z<100.01;z+=1.0)
        {
            if(count%2==0)
                glColor3f(216.0/255.0,211.0/255.0,204.0/255.0);
            else
                glColor3f(216.0/255.0-0.1,211.0/255.0-0.1,204.0/255.0-0.1);
            count++;
            glVertex3f(x,y,z);
            glVertex3f(x+1.0,y,z);
            glVertex3f(x+1.0,y,z+1.0);
            glVertex3f(x,y,z+1.0);
        }
    }
    glEnd();
    glEnable(GL_LIGHTING);
}

void
PushWindow::
StepMotion()
{
    double t = mEnv->GetWorld()->getTime();
    double dt = 1./mEnv->GetControlHz();
    Eigen::VectorXd p = mEnv->GetCharacter()->GetTargetPositions(t, 1./mEnv->GetControlHz());
    mEnv->GetCharacter()->GetSkeleton()->setPositions(p);
    mEnv->GetCharacter()->GetSkeleton()->computeForwardKinematics(true,false,false);
    mEnv->GetWorld()->setTime(t + dt);
}
