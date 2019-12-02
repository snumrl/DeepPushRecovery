from pushrecoverybvhgenerator import pBvhFileIO as bvf
from pushrecoverybvhgenerator import pJointMotionEdit as jed
from pushrecoverybvhgenerator import pMath as mth
from pushrecoverybvhgenerator.pParamedStep import ParamedStep_3
from pushrecoverybvhgenerator import pJointMotion as jmt


def get_paramed_bvh(ori_bvh_path, crouch_angle, step_length, walk_speed):
    origmot = bvf.readBvhFile_JointMotion(ori_bvh_path)
    jed.alignMotionToOrigin(origmot)
    origmot = origmot[41:106]
    paramedStep = ParamedStep_3(origmot, mth.v3(0, 0, 1), 'Character1_RightUpLeg', 'Character1_LeftUpLeg',
                                'Character1_RightFoot', 'Character1_LeftFoot',
                                'Character1_RightFoot', 'Character1_LeftFoot', 'Character1_RightToeBase', 'Character1_RightLeg')

    p = paramedStep.calcDefaultParam()
    p[0], p[1], p[2] = crouch_angle, step_length, walk_speed
    endPosture = jed.getMirroredJointMotion(jmt.JointMotion(paramedStep.getParamAppliedEndPosture(p)), 'yz')
    orig2, orig2_fixed = paramedStep.getParamAppliedMotion(p, endPosture[0])
    orig2_fixed_mirrored = jed.getMirroredJointMotion(orig2_fixed, 'yz')
    d_orig2 = orig2[-1] - orig2_fixed_mirrored[0]
    for i in range(len(orig2_fixed_mirrored)):
        orig2_fixed_mirrored[i] += d_orig2
    bvh = bvf.Bvh()
    bvh.fromJointMotion(orig2_fixed[:-1]+orig2_fixed_mirrored[:-1])

    return str(bvh)


if __name__ == '__main__':
    import sys
    print(get_paramed_bvh(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])))
