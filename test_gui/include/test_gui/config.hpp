#ifndef CONFIG_H
#define CONFIG_H

/*****************************************************************************
** Define
*****************************************************************************/
#define LINEAR                              0
#define ANGULAR                             1
#define LEFT                                0
#define RIGHT                               1

#define TICK2RAD                            0.00553845 // [deg] * 3.14159265359 / 180
#define DEG2RAD(x)                          (x * 0.01745329252)  // *PI/180
#define RAD2DEG(x)                          (x * 57.2957795131)  // *180/PI

#define ARM_JOINT_NUM                       6
#define ARM_JOINT_VEL_LIMIT                 0.00150 // [rad/s]

#define BOXID1                              1
#define BOXID2                              2
#define BOXID3                              3
#define BOXID4                              4


#endif // CONFIG_H
