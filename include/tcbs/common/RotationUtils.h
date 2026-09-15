#ifndef TCBS_ROTATION_UTILS_H
#define TCBS_ROTATION_UTILS_H

// ============================================================================
// RotationUtils.h — 3×3 旋转矩阵与欧拉角小工具（ZXY 约定: R = Rz(yaw)·Rx(pitch)·Ry(roll)）
//
// 与旧版 FusionFilter 的约定保持一致（pitch 绕 x），供状态估计器与反解使用。
// ============================================================================
#include <cmath>

namespace tcbs {

namespace rot {

struct Mat3 {
    double m[3][3];
};

inline Mat3 identity() {
    Mat3 R{};
    R.m[0][0] = R.m[1][1] = R.m[2][2] = 1.0;
    return R;
}

inline Mat3 rotX(double a) {
    Mat3 R = identity();
    const double c = std::cos(a), s = std::sin(a);
    R.m[1][1] = c; R.m[1][2] = -s;
    R.m[2][1] = s; R.m[2][2] = c;
    return R;
}

inline Mat3 rotY(double a) {
    Mat3 R = identity();
    const double c = std::cos(a), s = std::sin(a);
    R.m[0][0] = c;  R.m[0][2] = s;
    R.m[2][0] = -s; R.m[2][2] = c;
    return R;
}

inline Mat3 rotZ(double a) {
    Mat3 R = identity();
    const double c = std::cos(a), s = std::sin(a);
    R.m[0][0] = c; R.m[0][1] = -s;
    R.m[1][0] = s; R.m[1][1] = c;
    return R;
}

// ZXY 欧拉角 → 矩阵（R = Rz(yaw)·Rx(pitch)·Ry(roll)）
inline Mat3 eulerZXY(double yaw, double pitch, double roll) {
    const double cy = std::cos(yaw),  sy = std::sin(yaw);
    const double cp = std::cos(pitch), sp = std::sin(pitch);
    const double cr = std::cos(roll), sr = std::sin(roll);
    Mat3 R;
    R.m[0][0] = cy * cr - sy * sp * sr;  R.m[0][1] = -sy * cp;  R.m[0][2] = cy * sr + sy * sp * cr;
    R.m[1][0] = sy * cr + cy * sp * sr;  R.m[1][1] =  cy * cp;  R.m[1][2] = sy * sr - cy * sp * cr;
    R.m[2][0] = -cp * sr;                R.m[2][1] =  sp;       R.m[2][2] = cp * cr;
    return R;
}

inline Mat3 mul(const Mat3& A, const Mat3& B) {
    Mat3 R;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R.m[i][j] = A.m[i][0] * B.m[0][j] + A.m[i][1] * B.m[1][j] + A.m[i][2] * B.m[2][j];
    return R;
}

inline Mat3 transpose(const Mat3& A) {
    Mat3 R;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) R.m[i][j] = A.m[j][i];
    return R;
}

inline void mulVec(const Mat3& A, const double v[3], double out[3]) {
    for (int i = 0; i < 3; ++i)
        out[i] = A.m[i][0] * v[0] + A.m[i][1] * v[1] + A.m[i][2] * v[2];
}

// ZXY 欧拉角提取: pitch = asin(M21), yaw = atan2(-M01, M11), roll = atan2(-M20, M22)
inline void matToEulerZXY(const Mat3& R, double& yaw, double& pitch, double& roll) {
    pitch = std::asin(std::max(-1.0, std::min(1.0, R.m[2][1])));
    const double eps = 1e-6;
    if (std::fabs(std::cos(pitch)) > eps) {
        yaw  = std::atan2(-R.m[0][1], R.m[1][1]);
        roll = std::atan2(-R.m[2][0], R.m[2][2]);
    } else {
        roll = 0.0;
        yaw  = std::atan2(R.m[1][0], R.m[0][0]);
    }
}

// 解卷绕: 将 angle 延展为与 prev 连续的等价角（修正量累计到 corr）
inline double unwrapTo(double angle, double prev, double& corr) {
    double val = angle + corr;
    double diff = val - prev;
    while (diff >  M_PI) { corr -= 2.0 * M_PI; diff -= 2.0 * M_PI; }
    while (diff < -M_PI) { corr += 2.0 * M_PI; diff += 2.0 * M_PI; }
    return angle + corr;
}

} // namespace rot

} // namespace tcbs

#endif // TCBS_ROTATION_UTILS_H