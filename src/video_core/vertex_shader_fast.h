#include "common/common_types.h"
#include "video_core/vertex_shader.h"

namespace Pica {

namespace VertexShaderFast {

//struct Register {
//public:
//    union {
//        struct {
//            f32 x;
//            f32 y;
//            f32 z;
//            f32 w;
//        };
//        f32 value[4];
//    };
//
//    Register() = default;
//    Register(const f32 a[4]) : x(a[0]), y(a[1]), z(a[2]), w(a[3]) {}
//    Register(f32 x, f32 y, f32 z, f32 w) : x(x), y(y), z(z), w(w) {}
//
//    Register operator +(const Register& other) const {
//        return Register(x + other.x, y + other.y, z + other.z, w + other.w);
//    }
//
//    Register operator *(const Register& other) const {
//        return Register(x * other.x, y * other.y, z * other.z, w * other.w);
//    }
//
//    Register Dot3(const Register& other) const {
//        f32 dot = x * other.x + y * other.y + z * other.z;
//        return Register(dot, dot, dot, dot);
//    }
//
//    Register Dot4(const Register& other) const {
//        f32 dot = x * other.x + y * other.y + z * other.z + w * other.w;
//        return Register(dot, dot, dot, dot);
//    }
//};
//static_assert(sizeof(Register) == 0x10, "Incorrect structure size");

VertexShader::OutputVertex RunShader(const VertexShader::InputVertex& input, int num_attributes);

} // namespace VertexShaderFast

} // namespace Pica
