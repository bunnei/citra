#include "common/common_types.h"
#include "video_core/vertex_shader.h"

namespace Pica {

namespace VertexShaderFast {

VertexShader::OutputVertex RunShader(const VertexShader::InputVertex& input, int num_attributes);

} // namespace VertexShaderFast

} // namespace Pica
