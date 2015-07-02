#include "common/common_types.h"
#include "video_core/vertex_shader.h"

namespace Pica {

namespace VertexShaderFast {

void LoadShader();

VertexShader::OutputVertex RunShader(const VertexShader::InputVertex& input, int num_attributes, int core_num);

} // namespace VertexShaderFast

} // namespace Pica
