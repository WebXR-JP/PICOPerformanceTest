#version 450
#extension GL_EXT_multiview : enable

layout(push_constant) uniform PushConstants {
    mat4 mvp[2];   // [0]=左目, [1]=右目
    int aluIters;  // ALU負荷テスト用ループ回数（0=無効）
} pc;

layout(location = 0) in vec3 inPosition;

void main() {
    // ALU負荷: sinループ（デッドコード除去されないよう出力に微小加算）
    float dummy = 0.0;
    for (int i = 0; i < pc.aluIters; i++) {
        dummy += sin(float(i) * 0.001 + inPosition.x);
    }
    gl_Position = pc.mvp[gl_ViewIndex] * vec4(inPosition, 1.0) + vec4(dummy * 0.000001, 0.0, 0.0, 0.0);
}
