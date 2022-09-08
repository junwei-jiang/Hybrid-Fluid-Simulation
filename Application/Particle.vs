#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aOffset;
layout(location = 2) in vec3 aColor;

out vec4 Color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
  vec3 Position = aPos + aOffset;
  gl_Position = projection * view * model * vec4(-Position.x - 10.0, Position.yz, 1.0f);
  Color = vec4(aColor, 1.0f);
}