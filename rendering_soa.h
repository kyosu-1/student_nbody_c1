#ifndef RENDERING_SOA__H
#define RENDERING_SOA__H

#include "configuration.h"

void init_renderer();
void close_renderer();
void draw(float* host_Body_pos_x, float* host_Body_pos_y,
          float* host_Body_mass, int num_bodies = kNumBodies);

#endif  // RENDERING_SOA__H
