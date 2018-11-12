#ifndef RENDERING_H
#define RENDERING_H

class Body;

void init_renderer();
void close_renderer();
bool draw(Body* host_bodies);

#endif  // RENDERING_H
