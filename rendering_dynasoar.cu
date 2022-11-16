#include <SDL2/SDL.h>

#include "configuration.h"
#include "rendering_dynasoar.h"

static const int kWindowWidth = 1000;
static const int kWindowHeight = 1000;
static const int kMaxRect = 20;

// SDL rendering variables.
SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;

void init_frame() {
  // Clear scene.
  SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
  SDL_RenderClear(renderer);
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
}

bool show_frame() {
  SDL_RenderPresent(renderer);

  // Continue until the user closes the window.
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    if (event.type == SDL_QUIT) {
      return false;
    }
  }

  return true;
}

void draw_body(float pos_x, float pos_y, float mass) {
  SDL_Rect rect;
  rect.w = rect.h = mass / kMaxMass * kMaxRect;
  rect.x = (pos_x/2 + 0.5) * kWindowWidth - rect.w/2;
  rect.y = (pos_y/2 + 0.5) * kWindowHeight - rect.h/2;
  SDL_RenderDrawRect(renderer, &rect);
}

void init_renderer() {
  // Initialize graphical output.
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    printf("Could not initialize SDL!\n");
    exit(1);
  }

  if (SDL_CreateWindowAndRenderer(kWindowWidth, kWindowHeight, 0,
        &window, &renderer) != 0) {
    printf("Could not create window/render!\n");
    exit(1);
  }
}

void close_renderer() {
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
}
