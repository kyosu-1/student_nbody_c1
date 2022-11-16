#include <chrono>
#include <curand_kernel.h>

#include "barnes_hut.h"
#include "../configuration.h"

#ifdef OPTION_RENDER
#include "../rendering.h"
#endif  // OPTION_RENDER


// Allocator handles.
AllocatorHandle<AllocatorT>* allocator_handle;
__device__ AllocatorT* device_allocator;


// Root of the quad tree.
__DEV__ TreeNode* tree;
__DEV__ bool bfs_root_done;


template<typename T>
__DEV__ T* pointerCAS(T** addr, T* assumed, T* value) {
  auto* i_addr = reinterpret_cast<unsigned long long int*>(addr);
  auto i_assumed = reinterpret_cast<unsigned long long int>(assumed);
  auto i_value = reinterpret_cast<unsigned long long int>(value);
  return reinterpret_cast<T*>(atomicCAS(i_addr, i_assumed, i_value));
}


__DEV__ NodeBase::NodeBase(TreeNode* parent, double pos_x, double pos_y,
                           double mass)
    : parent_(parent), pos_x_(pos_x), pos_y_(pos_y), mass_(mass) {}


__DEV__ BodyNode::BodyNode(double pos_x, double pos_y, double vel_x, double vel_y,
                           double mass)
    : NodeBase(/*parent=*/ nullptr, pos_x, pos_y, mass),
      vel_x_(vel_x), vel_y_(vel_y) {}


__DEV__ TreeNode::TreeNode(TreeNode* parent, double p1_x, double p1_y,
                           double p2_x, double p2_y)
    : NodeBase(parent, 0.0f, 0.0f, 0.0f),
      p1_x_(p1_x), p1_y_(p1_y), p2_x_(p2_x), p2_y_(p2_y) {
  assert(p1_x < p2_x);
  assert(p1_y < p2_y);

  for (int i = 0; i < 4; ++i) {
    children_[i] = nullptr;
  }

  if (parent != nullptr) { assert(parent->contains(this)); }
}


__DEV__ double NodeBase::distance_to(NodeBase* other) {
  double dx = other->pos_x() - pos_x_;
  double dy = other->pos_y() - pos_y_;
  return sqrt(dx*dx + dy*dy);
}


__DEV__ void NodeBase::apply_force(BodyNode* body) {
  // Update `body`.
  if (body != this) {
    double dx = pos_x_ - body->pos_x();
    double dy = pos_y_ - body->pos_y();
    double dist = sqrt(dx*dx + dy*dy);
    double F = kGravityConstant * mass_ * body->mass()
        / (dist * dist + kDampeningFactor);
    body->add_force(F*dx / dist, F*dy / dist);
  }
}


__DEV__ void BodyNode::compute_force() {
  force_x_ = 0.0f;
  force_y_ = 0.0f;

  // TODO: We may need a while loop here instead of recursion.
  tree->check_apply_force(this);
}


__DEV__ void NodeBase::check_apply_force(BodyNode* body) {
  // TODO: This function should be virtual but we do not have native support
  // for virtual functions in SoaAlloc yet.
  TreeNode* tree_node = this->cast<TreeNode>();
  if (tree_node != nullptr) {
    tree_node->check_apply_force(body);
  } else {
    BodyNode* body_node = this->cast<BodyNode>();
    if (body_node != nullptr) {
      body_node->check_apply_force(body);
    } else {
      assert(false);
    }
  }
}


__DEV__ void TreeNode::check_apply_force(BodyNode* body) {
  if (contains(body) || distance_to(body) <= kDistThreshold) {
    // Too close. Recurse.
    for (int i = 0; i < 4; ++i) {
      if (children_[i] != nullptr) {
        children_[i]->check_apply_force(body);
      }
    }
  } else {
    // Far enough away to use approximation.
    apply_force(body);
  }
}


__DEV__ void BodyNode::check_apply_force(BodyNode* body) {
  apply_force(body);
}


__DEV__ void BodyNode::update() {
  vel_x_ += force_x_*kDt / mass_;
  vel_y_ += force_y_*kDt / mass_;
  pos_x_ += vel_x_*kDt;
  pos_y_ += vel_y_*kDt;

  if (pos_x_ < -1) {
    vel_x_ = -vel_x_;
    pos_x_ = -1.0f;
  } else if (pos_x_ > 0.99999999) {
    vel_x_ = -vel_x_;
    pos_x_ = 0.99999999;
  }

  if (pos_y_ < -1) {
    vel_y_ = -vel_y_;
    pos_y_ = -1.0f;
  } else if (pos_y_ > 0.99999999) {
    vel_y_ = -vel_y_;
    pos_y_ = 0.99999999;
  }
}


__DEV__ void BodyNode::clear_node() {
  assert(parent_ != nullptr);

  // Remove node if:
  // (a) Moved to another parent.
  // (b) Moved to another segment in the same tree node.
  if (parent_->child_index(this) != child_index_) {
    parent_->remove(this);
    parent_ = nullptr;
  }
}


__DEV__ void TreeNode::remove(NodeBase* node) {
  assert(children_[node->child_index_] == node);
  children_[node->child_index_] = nullptr;
}


__DEV__ void BodyNode::add_to_tree() {
  if (parent_ == nullptr) {
    TreeNode* current = tree;
    bool is_done = false;

    while (!is_done) {
      assert(current->contains(this));
      assert(current->cast<TreeNode>() != nullptr);

      // Check where to insert in this node.
      int c_idx = current->child_index(this);
      auto** child_ptr = &current->children_[c_idx];

      // TODO: Read volatile?
      NodeBase* child = *child_ptr;

      if (child == nullptr) {
        // Slot not in use.
        if (pointerCAS<NodeBase>(child_ptr, nullptr, this) == nullptr) {
          // Ensure that parent pointer updates are sequentially consistent.
          // this->set_parent(current);
          auto* p_before = pointerCAS<TreeNode>(&parent_, nullptr, current);
          assert(p_before == nullptr);
          child_index_ = c_idx;

          // Note: while(true) with break deadlocks due to unfortunate divergent
          // warp branch scheduling.
          is_done = true;
        }
      } else if (child->cast<TreeNode>() != nullptr) {
        // There is a subtree here.
        assert(current->contains(child->cast<TreeNode>()));
        current = child->cast<TreeNode>();
      } else {
        // There is a Body here.
        BodyNode* other = child->cast<BodyNode>();
        assert(other != nullptr);

        assert(current->contains(other));
        assert(current->child_index(other) == c_idx);

        // Replace BodyNode with TreeNode.
        auto* new_node = current->make_child_tree_node(c_idx);
        new_node->child_index_ = c_idx;
        assert(new_node->contains(other));
        assert(new_node->contains(this));

        // Insert other into new node.
        int other_c_idx = new_node->child_index(other);
        new_node->children_[other_c_idx] = other;
        __threadfence();

        // Try to install this node. (Retry.)
        if (pointerCAS<NodeBase>(child_ptr, other, new_node) == other) {
          // other->set_parent(new_node);
          // It may take a while until we see the correct parent, because
          // another may not be done inserting this node yet.
          TreeNode* parent_before = nullptr;

          {
#ifndef NDEBUG
            int retries = 10000;
#endif  // NDEBUG
            do {
              parent_before = pointerCAS<TreeNode>(
                  &other->parent_, current, new_node);
#ifndef NDEBUG
              //assert(--retries > 0);
              if (--retries < 0) {
                assert(false);
              }
#endif  // NDEBUG
            } while (parent_before != current);

            other->child_index_ = other_c_idx;
          }

          // Now insert body.
          current = new_node;
        } else {
          // Another thread installed a node here. Rollback.
          destroy(device_allocator, new_node);
        }
      }
    }
  }
}


__DEV__ int TreeNode::child_index(BodyNode* body) {
  //                            p2
  // (-1, 1)   |-----------|  (1, 1)
  //           |  2  |  3  |
  // (-1, 0)   |-----|-----|  (1, 0)
  //           |  0  |  1  |
  // (-1, -1)  |-----------|  (1, -1)
  //    p1

  float p1x = p1_x_;  float p1y = p1_y_;
  float p2x = p2_x_;  float p2y = p2_y_;

  if (body->pos_x() < p1x || body->pos_x() >= p2x
      || body->pos_y() < p1y || body->pos_y() >= p2y) {
    // Out of bounds.
    return -1;
  } else {
    assert(contains(body));
    int c_idx = 0;
    if (body->pos_x() > (p1x + p2x) / 2) c_idx = 1;
    if (body->pos_y() > (p1y + p2y) / 2) c_idx += 2;
    return c_idx;
  }
}


__DEV__ TreeNode* TreeNode::make_child_tree_node(int c_idx) {
  double new_p1_x = (c_idx == 0 || c_idx == 2) ? p1_x_ : (p1_x_ + p2_x_) / 2;
  double new_p2_x = (c_idx == 0 || c_idx == 2) ? (p1_x_ + p2_x_) / 2 : p2_x_;
  double new_p1_y = (c_idx == 0 || c_idx == 1) ? p1_y_ : (p1_y_ + p2_y_) / 2;
  double new_p2_y = (c_idx == 0 || c_idx == 1) ? (p1_y_ + p2_y_) / 2 : p2_y_;
  auto* result = new(device_allocator) TreeNode(
      /*parent=*/ this, new_p1_x, new_p1_y, new_p2_x, new_p2_y);

  return result;
}


__DEV__ void TreeNode::collapse_tree() {
  if (bfs_frontier_) {
    bfs_frontier_ = false;

    // Count children.
    int num_children = 0;
    NodeBase* single_child = nullptr;

    for (int i = 0; i < 4; ++i) {
      if (children_[i] != nullptr) {
        ++num_children;
        single_child = children_[i];
      }
    }

    if (num_children == 0) {
      // Remove node without children.
      if (parent_ != nullptr) {
        parent_->remove(this);
        destroy(device_allocator, this);
        return;
      } else { assert(parent_ == tree); }
    } else if (num_children == 1) {
      // Store child in parent.
      if (parent_ != nullptr) {
        if (single_child->cast<BodyNode>() != nullptr) {
          // This is a Body, store in parent.
          assert(parent_->children_[child_index_] == this);
          single_child->parent_ = parent_;
          single_child->child_index_ = child_index_;
          parent_->children_[child_index_] = single_child;
          assert(parent_->contains(single_child));

          destroy(device_allocator, this);
          return;
        }  // else: TreeNode child cannot be collapsed.
      } else { assert(parent_ == tree); }
    }

    // Ready propagate result to parent.
    bfs_done_ = true;

    if (this == tree) {
      // Terminate algorithm loop.
      assert(parent_ == nullptr);
      bfs_root_done = true;
    }
  }
}


__DEV__ bool TreeNode::is_leaf() {
  for (int i = 0; i < 4; ++i) {
    if (children_[i]->cast<TreeNode>() != nullptr) {
      return false;
    }
  }

  return true;
}


__DEV__ int TreeNode::num_direct_children() {
  int counter = 0;

  for (int i = 0; i < 4; ++i) {
    if (children_[i] != nullptr) { ++counter; }
  }

  return counter;
}


__DEV__ void TreeNode::check_consistency() {
  // There should be no empty nodes.
  assert(num_direct_children() > 0);

  for (int i = 0; i < 4; ++i) {
    NodeBase* node = children_[i];
    if (node != nullptr) {
      assert(node->parent_ == this);

      if (node->cast<BodyNode>() != nullptr) {
        assert(child_index(node->cast<BodyNode>()) == i);
      }
    }
  }
}


__DEV__ bool TreeNode::contains(BodyNode* body) {
  double x = body->pos_x();
  double y = body->pos_y();
  return x >= p1_x_ && x < p2_x_ && y >= p1_y_ && y < p2_y_;
}


__DEV__ bool TreeNode::contains(TreeNode* node) {
  return node->p1_x_ >= p1_x_ && node->p2_x_ <= p2_x_
      && node->p1_y_ >= p1_y_ && node->p2_y_ <= p2_y_;
}


__DEV__ bool TreeNode::contains(NodeBase* node) {
  if (node->cast<BodyNode>() != nullptr) {
    return contains(node->cast<BodyNode>());
  }
  else if (node->cast<TreeNode>() != nullptr) {
    return contains(node->cast<TreeNode>());
  } else {
    assert(false);
  }

  return false;
}


__DEV__ void TreeNode::initialize_frontier() {
  bfs_frontier_ = is_leaf();
  bfs_done_ = false;
  bfs_root_done = false;
}


__DEV__ void TreeNode::update_frontier() {
  if (!bfs_done_) {
    for (int i = 0; i < 4; ++i) {
      TreeNode* child = children_[i]->cast<TreeNode>();
      if (child != nullptr && !child->bfs_done_) { return; }
    }

    // All child nodes are done. Put in frontier.
    bfs_frontier_ = true;
  }
}


__DEV__ void TreeNode::bfs_step() {
  if (bfs_frontier_) {
    bfs_frontier_ = false;

    // Update pos_x and pos_y: gravitational center
    double total_mass = 0.0f;
    double sum_pos_x = 0.0f;
    double sum_pos_y = 0.0f;

    for (int i = 0; i < 4; ++i) {
      if (children_[i] != nullptr) {
        total_mass += children_[i]->mass();
        sum_pos_x += children_[i]->mass()*children_[i]->pos_x();
        sum_pos_y += children_[i]->mass()*children_[i]->pos_y();
      }
    }

    mass_ = total_mass;
    pos_x_ = sum_pos_x/total_mass;
    pos_y_ = sum_pos_y/total_mass;

    // Ready propagate result to parent.
    bfs_done_ = true;

    if (this == tree) {
      // Terminate algorithm loop.
      bfs_root_done = true;
    }
  }
}


void step() {
  // Generate tree summaries.
  allocator_handle->parallel_do<TreeNode, &TreeNode::initialize_frontier>();
  bool root_done = false;

  while (!root_done) {
    allocator_handle->fast_parallel_do<TreeNode, &TreeNode::bfs_step>();
    allocator_handle->fast_parallel_do<TreeNode, &TreeNode::update_frontier>();

    cudaMemcpyFromSymbol(&root_done, bfs_root_done, sizeof(bool), 0,
                         cudaMemcpyDeviceToHost);
  }

  // N-Body simulation.
  allocator_handle->parallel_do<BodyNode, &BodyNode::compute_force>();
  allocator_handle->fast_parallel_do<BodyNode, &BodyNode::update>();
  allocator_handle->fast_parallel_do<BodyNode, &BodyNode::clear_node>();

  // Re-insert nodes into the tree.
  allocator_handle->fast_parallel_do<BodyNode, &BodyNode::add_to_tree>();

  // Collapse the tree (remove empty TreeNodes).
  allocator_handle->parallel_do<TreeNode, &TreeNode::initialize_frontier>();
  root_done = false;

  while (!root_done) {
    allocator_handle->fast_parallel_do<TreeNode, &TreeNode::collapse_tree>();
    allocator_handle->parallel_do<TreeNode, &TreeNode::update_frontier>();

    cudaMemcpyFromSymbol(&root_done, bfs_root_done, sizeof(bool), 0,
                         cudaMemcpyDeviceToHost);
  }

#ifndef NDEBUG
  allocator_handle->parallel_do<TreeNode, &TreeNode::check_consistency>();
#endif  // NDEBUG
}


__global__ void kernel_init_tree() {
  tree = new(device_allocator) TreeNode(
      /*parent=*/ nullptr,
      /*p1_x=*/ -1.0f,
      /*p1_y=*/ -1.0f,
      /*p2_x=*/ 1.0f,
      /*p2_y=*/ 1.0f);
}


__global__ void kernel_init_bodies() {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  curandState rand_state;
  curand_init(kSeed, tid, 0, &rand_state);

  for (int i = tid; i < kNumBodies; i += blockDim.x * gridDim.x) {
    new(device_allocator) BodyNode(
        /*pos_x=*/ 1.98 * curand_uniform(&rand_state) - 0.99,
        /*pos_y=*/ 1.98 * curand_uniform(&rand_state) - 0.99,
        /*vel_x=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
        /*vel_y=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
        /*mass=*/ (curand_uniform(&rand_state)/2 + 0.5) * kMaxMass);
  }
}


void initialize_simulation() {
  kernel_init_tree<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_init_bodies<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());

  allocator_handle->parallel_do<BodyNode, &BodyNode::add_to_tree>();

#ifndef NDEBUG
  allocator_handle->parallel_do<TreeNode, &TreeNode::check_consistency>();
#endif  // NDEBUG
}


int main(int /*argc*/, char** /*argv*/) {
#ifdef OPTION_RENDER
  init_renderer();
#endif  // OPTION_RENDER

  // Create new allocator.
  allocator_handle = new AllocatorHandle<AllocatorT>(/*unified_memory=*/ true);
  AllocatorT* dev_ptr = allocator_handle->device_pointer();
  cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                     cudaMemcpyHostToDevice);

  initialize_simulation();

#ifdef OPTION_RENDER
  // TODO: We could just read this off tree->mass() after the first iteration.
  float max_mass = 0.0f;

  allocator_handle->template device_do<BodyNode>([&](BodyNode* body) {
    max_mass += body->mass();
  });
#endif  // OPTION_RENDER

  auto time_start = std::chrono::system_clock::now();

  for (int i = 0; i < kIterations; ++i) {
    step();

#ifdef OPTION_RENDER
    init_frame();

    allocator_handle->template device_do<TreeNode>([&](TreeNode* node) {
      draw_tree_node(node->p1_x(), node->p1_y(), node->p2_x(), node->p2_y());
    });

    allocator_handle->template device_do<BodyNode>([&](BodyNode* body) {
      draw_body(body->pos_x(), body->pos_y(), body->mass(), max_mass);
    });

    show_frame();
#endif  // OPTION_RENDER
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_start;
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(elapsed)
      .count();

  printf("%lu, %lu\n", micros, allocator_handle->DBG_get_enumeration_time());

#ifdef OPTION_RENDER
  close_renderer();
#endif  // OPTION_RENDER

  return 0;
}
