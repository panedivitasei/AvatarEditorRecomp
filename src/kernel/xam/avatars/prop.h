/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_AVATARS_PROP_H_
#define XENIA_AVATARS_PROP_H_

#include <memory>
#include <vector>

#include "animation.h"
#include "asset_pack.h"
#include "blend_shape.h"
#include "common.h"
#include "model.h"
#include "skeleton.h"
#include <rex/types.h>

#include "xe_compat.h"

namespace rex {
namespace avatars {

struct PropLoadOptions {
  ModelLoadOptions model;
  SkeletonLoadOptions skeleton;
  AnimationLoadOptions animation;
  BlendShapeLoadOptions blend_shape;
};

class Prop {
 public:
  std::shared_ptr<Model> model;
  std::shared_ptr<Skeleton> skeleton;
  std::shared_ptr<Animation> animation;

 public:
  static std::shared_ptr<Prop> Load(const uint8_t* strb_buffer,
                                    size_t strb_size,
                                    const PropLoadOptions& load_options);
};

}  // namespace avatars
}  // namespace rex

#endif  // XENIA_AVATARS_PROP_H_
