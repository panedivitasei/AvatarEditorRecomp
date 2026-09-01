/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_AVATARS_BLEND_SHAPE_APPLY_H_
#define XENIA_AVATARS_BLEND_SHAPE_APPLY_H_

#include <memory>
#include <vector>

#include "asset_pack.h"
#include "blend_shape.h"
#include "common.h"
#include "model.h"
#include <rex/types.h>

#include "xe_compat.h"

namespace rex {
namespace avatars {

bool ApplyBlendShape(std::shared_ptr<BlendShape> blend_shape,
                     const AssetId& model_asset_id,
                     std::shared_ptr<Model> model);

// Detect and repair an item authored against the pre-Fall-2010 body revision.
// Such items share the current body's topology and so pass every buffer-size
// gate, but their garment mesh and their hiding shape's absolute tuck
// positions live in the older, smaller body space.
//
// With legacy_body_model (the pre-2010 body from AvatarAssetPackLegacyV1,
// which shares topology with the current body) the repair is exact: tucks are
// re-based per addressed vertex (current position + authored tuck delta), and
// garment vertices are warped by the displacement field of their nearest
// legacy-body vertices, so hems and rims land on the matching current body
// region. Falls back to a per-axis affine fit, which extrapolates poorly away
// from the tucked region, when the legacy body is unavailable or its topology
// mismatches.
// item_model may be null (shape-only assets); body_model must be the model
// the shape targets (caller pre-matched ids and buffer sizes).
bool RescueOldBodyItem(std::shared_ptr<BlendShape> blend_shape,
                       std::shared_ptr<Model> body_model,
                       std::shared_ptr<Model> item_model,
                       std::shared_ptr<Model> legacy_body_model);

}  // namespace avatars
}  // namespace rex

#endif  // XENIA_AVATARS_BLEND_SHAPE_APPLY_H_
