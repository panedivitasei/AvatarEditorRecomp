/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "blend_shape_apply.h"

#include <cmath>
#include <utility>

#include "blend_shape.h"
#include "common.h"
#include "model.h"
#include <rex/logging.h>

#include "xe_compat.h"

namespace rex {
namespace avatars {

static void ApplyIndexPatch(const BlendShapeIndexPatch& patch,
                            std::shared_ptr<Model> model) {
  std::vector<size_t> offsets(model->triangle_batches.size() + 1);
  {
    size_t offset = 0;
    size_t i;
    for (i = 0; i < model->triangle_batches.size(); ++i) {
      offsets[i] = offset;
      offset += model->triangle_batches[i].triangle_count;
    }
    offsets[i] = offset;
  }
  for (size_t i = 0; i < patch.indices.size(); ++i) {
    size_t index = patch.indices[i];
    // Bound j by the batch count (offsets has batch_count+1 entries), not by
    // the triangle index: an out-of-range index from a malformed shape would
    // otherwise read offsets[] out of bounds, and index 0 would never patch.
    for (size_t j = 1; j <= model->triangle_batches.size(); ++j) {
      if (index < offsets[j]) {
        j--;
        index -= offsets[j];
        auto& indices = model->triangle_batches[j].indices;
        int value = indices[index * 3 + 0];
        indices[index * 3 + 1] = value;
        indices[index * 3 + 2] = value;
        break;
      }
    }
  }
}

static void ApplyVertexPatch(const BlendShapeVertexPatch& patch,
                             std::shared_ptr<Model> model) {
  std::vector<size_t> offsets(model->triangle_batches.size() + 1);
  std::vector<size_t> sizes(model->triangle_batches.size());
  {
    size_t offset = 0;
    size_t i;
    for (i = 0; i < model->triangle_batches.size(); ++i) {
      const auto& triangleBatch = model->triangle_batches[i];
      offsets[i] = offset;
      sizes[i] = triangleBatch.vertex_size;
      offset += triangleBatch.vertices.size() * triangleBatch.vertex_size;
    }
    offsets[i] = offset;
  }
  // Body-hiding tucks only nudge rim vertices under the garment. Shapes
  // authored against the pre-Fall-2010 body revision carry absolute positions
  // from that older, smaller mesh (same topology, so every buffer-size gate
  // passes) and would drag current body vertices far out of place, so skip
  // implausible tucks per vertex and keep the topology-based hiding.
  // Face/hair morphs target non-body assets and are exempt.
  const bool clamp_tucks = patch.original_asset_id.a.get() == 2u;
  constexpr float kMaxTuckDistance = 0.04f;
  for (size_t i = 0; i < patch.vertices.size(); ++i) {
    const auto& vertexPatch = patch.vertices[i];
    size_t originalOffset = vertexPatch.original_offset;
    for (size_t j = 1; j <= model->triangle_batches.size(); ++j) {
      if (originalOffset < offsets[j]) {
        j--;
        size_t index = (originalOffset - offsets[j]) / sizes[j];
        auto& vertex = model->triangle_batches[j].vertices[index];
        if (clamp_tucks) {
          const float dx = vertex.position.x - vertexPatch.position.x;
          const float dy = vertex.position.y - vertexPatch.position.y;
          const float dz = vertex.position.z - vertexPatch.position.z;
          if (dx * dx + dy * dy + dz * dz > kMaxTuckDistance * kMaxTuckDistance) {
            break;
          }
        }
        vertex.position = vertexPatch.position;
        vertex.normal = vertexPatch.normal;
        vertex.blend_weight = vertexPatch.blend_weight;
        vertex.blend_indices = vertexPatch.blend_indices;
        vertex.color = vertexPatch.color;
        break;
      }
    }
  }
}

// Flattens a model's vertices into one position array (batch order), the
// index space the vertex-patch byte offsets address after dividing by
// vertex_size. Returns false when batch layouts differ from `like`.
static bool FlattenPositions(const std::shared_ptr<Model>& model,
                             const std::shared_ptr<Model>& like,
                             std::vector<Vector3<float>>& out) {
  if (like && model->triangle_batches.size() != like->triangle_batches.size()) {
    return false;
  }
  out.clear();
  for (size_t i = 0; i < model->triangle_batches.size(); ++i) {
    const auto& batch = model->triangle_batches[i];
    if (like) {
      const auto& other = like->triangle_batches[i];
      if (batch.vertices.size() != other.vertices.size() ||
          batch.vertex_size != other.vertex_size) {
        return false;
      }
    }
    for (const auto& v : batch.vertices) {
      out.push_back(v.position);
    }
  }
  return true;
}

bool RescueOldBodyItem(std::shared_ptr<BlendShape> blend_shape,
                       std::shared_ptr<Model> body_model,
                       std::shared_ptr<Model> item_model,
                       std::shared_ptr<Model> legacy_body_model) {
  auto& patch = blend_shape->vertex_patch;
  if (patch.original_asset_id.a.get() != 2u || patch.vertices.size() < 4) {
    return false;
  }
  // Only trust the fit when the patch was authored for this body's buffer
  // (same gate ApplyBlendShape uses for the vertex patch).
  size_t total_vertex_buffer_size = 0;
  for (const auto& batch : body_model->triangle_batches) {
    total_vertex_buffer_size += batch.vertices.size() * batch.vertex_size;
  }
  total_vertex_buffer_size = (total_vertex_buffer_size + 127) & ~127;
  if (total_vertex_buffer_size != patch.total_buffer_size) {
    return false;
  }

  // Collect (current body position, patch position) pairs.
  std::vector<size_t> offsets(body_model->triangle_batches.size() + 1);
  {
    size_t offset = 0;
    size_t i;
    for (i = 0; i < body_model->triangle_batches.size(); ++i) {
      offsets[i] = offset;
      offset += body_model->triangle_batches[i].vertices.size() *
                body_model->triangle_batches[i].vertex_size;
    }
    offsets[i] = offset;
  }
  std::vector<std::pair<Vector3<float>, Vector3<float>>> pairs;  // body, patch
  float sum_disp = 0.0f;
  for (const auto& pv : patch.vertices) {
    size_t off = (size_t)pv.original_offset;
    for (size_t j = 1; j <= body_model->triangle_batches.size(); ++j) {
      if (off < offsets[j]) {
        j--;
        const auto& batch = body_model->triangle_batches[j];
        size_t vi = (off - offsets[j]) / batch.vertex_size;
        if (vi < batch.vertices.size()) {
          const auto& bp = batch.vertices[vi].position;
          pairs.push_back({bp, pv.position});
          const float dx = bp.x - pv.position.x;
          const float dy = bp.y - pv.position.y;
          const float dz = bp.z - pv.position.z;
          sum_disp += std::sqrt(dx * dx + dy * dy + dz * dz);
        }
        break;
      }
    }
  }
  if (pairs.size() < 4) {
    return false;
  }
  const float mean_disp = sum_disp / pairs.size();
  // A real tuck stays close to the body; this much mean displacement means the
  // shape was authored against a different body revision.
  if (mean_disp <= 0.04f) {
    return false;
  }

  // Exact path: per-vertex correspondence via the legacy body (identical
  // topology, old positions). Tucks re-base to the current body's vertex plus
  // the authored tuck delta; garment vertices follow the displacement field of
  // their 3 nearest legacy-body vertices, so every region lands on the
  // matching current body region.
  std::vector<Vector3<float>> cur_pos, old_pos;
  // (vi below divides by batch[0].vertex_size; stock bodies are single
  // batch; multi-batch bodies take the affine fallback.)
  if (legacy_body_model && body_model->triangle_batches.size() == 1 &&
      FlattenPositions(body_model, nullptr, cur_pos) &&
      FlattenPositions(legacy_body_model, body_model, old_pos)) {
    // Confirm the item really is authored against the legacy body: the tuck
    // positions must sit near the legacy body's addressed vertices.
    float old_sum = 0.0f;
    size_t old_n = 0;
    for (const auto& pv : patch.vertices) {
      const size_t vi = (size_t)pv.original_offset / body_model->triangle_batches[0].vertex_size;
      if (vi >= old_pos.size()) continue;
      const float dx = old_pos[vi].x - pv.position.x;
      const float dy = old_pos[vi].y - pv.position.y;
      const float dz = old_pos[vi].z - pv.position.z;
      old_sum += std::sqrt(dx * dx + dy * dy + dz * dz);
      ++old_n;
    }
    const float old_mean = old_n ? old_sum / old_n : 1e9f;
    if (old_mean <= 0.03f) {
      for (auto& pv : patch.vertices) {
        const size_t vi =
            (size_t)pv.original_offset / body_model->triangle_batches[0].vertex_size;
        if (vi >= old_pos.size() || vi >= cur_pos.size()) continue;
        pv.position.x = cur_pos[vi].x + (pv.position.x - old_pos[vi].x);
        pv.position.y = cur_pos[vi].y + (pv.position.y - old_pos[vi].y);
        pv.position.z = cur_pos[vi].z + (pv.position.z - old_pos[vi].z);
      }
      if (item_model) {
        for (auto& batch : item_model->triangle_batches) {
          for (auto& v : batch.vertices) {
            // 3 nearest legacy-body vertices, inverse-distance weighted.
            size_t n0 = 0, n1 = 0, n2 = 0;
            float d0 = 1e9f, d1 = 1e9f, d2 = 1e9f;
            for (size_t i = 0; i < old_pos.size(); ++i) {
              const float dx = old_pos[i].x - v.position.x;
              const float dy = old_pos[i].y - v.position.y;
              const float dz = old_pos[i].z - v.position.z;
              const float d = dx * dx + dy * dy + dz * dz;
              if (d < d0) {
                d2 = d1; n2 = n1;
                d1 = d0; n1 = n0;
                d0 = d; n0 = i;
              } else if (d < d1) {
                d2 = d1; n2 = n1;
                d1 = d; n1 = i;
              } else if (d < d2) {
                d2 = d; n2 = i;
              }
            }
            const float w0 = 1.0f / (std::sqrt(d0) + 1e-4f);
            const float w1 = 1.0f / (std::sqrt(d1) + 1e-4f);
            const float w2 = 1.0f / (std::sqrt(d2) + 1e-4f);
            const float wsum = w0 + w1 + w2;
            v.position.x += (w0 * (cur_pos[n0].x - old_pos[n0].x) +
                             w1 * (cur_pos[n1].x - old_pos[n1].x) +
                             w2 * (cur_pos[n2].x - old_pos[n2].x)) / wsum;
            v.position.y += (w0 * (cur_pos[n0].y - old_pos[n0].y) +
                             w1 * (cur_pos[n1].y - old_pos[n1].y) +
                             w2 * (cur_pos[n2].y - old_pos[n2].y)) / wsum;
            v.position.z += (w0 * (cur_pos[n0].z - old_pos[n0].z) +
                             w1 * (cur_pos[n1].z - old_pos[n1].z) +
                             w2 * (cur_pos[n2].z - old_pos[n2].z)) / wsum;
          }
        }
      }
      return true;
    }
    REXKRNL_WARN("[avatar] old-body candidate does not match the legacy body either "
                 "(tucks {:.1f}cm from legacy); falling back to affine fit",
                 old_mean * 100.0f);
  }

  // Per-axis least-squares fit: patch = scale * body + offset. Axes with no
  // spread (e.g. a narrow neckline's X) fall back to a pure offset.
  float scale[3], offset[3];
  for (int axis = 0; axis < 3; ++axis) {
    auto get = [axis](const Vector3<float>& v) {
      return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
    };
    float mb = 0, mp = 0;
    for (const auto& pr : pairs) {
      mb += get(pr.first);
      mp += get(pr.second);
    }
    mb /= pairs.size();
    mp /= pairs.size();
    float var = 0, cov = 0;
    for (const auto& pr : pairs) {
      const float db = get(pr.first) - mb;
      var += db * db;
      cov += db * (get(pr.second) - mp);
    }
    float s = (var > 1e-6f) ? cov / var : 1.0f;
    if (s < 0.7f || s > 1.3f) {
      s = 1.0f;  // implausible fit; fall back to pure offset for this axis
    }
    scale[axis] = s;
    offset[axis] = mp - s * mb;
  }

  // Map old-body space -> current body space: p_new = (p_old - offset) / scale.
  auto unmap = [&](Vector3<float>& p) {
    p.x = (p.x - offset[0]) / scale[0];
    p.y = (p.y - offset[1]) / scale[1];
    p.z = (p.z - offset[2]) / scale[2];
  };
  for (auto& pv : patch.vertices) {
    unmap(pv.position);
  }
  if (item_model) {
    for (auto& batch : item_model->triangle_batches) {
      for (auto& v : batch.vertices) {
        unmap(v.position);
      }
    }
  }
  return true;
}

bool ApplyBlendShape(std::shared_ptr<BlendShape> blendShape,
                     const AssetId& modelAssetId,
                     std::shared_ptr<Model> model) {
  if (!BlendShapeTargetMatches(blendShape->index_patch.original_asset_id, modelAssetId)) {
    return false;
  }
  if (!BlendShapeTargetMatches(blendShape->vertex_patch.original_asset_id, modelAssetId)) {
    return false;
  }

  size_t totalIndexBufferSize = 0;
  size_t totalVertexBufferSize = 0;
  for (const auto& batch : model->triangle_batches) {
    totalIndexBufferSize += batch.triangle_count * 6;
    totalVertexBufferSize += batch.vertices.size() * batch.vertex_size;
  }
  totalVertexBufferSize = (totalVertexBufferSize + 127) & ~127;

  if (totalIndexBufferSize != blendShape->index_patch.total_buffer_size) {
    return false;
  }

  if (totalVertexBufferSize != blendShape->vertex_patch.total_buffer_size) {
    return false;
  }

  ApplyIndexPatch(blendShape->index_patch, model);
  ApplyVertexPatch(blendShape->vertex_patch, model);
  return true;
}

}  // namespace avatars
}  // namespace rex
