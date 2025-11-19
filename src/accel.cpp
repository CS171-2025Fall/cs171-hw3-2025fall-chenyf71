#include "rdr/accel.h"

#include "rdr/canary.h"
#include "rdr/interaction.h"
#include "rdr/math_aliases.h"
#include "rdr/platform.h"
#include "rdr/shape.h"

RDR_NAMESPACE_BEGIN

/* ===================================================================== *
 *
 * AABB Implementations
 *
 * ===================================================================== */

bool AABB::isOverlap(const AABB &other) const {
  return ((other.low_bnd[0] >= this->low_bnd[0] &&
           other.low_bnd[0] <= this->upper_bnd[0]) ||
          (this->low_bnd[0] >= other.low_bnd[0] &&
           this->low_bnd[0] <= other.upper_bnd[0])) &&
         ((other.low_bnd[1] >= this->low_bnd[1] &&
           other.low_bnd[1] <= this->upper_bnd[1]) ||
          (this->low_bnd[1] >= other.low_bnd[1] &&
           this->low_bnd[1] <= other.upper_bnd[1])) &&
         ((other.low_bnd[2] >= this->low_bnd[2] &&
           other.low_bnd[2] <= this->upper_bnd[2]) ||
          (this->low_bnd[2] >= other.low_bnd[2] &&
           this->low_bnd[2] <= other.upper_bnd[2]));
}

bool AABB::intersect(const Ray &ray, Float *t_in, Float *t_out) const {
  // TODO(HW3): implement ray intersection with AABB.
  // ray distance for two intersection points are returned by pointers.
  //
  // This method should modify t_in and t_out as the "time"
  // when the ray enters and exits the AABB respectively.
  //
  // And return true if there is an intersection, false otherwise.
  //
  // Useful Functions:
  // @see Ray::safe_inverse_direction
  //    for getting the inverse direction of the ray.
  // @see Min/Max/ReduceMin/ReduceMax
  //    for vector min/max operations.
  const Vec3f &invdir = ray.safe_inverse_direction;

  Float tx1 = (low_bnd.x - ray.origin.x) * invdir.x;
  Float tx2 = (upper_bnd.x - ray.origin.x) * invdir.x;
  Float tmin = std::min(tx1, tx2);
  Float tmax = std::max(tx1, tx2);

  Float ty1 = (low_bnd.y - ray.origin.y) * invdir.y;
  Float ty2 = (upper_bnd.y - ray.origin.y) * invdir.y;
  tmin = std::max(tmin, std::min(ty1, ty2));
  tmax = std::min(tmax, std::max(ty1, ty2));

  Float tz1 = (low_bnd.z - ray.origin.z) * invdir.z;
  Float tz2 = (upper_bnd.z - ray.origin.z) * invdir.z;
  tmin = std::max(tmin, std::min(tz1, tz2));
  tmax = std::min(tmax, std::max(tz1, tz2));

  if (tmax < tmin || tmax < ray.t_min || tmin > ray.t_max)
    return false;

  if (t_in)
    *t_in = tmin;
  if (t_out)
    *t_out = tmax;
  return true;
}

/* ===================================================================== *
 *
 * Accelerator Implementations
 *
 * ===================================================================== */

bool TriangleIntersect(Ray &ray, const uint32_t &triangle_index,
                       const ref<TriangleMeshResource> &mesh,
                       SurfaceInteraction &interaction) {
  using InternalScalarType = Double;
  using InternalVecType = Vec<InternalScalarType, 3>;

  AssertAllValid(ray.direction, ray.origin);
  AssertAllNormalized(ray.direction);

  const auto &vertices = mesh->vertices;
  const Vec3u v_idx(&mesh->v_indices[3 * triangle_index]);
  assert(v_idx.x < mesh->vertices.size());
  assert(v_idx.y < mesh->vertices.size());
  assert(v_idx.z < mesh->vertices.size());

  InternalVecType dir = Cast<InternalScalarType>(ray.direction);
  InternalVecType v0 = Cast<InternalScalarType>(vertices[v_idx[0]]);
  InternalVecType v1 = Cast<InternalScalarType>(vertices[v_idx[1]]);
  InternalVecType v2 = Cast<InternalScalarType>(vertices[v_idx[2]]);

  // TODO(HW3): implement ray-triangle intersection test.
  // You should compute the u, v, t as InternalScalarType
  //
  //   InternalScalarType u = ...;
  //   InternalScalarType v = ...;
  //   InternalScalarType t = ...;
  //
  // And exit early with `return false` if there is no intersection.
  //
  // The intersection points is denoted as:
  // (1 - u - v) * v0 + u * v1 + v * v2 == ray.origin + t * ray.direction
  // where the left side is the barycentric interpolation of the triangle
  // vertices, and the right side is the parametric equation of the ray.
  //
  // You should also make sure that:
  // u >= 0, v >= 0, u + v <= 1, and, ray.t_min <= t <= ray.t_max
  //
  // Useful Functions:
  // You can use @see Cross and @see Dot for determinant calculations.

  // Delete the following lines after you implement the function
  InternalScalarType u = InternalScalarType(0);
  InternalScalarType v = InternalScalarType(0);
  InternalScalarType t = InternalScalarType(0);
  auto e1 = v1 - v0;
  auto e2 = v2 - v0;
  auto s = Cast<InternalScalarType>(ray.origin) - v0;
  auto s1 = Cross(dir, e2);
  auto s2 = Cross(s, e1);
  auto denom = Dot(s1, e1);
  u = Dot(s1, s) / denom;
  v = Dot(s2, dir) / denom;
  t = Dot(s2, e2) / denom;
  if (std::abs(denom) < InternalScalarType(1e-8) || u < InternalScalarType(0) ||
      v < InternalScalarType(0) || u + v > InternalScalarType(1) ||
      t < InternalScalarType(ray.t_min) || t > InternalScalarType(ray.t_max)) {
    return false;
  }

  // We will reach here if there is an intersection

  CalculateTriangleDifferentials(interaction,
                                 {static_cast<Float>(1 - u - v),
                                  static_cast<Float>(u), static_cast<Float>(v)},
                                 mesh, triangle_index);
  AssertNear(interaction.p, ray(t));
  assert(ray.withinTimeRange(t));
  ray.setTimeMax(t);
  return true;
}

void Accel::setTriangleMesh(const ref<TriangleMeshResource> &mesh) {
  // Build the bounding box
  AABB bound(Vec3f(Float_INF, Float_INF, Float_INF),
             Vec3f(Float_MINUS_INF, Float_MINUS_INF, Float_MINUS_INF));
  for (auto &vertex : mesh->vertices) {
    bound.low_bnd = Min(bound.low_bnd, vertex);
    bound.upper_bnd = Max(bound.upper_bnd, vertex);
  }

  this->mesh = mesh;   // set the pointer
  this->bound = bound; // set the bounding box
}

void Accel::build() {}

AABB Accel::getBound() const { return bound; }

bool Accel::intersect(Ray &ray, SurfaceInteraction &interaction) const {
  bool success = false;
  for (int i = 0; i < mesh->v_indices.size() / 3; i++)
    success |= TriangleIntersect(ray, i, mesh, interaction);
  return success;
}

RDR_NAMESPACE_END
