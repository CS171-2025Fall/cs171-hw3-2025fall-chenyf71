#include "rdr/integrator.h"

#include <omp.h>

#include "rdr/bsdf.h"
#include "rdr/camera.h"
#include "rdr/canary.h"
#include "rdr/film.h"
#include "rdr/halton.h"
#include "rdr/interaction.h"
#include "rdr/light.h"
#include "rdr/math_aliases.h"
#include "rdr/math_utils.h"
#include "rdr/platform.h"
#include "rdr/properties.h"
#include "rdr/ray.h"
#include "rdr/scene.h"
#include "rdr/sdtree.h"

RDR_NAMESPACE_BEGIN

/* ===================================================================== *
 *
 * Intersection Test Integrator's Implementation
 *
 * ===================================================================== */

void IntersectionTestIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // Statistics
  std::atomic<int> cnt = 0;

  const Vec2i &resolution = camera->getFilm()->getResolution();
#pragma omp parallel for schedule(dynamic)
  for (int dx = 0; dx < resolution.x; dx++) {
    ++cnt;
    if (cnt % (resolution.x / 10) == 0)
      Info_("Rendering: {:.02f}%", cnt * 100.0 / resolution.x);
    Sampler sampler;
    sampler.setSeed(dx);
    for (int dy = 0; dy < resolution.y; dy++) {
      sampler.setPixelIndex2D(Vec2i(dx, dy));
      for (int sample = 0; sample < spp; sample++) {
        // TODO(HW3): generate #spp rays for each pixel and use Monte Carlo
        // integration to compute radiance.
        //
        // Useful Functions:
        //
        // @see Sampler::getPixelSample for getting the current pixel sample
        // as Vec2f.
        //
        // @see Camera::generateDifferentialRay for generating rays given
        // pixel sample positions as 2 floats.

        // You should assign the following two variables
        // const Vec2f &pixel_sample = ...
        // auto ray = ...

        const Vec2f &pixel_sample = sampler.getPixelSample();
        auto ray =
            camera->generateDifferentialRay(pixel_sample.x, pixel_sample.y);
        // After you assign pixel_sample and ray, you can uncomment the
        // following lines to accumulate the radiance to the film.
        //
        //
        // Accumulate radiance
        assert(pixel_sample.x >= dx && pixel_sample.x <= dx + 1);
        assert(pixel_sample.y >= dy && pixel_sample.y <= dy + 1);
        const Vec3f &L = Li(scene, ray, sampler);
        camera->getFilm()->commitSample(pixel_sample, L);
      }
    }
  }
}

Vec3f IntersectionTestIntegrator::Li(ref<Scene> scene, DifferentialRay &ray,
                                     Sampler &sampler) const {
  Vec3f color(0.0);

  // Cast a ray until we hit a non-specular surface or miss
  // Record whether we have found a diffuse surface
  bool diffuse_found = false;
  SurfaceInteraction interaction;

  for (int i = 0; i < max_depth; ++i) {
    interaction = SurfaceInteraction();
    bool intersected = scene->intersect(ray, interaction);

    // Perform RTTI to determine the type of the surface
    bool is_ideal_diffuse =
        dynamic_cast<const IdealDiffusion *>(interaction.bsdf) != nullptr;
    bool is_perfect_refraction =
        dynamic_cast<const PerfectRefraction *>(interaction.bsdf) != nullptr;

    // Set the outgoing direction
    interaction.wo = -ray.direction;

    if (!intersected) {
      break;
    }

    if (is_perfect_refraction) {
      // We should follow the specular direction
      // TODO(HW3): call the interaction.bsdf->sample to get the new direction
      // and update the ray accordingly.
      //
      // Useful Functions:
      // @see BSDF::sample
      // @see SurfaceInteraction::spawnRay
      //
      // You should update ray = ... with the spawned ray
      Float pdf = 0.0f;
      interaction.bsdf->sample(interaction, sampler, &pdf);
      ray = interaction.spawnRay(interaction.wi);
      continue;
    }

    if (is_ideal_diffuse) {
      // We only consider diffuse surfaces for direct lighting
      diffuse_found = true;
      break;
    }

    // We simply omit any other types of surfaces
    break;
  }

  if (!diffuse_found) {
    return color;
  }

  color = directLighting(scene, interaction, sampler);
  return color;
}

Vec3f IntersectionTestIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction, Sampler &sampler) const {
  Vec3f color(0, 0, 0);

  for (const auto &light : lights) {
    Vec3f L_incoming;
    Vec3f light_dir;
    Float dist_to_light;

    if (light.is_area_light) {
      Vec2f uv = sampler.get2D();
      Vec3f sample_pos = light.position +
                         light.u * (uv.x - 0.5f) * light.size.x +
                         light.v * (uv.y - 0.5f) * light.size.y;

      Vec3f diff = sample_pos - interaction.p;
      Float dist_sq = Dot(diff, diff);
      dist_to_light = std::sqrt(dist_sq);
      light_dir = diff / dist_to_light;
      Float cos_light = Dot(light.normal, -light_dir);

      if (cos_light <= 0.0f)
        continue;

      L_incoming = light.emission * light.area * (cos_light / dist_sq);
    } else {
      dist_to_light = Norm(light.position - interaction.p);
      light_dir = Normalize(light.position - interaction.p);
      Float attenuation = 1.0f / (4.0f * PI * dist_to_light * dist_to_light);
      L_incoming = light.emission * attenuation;
    }

    Ray test_ray(interaction.p, light_dir, 1e-4f, dist_to_light - 1e-4f);
    SurfaceInteraction shadow_isect;
    if (scene->intersect(test_ray, shadow_isect)) {
      continue;
    }
    const BSDF *bsdf = interaction.bsdf;
    bool is_ideal_diffuse =
        dynamic_cast<const IdealDiffusion *>(bsdf) != nullptr;

    if (bsdf != nullptr && is_ideal_diffuse) {
      Float cos_theta = std::max(Dot(light_dir, interaction.normal), 0.0f);

      color += bsdf->evaluate(interaction) * L_incoming * cos_theta;
    }
  }

  return color * 2;
}

/* ===================================================================== *
 *
 * Path Integrator's Implementation
 *
 * ===================================================================== */

void PathIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f PathIntegrator::Li(ref<Scene> scene, DifferentialRay &ray,
                         Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f PathIntegrator::directLighting(ref<Scene> scene,
                                     SurfaceInteraction &interaction,
                                     Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

/* ===================================================================== *
 *
 * New Integrator's Implementation
 *
 * ===================================================================== */

// Instantiate template
// clang-format off
template Vec3f
IncrementalPathIntegrator::Li<Path>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
template Vec3f
IncrementalPathIntegrator::Li<PathImmediate>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
// clang-format on

// This is exactly a way to separate dec and def
template <typename PathType>
Vec3f IncrementalPathIntegrator::Li( // NOLINT
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

RDR_NAMESPACE_END
