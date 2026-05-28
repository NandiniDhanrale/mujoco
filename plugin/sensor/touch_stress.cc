// Copyright 2024 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "touch_stress.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjplugin.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mujoco.h>

namespace mujoco::plugin::sensor {

namespace {

// Checks that a plugin config attribute exists.
bool CheckAttr(const std::string& input) {
  char* end;
  std::string value = input;
  value.erase(std::remove_if(value.begin(), value.end(), isspace), value.end());
  strtod(value.c_str(), &end);
  return end == value.data() + value.size();
}

// Converts a string into a numeric vector
template <typename T>
void ReadVector(std::vector<T>& output, const std::string& input) {
  std::stringstream ss(input);
  std::string item;
  char delim = ' ';
  while (getline(ss, item, delim)) {
    CheckAttr(item);
    output.push_back(strtod(item.c_str(), nullptr));
  }
}

// Evenly spaced numbers over a specified interval.
void LinSpace(mjtNum lower, mjtNum upper, int n, mjtNum array[]) {
  mjtNum increment = n > 1 ? (upper - lower) / (n - 1) : 0;
  for (int i = 0; i < n; ++i) {
    *array = lower;
    ++array;
    lower += increment;
  }
}

// Parametrized linear/quintic interpolated nonlinearity.
mjtNum Fovea(mjtNum x, mjtNum gamma) {
  if (!gamma) return x;
  mjtNum g = mjMAX(0, mjMIN(1, gamma));
  return g*mju_pow(x, 5) + (1 - g)*x;
}

// Make bin edges.
void BinEdges(mjtNum* x_edges, mjtNum* y_edges, int size[2], mjtNum fov[2],
              mjtNum gamma) {
  LinSpace(-1, 1, size[0] + 1, x_edges);
  LinSpace(-1, 1, size[1] + 1, y_edges);
  for (int i = 0; i < size[0] + 1; i++) {
    x_edges[i] = Fovea(x_edges[i], gamma);
  }
  for (int i = 0; i < size[1] + 1; i++) {
    y_edges[i] = Fovea(y_edges[i], gamma);
  }
  mju_scl(x_edges, x_edges, fov[0]*mjPI / 180, size[0] + 1);
  mju_scl(y_edges, y_edges, fov[1]*mjPI / 180, size[1] + 1);
}

// Transform spherical (azimuth, elevation, radius) to Cartesian (x,y,z)
// in the sensor frame (points down -z).
void SphericalToCartesian(const mjtNum aer[3], mjtNum xyz[3]) {
  mjtNum a = aer[0], e = aer[1], r = aer[2];
  xyz[0] = r * mju_cos(e) * mju_sin(a);
  xyz[1] = r * mju_sin(e);
  xyz[2] = -r * mju_cos(e) * mju_cos(a);
}

// Compute the orthonormal taxel frame for a given (azimuth, elevation).
// The z-axis points outward from the sensor center (radial direction).
// The x-axis points in the azimuth direction (horizontal tangent).
// The y-axis points in the elevation direction (vertical tangent).
void TaxelFrame(mjtNum a, mjtNum e,
                mjtNum z[3], mjtNum x[3], mjtNum y[3]) {
  z[0] = mju_cos(e) * mju_sin(a);
  z[1] = mju_sin(e);
  z[2] = -mju_cos(e) * mju_cos(a);

  x[0] = mju_cos(a);
  x[1] = 0;
  x[2] = mju_sin(a);

  mju_cross(y, z, x);
}

}  // namespace

// Creates a TouchStress instance if all config attributes are defined and
// within their allowed bounds.
TouchStress* TouchStress::Create(const mjModel* m, mjData* d, int instance) {
  if (CheckAttr(std::string(mj_getPluginConfig(m, instance, "gamma"))) &&
      CheckAttr(std::string(mj_getPluginConfig(m, instance, "nchannel")))) {
    // nchannel
    int nchannel = strtod(mj_getPluginConfig(m, instance, "nchannel"), nullptr);
    if (!nchannel) nchannel = 3;
    if (nchannel < 1 || nchannel > 6) {
      mju_error("nchannel must be between 1 and 6");
      return nullptr;
    }

    // size
    std::vector<int> size;
    std::string size_str = std::string(mj_getPluginConfig(m, instance, "size"));
    ReadVector(size, size_str.c_str());
    if (size.size() != 2) {
      mju_error("Both horizontal and vertical resolutions must be specified");
      return nullptr;
    }
    if (size[0] <= 0 || size[1] <= 0) {
      mju_error("Horizontal and vertical resolutions must be positive");
      return nullptr;
    }

    // field of view
    std::vector<mjtNum> fov;
    std::string fov_str = std::string(mj_getPluginConfig(m, instance, "fov"));
    ReadVector(fov, fov_str.c_str());
    if (fov.size() != 2) {
      mju_error(
          "Both horizontal and vertical fields of view must be specified");
      return nullptr;
    }
    if (fov[0] <= 0 || fov[0] > 180) {
      mju_error("`fov[0]` must be a float between (0, 180] degrees");
      return nullptr;
    }
    if (fov[1] <= 0 || fov[1] > 90) {
      mju_error("`fov[1]` must be a float between (0, 90] degrees");
      return nullptr;
    }

    // gamma
    mjtNum gamma = strtod(mj_getPluginConfig(m, instance, "gamma"), nullptr);
    if (gamma < 0 || gamma > 1) {
      mju_error("`gamma` must be a nonnegative float between [0, 1]");
      return nullptr;
    }

    return new TouchStress(m, d, instance, nchannel, size.data(), fov.data(),
                           gamma);
  } else {
    mju_error("Invalid or missing parameters in touch_stress sensor plugin");
    return nullptr;
  }
}

TouchStress::TouchStress(const mjModel* m, mjData* d, int instance,
                         int nchannel, int size[2], mjtNum fov[2],
                         mjtNum gamma)
    : nchannel_(nchannel),
      size_{size[0], size[1]},
      fov_{fov[0], fov[1]},
      gamma_(gamma) {
  // Make sure sensor is attached to a site.
  for (int i = 0; i < m->nsensor; ++i) {
    if (m->sensor_type[i] == mjSENS_PLUGIN &&
        m->sensor_plugin[i] == instance) {
      if (m->sensor_objtype[i] != mjOBJ_SITE) {
        mju_error("Touch Stress sensor must be attached to a site");
      }
    }
  }
}

void TouchStress::Reset(const mjModel* m, int instance) {}

void TouchStress::Compute(const mjModel* m, mjData* d, int instance) {
  mj_markStack(d);

  // Get sensor id.
  int id;
  for (id = 0; id < m->nsensor; ++id) {
    if (m->sensor_type[id] == mjSENS_PLUGIN &&
        m->sensor_plugin[id] == instance) {
      break;
    }
  }

  // Clear sensordata.
  mjtNum* sensordata = d->sensordata + m->sensor_adr[id];
  int frame = size_[0] * size_[1];
  mju_zero(sensordata, m->sensor_dim[id]);

  // Get site id and parent weld body.
  int site_id = m->sensor_objid[id];
  int parent_body = m->body_weldid[m->site_bodyid[site_id]];
  int parent_weld = m->body_weldid[parent_body];

  // Collect unique contacting geom ids.
  int* contact_geom_ids = mj_stackAllocInt(d, d->ncon);
  int ncontact = 0;
  for (int k = 0; k < d->ncon; k++) {
    int body1 = m->body_weldid[m->geom_bodyid[d->contact[k].geom1]];
    int body2 = m->body_weldid[m->geom_bodyid[d->contact[k].geom2]];
    if (body1 == parent_weld) {
      int add = 1;
      for (int j = 0; j < ncontact; j++) {
        if (contact_geom_ids[j] == d->contact[k].geom2) {
          add = 0;
          break;
        }
      }
      if (add) {
        contact_geom_ids[ncontact] = d->contact[k].geom2;
        ncontact++;
      }
    }
    if (body2 == parent_weld) {
      int add = 1;
      for (int j = 0; j < ncontact; j++) {
        if (contact_geom_ids[j] == d->contact[k].geom1) {
          add = 0;
          break;
        }
      }
      if (add) {
        contact_geom_ids[ncontact] = d->contact[k].geom1;
        ncontact++;
      }
    }
  }

  if (!ncontact) {
    mj_freeStack(d);
    return;
  }

  // Get site frame.
  mjtNum* site_pos = d->site_xpos + 3*site_id;
  mjtNum* site_mat = d->site_xmat + 9*site_id;

  // Compute bin edges.
  mjtNum* x_edges = mj_stackAllocNum(d, size_[0] + 1);
  mjtNum* y_edges = mj_stackAllocNum(d, size_[1] + 1);
  BinEdges(x_edges, y_edges, size_, fov_, gamma_);

  // Allocate per-taxel output buffer: 3 channels (normal, tang1, tang2).
  mjtNum* forcesT = mj_stackAllocNum(d, 3 * frame);
  mju_zero(forcesT, 3 * frame);

  // Reference range for SDF queries.
  const mjtNum kReferenceRange = 0.1;

  // Iterate over contacting geoms.
  for (int g = 0; g < ncontact; g++) {
    int geom = contact_geom_ids[g];
    int body = m->geom_bodyid[geom];

    // Get SDF plugin info for this geom.
    int sdf_instance[2] = {-1, -1};
    mjtGeom geomtype[2] = {mjGEOM_SDF, mjGEOM_SPHERE};
    const mjpPlugin* sdf_ptr[2] = {NULL, NULL};

    if (m->geom_type[geom] == mjGEOM_SDF) {
      sdf_instance[0] = m->geom_plugin[geom];
      sdf_ptr[0] = mjc_getSDF(m, geom);
    } else if (m->geom_type[geom] == mjGEOM_MESH) {
      sdf_instance[0] = m->geom_dataid[geom];
      geomtype[0] = (mjtGeom)m->geom_type[geom];
    } else {
      sdf_instance[0] = geom;
      geomtype[0] = (mjtGeom)m->geom_type[geom];
    }

    // Skip mesh geoms not having an octree.
    if (geomtype[0] == mjGEOM_MESH &&
        m->mesh_octadr[m->geom_dataid[geom]] == -1) {
      continue;
    }

    // Build SDF descriptor.
    mjSDF geom_sdf;
    geom_sdf.id = &sdf_instance[0];
    geom_sdf.type = mjSDFTYPE_SINGLE;
    geom_sdf.plugin = &sdf_ptr[0];
    geom_sdf.geomtype = &geomtype[0];

    // For each taxel.
    int node = 0;
    for (int i = 0; i < size_[0]; i++) {
      for (int j = 0; j < size_[1]; j++) {
        // Taxel center in spherical coordinates.
        mjtNum a = 0.5 * (x_edges[i] + x_edges[i + 1]);
        mjtNum e = 0.5 * (y_edges[j] + y_edges[j + 1]);

        // Position in sensor frame at reference range.
        mjtNum aer[3] = {a, e, kReferenceRange};
        mjtNum pos_sensor[3];
        SphericalToCartesian(aer, pos_sensor);

        // Transform to global frame.
        mjtNum xpos[3];
        mju_mulMatVec3(xpos, site_mat, pos_sensor);
        mju_addTo3(xpos, site_pos);

        // Transform to geom local frame.
        mjtNum lpos[3], tmp[3];
        mju_sub3(tmp, xpos, d->geom_xpos + 3*geom);
        mju_mulMatTVec3(lpos, d->geom_xmat + 9*geom, tmp);

        // For SDF plugins, account for mesh offset.
        if (sdf_ptr[0] != NULL) {
          mjtNum mesh_mat[9];
          mju_quat2Mat(mesh_mat, m->mesh_quat + 4*m->geom_dataid[geom]);
          mju_mulMatVec3(lpos, mesh_mat, lpos);
          mju_addTo3(lpos, m->mesh_pos + 3*m->geom_dataid[geom]);
        }

        // Query signed distance.
        mjtNum dist = mju_min(mjc_distance(m, d, &geom_sdf, lpos), 0);
        if (dist == 0) {
          node++;
          continue;
        }

        // Relative velocity at contact point.
        mjtNum vel_sensor[6], vel_other[6], vel_rel[3];
        mju_transformSpatial(
            vel_sensor, d->cvel + 6*parent_weld, 0, xpos,
            d->subtree_com + 3*m->body_rootid[parent_weld], NULL);
        mju_transformSpatial(
            vel_other, d->cvel + 6*body, 0, d->geom_xpos + 3*geom,
            d->subtree_com + 3*m->body_rootid[body], NULL);
        mju_sub3(vel_rel, vel_sensor+3, vel_other+3);

        // Taxel local frame (in sensor coordinates).
        mjtNum taxel_n[3], taxel_x[3], taxel_y[3];
        TaxelFrame(a, e, taxel_n, taxel_x, taxel_y);

        // Transform taxel frame to world frame.
        mjtNum world_x[3], world_y[3];
        mju_mulMatVec3(world_x, site_mat, taxel_x);
        mju_mulMatVec3(world_y, site_mat, taxel_y);

        // Normal stress from SDF penetration depth.
        mjtNum depth = -dist;
        mjtNum kMaxDepth = 0.05;
        mjtNum pressure = depth / mju_max(kMaxDepth - depth, mjMINVAL);

        // Tangential stress from sliding velocity.
        mjtNum vt1 = mju_abs(mju_dot3(vel_rel, world_x));
        mjtNum vt2 = mju_abs(mju_dot3(vel_rel, world_y));

        // Accumulate across all contacting geoms.
        forcesT[0*frame + node] += pressure;
        forcesT[1*frame + node] += vt1;
        forcesT[2*frame + node] += vt2;
        node++;
      }
    }
  }

  // Write up to nchannel_ channels to sensordata.
  for (int c = 0; c < mjMIN(nchannel_, 3); c++) {
    mju_addTo(sensordata + c*frame, forcesT + c*frame, frame);
  }

  mj_freeStack(d);
}

void TouchStress::Visualize(const mjModel* m, mjData* d, const mjvOption* opt,
                            mjvScene* scn, int instance) {
  // Visualization not yet implemented.
}

void TouchStress::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name = "mujoco.sensor.touch_stress";
  plugin.capabilityflags |= mjPLUGIN_SENSOR;

  const char* attributes[] = {"nchannel", "size", "fov", "gamma"};
  plugin.nattribute = sizeof(attributes) / sizeof(attributes[0]);
  plugin.attributes = attributes;

  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

  plugin.nsensordata = +[](const mjModel* m, int instance, int sensor_id) {
    int nchannel = strtod(mj_getPluginConfig(m, instance, "nchannel"), nullptr);
    if (!nchannel) nchannel = 3;
    std::vector<int> size;
    std::string size_str = std::string(mj_getPluginConfig(m, instance, "size"));
    ReadVector(size, size_str.c_str());
    return nchannel * size[0] * size[1];
  };

  plugin.needstage = mjSTAGE_ACC;

  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    auto* TouchStress = TouchStress::Create(m, d, instance);
    if (!TouchStress) {
      return -1;
    }
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(TouchStress);
    return 0;
  };

  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<TouchStress*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };

  plugin.reset = +[](const mjModel* m, mjtNum* plugin_state, void* plugin_data,
                     int instance) {
    auto* TouchStress = reinterpret_cast<class TouchStress*>(plugin_data);
    TouchStress->Reset(m, instance);
  };

  plugin.compute =
      +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
        auto* TouchStress =
            reinterpret_cast<class TouchStress*>(d->plugin_data[instance]);
        TouchStress->Compute(m, d, instance);
      };

  plugin.visualize = +[](const mjModel* m, mjData* d, const mjvOption* opt,
                         mjvScene* scn, int instance) {
    auto* TouchStress =
        reinterpret_cast<class TouchStress*>(d->plugin_data[instance]);
    TouchStress->Visualize(m, d, opt, scn, instance);
  };

  mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::sensor
