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

#ifndef MUJOCO_PLUGIN_SENSOR_TOUCH_STRESS_H_
#define MUJOCO_PLUGIN_SENSOR_TOUCH_STRESS_H_

#include <vector>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include <mujoco/mjvisualize.h>

namespace mujoco::plugin::sensor {

class TouchStress {
 public:
  static TouchStress* Create(const mjModel* m, mjData* d, int instance);
  TouchStress(TouchStress&&) = default;
  ~TouchStress() = default;

  void Reset(const mjModel* m, int instance);
  void Compute(const mjModel* m, mjData* d, int instance);
  void Visualize(const mjModel* m, mjData* d, const mjvOption* opt,
                 mjvScene* scn, int instance);

  static void RegisterPlugin();

  int nchannel_;
  int size_[2];
  mjtNum fov_[2];
  mjtNum gamma_;

 private:
  TouchStress(const mjModel* m, mjData* d, int instance, int nchannel, int* size,
              mjtNum* fov, mjtNum gamma);
};

}  // namespace mujoco::plugin::sensor

#endif  // MUJOCO_PLUGIN_SENSOR_TOUCH_STRESS_H_
