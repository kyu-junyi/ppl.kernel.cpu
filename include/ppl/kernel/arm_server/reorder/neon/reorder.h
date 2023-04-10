// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef __ST_PPL_KERNEL_ARM_SERVER_REORDER_NEON_REORDER_H_
#define __ST_PPL_KERNEL_ARM_SERVER_REORDER_NEON_REORDER_H_

#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

ppl::common::RetCode trans_f32c4_to_f32nda  (const float* src,  const int64_t shape[4], float* dst);
ppl::common::RetCode trans_f32nda_to_f32c4  (const float* src,  const int64_t shape[4], float* dst);

#ifdef PPLNN_USE_ARMV8_2_FP16
ppl::common::RetCode trans_f16c8_to_f16nda  (const __fp16* src, const int64_t shape[4], __fp16* dst);
ppl::common::RetCode trans_f16nda_to_f16c8  (const __fp16* src, const int64_t shape[4], __fp16* dst);

ppl::common::RetCode cast_f32_to_f16        (const float* src,  const int64_t shape[4], __fp16* dst);
ppl::common::RetCode cast_f16_to_f32        (const __fp16* src, const int64_t shape[4], float* dst);

ppl::common::RetCode reorder_f32nda_to_f16c8(const float* src,  const int64_t shape[4], __fp16* dst);
ppl::common::RetCode reorder_f16c8_to_f32nda(const __fp16* src, const int64_t shape[4], float* dst);

ppl::common::RetCode reorder_f32c4_to_f16c8 (const float* src,  const int64_t shape[4], __fp16* dst);
ppl::common::RetCode reorder_f16c8_to_f32c4 (const __fp16* src, const int64_t shape[4], float* dst);
#endif

}}}}; // namespace ppl::kernel::arm_server::neon

#endif
