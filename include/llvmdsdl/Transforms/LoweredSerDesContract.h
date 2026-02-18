#ifndef LLVMDSDL_TRANSFORMS_LOWERED_SERDES_CONTRACT_H
#define LLVMDSDL_TRANSFORMS_LOWERED_SERDES_CONTRACT_H

#include <cstdint>

namespace llvmdsdl {

inline constexpr std::int64_t kLoweredSerDesContractVersion = 1;
inline constexpr char kLoweredSerDesContractVersionAttr[] =
    "llvmdsdl.lowered_contract_version";
inline constexpr char kLoweredSerDesContractProducerAttr[] =
    "llvmdsdl.lowered_contract_producer";
inline constexpr char kLoweredSerDesContractProducer[] =
    "lower-dsdl-serialization";
inline constexpr char kLoweredPlanMarkerAttr[] = "lowered";
inline constexpr char kLoweredMinBitsAttr[] = "lowered_min_bits";
inline constexpr char kLoweredMaxBitsAttr[] = "lowered_max_bits";
inline constexpr char kLoweredStepCountAttr[] = "lowered_step_count";
inline constexpr char kLoweredFieldCountAttr[] = "lowered_field_count";
inline constexpr char kLoweredPaddingCountAttr[] = "lowered_padding_count";
inline constexpr char kLoweredAlignCountAttr[] = "lowered_align_count";
inline constexpr char kLoweredCapacityCheckHelperAttr[] =
    "lowered_capacity_check_helper";
inline constexpr char kLoweredUnionTagValidateHelperAttr[] =
    "lowered_union_tag_validate_helper";
inline constexpr char kLoweredSerUnionTagHelperAttr[] =
    "lowered_ser_union_tag_helper";
inline constexpr char kLoweredDeserUnionTagHelperAttr[] =
    "lowered_deser_union_tag_helper";

} // namespace llvmdsdl

#endif // LLVMDSDL_TRANSFORMS_LOWERED_SERDES_CONTRACT_H
