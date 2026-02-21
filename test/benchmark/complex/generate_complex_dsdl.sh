#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
NS_ROOT="${ROOT_DIR}/civildrone"
CLASSIC_WORKLOAD_COUNT="${CLASSIC_WORKLOAD_COUNT:-400}"
VISION_WORKLOAD_COUNT="${VISION_WORKLOAD_COUNT:-500}"
MEGABUNDLE_COUNT="${MEGABUNDLE_COUNT:-180}"

rm -rf "${NS_ROOT}"
mkdir -p "${NS_ROOT}/core" "${NS_ROOT}/fleet" "${NS_ROOT}/workload" "${NS_ROOT}/vision/common"

cat > "${NS_ROOT}/core/NodeIdentity.1.0.dsdl" <<'EOT'
uavcan.node.Heartbeat.1.0 heartbeat
uavcan.time.SynchronizedTimestamp.1.0 timestamp
truncated uint8[<=64] vehicle_name
uint16 node_id
uint8 fleet_id
@sealed
EOT

cat > "${NS_ROOT}/core/GeoPoint.1.0.dsdl" <<'EOT'
float64 latitude_rad
float64 longitude_rad
float32 altitude_m
@sealed
EOT

cat > "${NS_ROOT}/core/GeoBounds.1.0.dsdl" <<'EOT'
civildrone.core.GeoPoint.1.0 southwest
civildrone.core.GeoPoint.1.0 northeast
@sealed
EOT

cat > "${NS_ROOT}/core/PoseKinematics.1.0.dsdl" <<'EOT'
uavcan.si.sample.length.Vector3.1.0 position
uavcan.si.sample.velocity.Vector3.1.0 velocity
uavcan.si.sample.acceleration.Vector3.1.0 acceleration
uavcan.si.sample.angle.Quaternion.1.0 attitude
uavcan.si.sample.angular_velocity.Vector3.1.0 angular_velocity
@sealed
EOT

cat > "${NS_ROOT}/core/WindEstimate.1.0.dsdl" <<'EOT'
uavcan.si.sample.velocity.Vector3.1.0 velocity
uavcan.si.sample.pressure.Scalar.1.0 pressure
uavcan.si.sample.temperature.Scalar.1.0 temperature
@sealed
EOT

cat > "${NS_ROOT}/core/BatterySnapshot.1.0.dsdl" <<'EOT'
uavcan.si.sample.voltage.Scalar.1.0 voltage
uavcan.si.sample.electric_current.Scalar.1.0 current
uavcan.si.sample.power.Scalar.1.0 power
float32 state_of_charge
float32 state_of_health
@sealed
EOT

cat > "${NS_ROOT}/core/LinkStats.1.0.dsdl" <<'EOT'
uint32 tx_frames
uint32 rx_frames
uint32 dropped_frames
float32 packet_error_rate
uavcan.diagnostic.Record.1.1 last_fault
@sealed
EOT

cat > "${NS_ROOT}/core/MissionPhase.1.0.dsdl" <<'EOT'
@union
uavcan.primitive.Empty.1.0 idle
uavcan.primitive.String.1.0 phase_name
uint8 phase_code
civildrone.core.GeoPoint.1.0 anchor
@sealed
EOT

cat > "${NS_ROOT}/core/FaultRecord.1.0.dsdl" <<'EOT'
uavcan.time.SynchronizedTimestamp.1.0 timestamp
uavcan.diagnostic.Record.1.1 diagnostic
uavcan.register.Value.1.0 captured_register
@sealed
EOT

cat > "${NS_ROOT}/core/SafetyEnvelope.1.0.dsdl" <<'EOT'
float32 max_speed_mps
float32 max_altitude_m
float32 max_climb_rate_mps
float32 max_descend_rate_mps
civildrone.core.GeoBounds.1.0 geofence
@sealed
EOT

cat > "${NS_ROOT}/core/SurveyTask.1.0.dsdl" <<'EOT'
uint32 task_id
civildrone.core.GeoBounds.1.0 area
float32 desired_ground_resolution_m
uavcan.primitive.String.1.0 sensor_profile
@sealed
EOT

cat > "${NS_ROOT}/core/SurveyPlan.1.0.dsdl" <<'EOT'
uint32 plan_id
civildrone.core.SurveyTask.1.0[<=32] tasks
civildrone.core.SafetyEnvelope.1.0 constraints
uavcan.time.SynchronizedTimestamp.1.0 created_at
@sealed
EOT

cat > "${NS_ROOT}/core/ActuatorCommand.1.0.dsdl" <<'EOT'
uint16 actuator_id
float32 command
float32 feed_forward
float32 rate_limit
@sealed
EOT

cat > "${NS_ROOT}/core/SensorHealth.1.0.dsdl" <<'EOT'
uint16 sensor_id
bool online
float32 confidence
uavcan.diagnostic.Record.1.1 last_diagnostic
@sealed
EOT

cat > "${NS_ROOT}/core/SystemSnapshot.1.0.dsdl" <<'EOT'
civildrone.core.NodeIdentity.1.0 identity
civildrone.core.PoseKinematics.1.0 pose
civildrone.core.WindEstimate.1.0 wind
civildrone.core.BatterySnapshot.1.0 battery
civildrone.core.LinkStats.1.0 link
civildrone.core.MissionPhase.1.0 mission_phase
civildrone.core.FaultRecord.1.0[<=16] recent_faults
@sealed
EOT

cat > "${NS_ROOT}/core/Dispatch.1.0.dsdl" <<'EOT'
civildrone.core.NodeIdentity.1.0 requester
civildrone.core.SurveyPlan.1.0 plan
civildrone.core.SafetyEnvelope.1.0 envelope
@sealed
---
bool accepted
uavcan.register.Value.1.0 scheduling_register
uavcan.primitive.String.1.0 message
@sealed
EOT

domains=(
  airframe mission navigation estimation control propulsion power payload
  survey mapping imaging lidar radar communications avoidance safety
  diagnostics weather geofence terrain autonomy perception planner traffic
  energy maintenance networking storage
)

vision_modules=(
  camera gimbal optics isp stream transport controlchannel codec encoder decoder
  packetizer depacketizer recorder synchronization calibration stereo depth
  vslam localization mapping feature extraction matching triangulation detection
  classification segmentation tracking reid inference fusion planner scenegraph
  objectstore telemetryvision
)

# Upper-case first character of a token.
upper_first() {
  local s="$1"
  if [ -z "${s}" ]; then
    return
  fi
  local first rest
  first="$(printf '%s' "${s}" | cut -c1 | tr '[:lower:]' '[:upper:]')"
  rest="$(printf '%s' "${s}" | cut -c2-)"
  printf '%s%s' "${first}" "${rest}"
}

for domain in "${domains[@]}"; do
  type_prefix="$(upper_first "${domain}")"
  dir="${NS_ROOT}/${domain}"
  mkdir -p "${dir}"

  cat > "${dir}/${type_prefix}Metric.1.0.dsdl" <<EOT
uavcan.time.SynchronizedTimestamp.1.0 timestamp
float32 score
uavcan.si.sample.temperature.Scalar.1.0 thermal
uavcan.si.sample.pressure.Scalar.1.0 pressure
uavcan.si.sample.power.Scalar.1.0 power
@sealed
EOT

  cat > "${dir}/${type_prefix}Setpoint.1.0.dsdl" <<EOT
float32[3] target_vector
saturated int16 gain_q8
bool enabled
@sealed
EOT

  cat > "${dir}/${type_prefix}Limits.1.0.dsdl" <<EOT
float32 min_value
float32 max_value
float32 max_rate
float32 max_jerk
@sealed
EOT

  cat > "${dir}/${type_prefix}State.1.0.dsdl" <<EOT
civildrone.${domain}.${type_prefix}Metric.1.0 metric
civildrone.${domain}.${type_prefix}Setpoint.1.0 setpoint
civildrone.${domain}.${type_prefix}Limits.1.0 limits
uavcan.diagnostic.Record.1.1 last_diagnostic
uavcan.register.Value.1.0 tuning_register
@sealed
EOT

  cat > "${dir}/${type_prefix}Event.1.0.dsdl" <<EOT
@union
civildrone.${domain}.${type_prefix}Metric.1.0 metric
civildrone.${domain}.${type_prefix}State.1.0 state
uavcan.primitive.String.1.0 text
uavcan.register.Value.1.0 register_value
@sealed
EOT

  cat > "${dir}/${type_prefix}Batch.1.0.dsdl" <<EOT
uavcan.time.SynchronizedTimestamp.1.0 collected_at
civildrone.${domain}.${type_prefix}Event.1.0[<=12] events
civildrone.${domain}.${type_prefix}State.1.0[<=6] states
uint16 dropped_samples
@sealed
EOT

  cat > "${dir}/${type_prefix}Report.1.0.dsdl" <<EOT
uavcan.node.Heartbeat.1.0 heartbeat
civildrone.core.NodeIdentity.1.0 node
civildrone.${domain}.${type_prefix}Batch.1.0 batch
uavcan.si.sample.voltage.Scalar.1.0 bus_voltage
uavcan.si.sample.electric_current.Scalar.1.0 bus_current
@sealed
EOT

  cat > "${dir}/${type_prefix}Profile.1.0.dsdl" <<EOT
uint8 MODE_DISABLED = 0
uint8 MODE_STANDBY  = 1
uint8 MODE_ACTIVE   = 2
uint8 mode
civildrone.${domain}.${type_prefix}Limits.1.0 limits
civildrone.${domain}.${type_prefix}Setpoint.1.0 default_setpoint
uavcan.register.Value.1.0 profile_register
@sealed
EOT

  cat > "${dir}/${type_prefix}History.1.0.dsdl" <<EOT
civildrone.${domain}.${type_prefix}Report.1.0[<=4] reports
civildrone.${domain}.${type_prefix}Event.1.0[<=24] timeline
uavcan.time.SynchronizedTimestamp.1.0 last_update
@sealed
EOT

  cat > "${dir}/${type_prefix}Control.1.0.dsdl" <<EOT
uint8 opcode
civildrone.core.NodeIdentity.1.0 requester
civildrone.${domain}.${type_prefix}Setpoint.1.0 desired
uavcan.register.Value.1.0 register_override
@sealed
---
bool accepted
civildrone.${domain}.${type_prefix}State.1.0 resulting_state
civildrone.${domain}.${type_prefix}Event.1.0 resulting_event
uavcan.register.Value.1.0 effective_register
@sealed
EOT
done

# Vision common types.
cat > "${NS_ROOT}/vision/common/PixelFormat.1.0.dsdl" <<'EOT'
@union
uavcan.primitive.Empty.1.0 unknown
uint8 mono8
uint8 rgb8
uint8 bgr8
uint8 yuv420
uint8 nv12
@sealed
EOT

cat > "${NS_ROOT}/vision/common/FrameHeader.1.0.dsdl" <<'EOT'
uavcan.time.SynchronizedTimestamp.1.0 timestamp
uint64 frame_index
uint16 stream_id
uint16 camera_id
uavcan.si.sample.angle.Quaternion.1.0 sensor_attitude
@sealed
EOT

cat > "${NS_ROOT}/vision/common/RegionOfInterest.1.0.dsdl" <<'EOT'
uint16 x
uint16 y
uint16 width
uint16 height
@sealed
EOT

cat > "${NS_ROOT}/vision/common/BoundingBox2D.1.0.dsdl" <<'EOT'
float32 cx
float32 cy
float32 width
float32 height
float32 confidence
@sealed
EOT

cat > "${NS_ROOT}/vision/common/Keypoint2D.1.0.dsdl" <<'EOT'
float32 x
float32 y
float32 scale
float32 orientation
@sealed
EOT

cat > "${NS_ROOT}/vision/common/TensorShape.1.0.dsdl" <<'EOT'
uint32[<=8] dimensions
@sealed
EOT

cat > "${NS_ROOT}/vision/common/Descriptor256.1.0.dsdl" <<'EOT'
uint8[32] bytes
@sealed
EOT

cat > "${NS_ROOT}/vision/common/ObjectClass.1.0.dsdl" <<'EOT'
@union
uavcan.primitive.Empty.1.0 none
uint16 class_id
uavcan.primitive.String.1.0 class_name
@sealed
EOT

cat > "${NS_ROOT}/vision/common/InferenceRuntime.1.0.dsdl" <<'EOT'
uint8 BACKEND_CPU   = 0
uint8 BACKEND_GPU   = 1
uint8 BACKEND_NPU   = 2
uint8 BACKEND_FPGA  = 3
uint8 backend
float32 max_latency_ms
float32 target_fps
@sealed
EOT

cat > "${NS_ROOT}/vision/common/CameraIntrinsics.1.0.dsdl" <<'EOT'
float32 fx
float32 fy
float32 cx
float32 cy
float32[<=8] distortion
@sealed
EOT

cat > "${NS_ROOT}/vision/common/CameraExtrinsics.1.0.dsdl" <<'EOT'
uavcan.si.sample.length.Vector3.1.0 translation
uavcan.si.sample.angle.Quaternion.1.0 rotation
@sealed
EOT

cat > "${NS_ROOT}/vision/common/StreamingQoS.1.0.dsdl" <<'EOT'
uint32 bitrate_bps
uint16 mtu_bytes
float32 max_jitter_ms
float32 max_latency_ms
bool reliable_control_channel
@sealed
EOT

# Vision module types.
for idx in "${!vision_modules[@]}"; do
  module="${vision_modules[${idx}]}"
  pfx="$(upper_first "${module}")"
  next_module="${vision_modules[$(( (${idx} + 1) % ${#vision_modules[@]} ))]}"
  prev_module="${vision_modules[$(( (${idx} + ${#vision_modules[@]} - 1) % ${#vision_modules[@]} ))]}"
  next_pfx="$(upper_first "${next_module}")"
  prev_pfx="$(upper_first "${prev_module}")"

  dir="${NS_ROOT}/vision/${module}"
  mkdir -p "${dir}"

  cat > "${dir}/${pfx}ControlChannel.1.0.dsdl" <<EOT
uint16 channel_id
uint8 priority
bool reliable
float32 target_rate_hz
uavcan.register.Value.1.0 channel_register
civildrone.vision.common.StreamingQoS.1.0 qos
@sealed
EOT

  cat > "${dir}/${pfx}LowLevelControlCommand.1.0.dsdl" <<EOT
uint8 mode
float32[<=32] gains
float32[<=32] offsets
uavcan.register.Value.1.0 override
@sealed
EOT

  cat > "${dir}/${pfx}LowLevelControl.1.0.dsdl" <<EOT
civildrone.vision.${module}.${pfx}LowLevelControlCommand.1.0 command
@sealed
---
bool accepted
uavcan.register.Value.1.0 effective
uavcan.primitive.String.1.0 status
@sealed
EOT

  cat > "${dir}/${pfx}FrameMeta.1.0.dsdl" <<EOT
civildrone.vision.common.FrameHeader.1.0 header
civildrone.vision.common.PixelFormat.1.0 format
uint16 width
uint16 height
float32 exposure_ms
float32 gain_db
civildrone.vision.common.CameraIntrinsics.1.0 intrinsics
civildrone.vision.common.CameraExtrinsics.1.0 extrinsics
@sealed
EOT

  cat > "${dir}/${pfx}FramePacket.1.0.dsdl" <<EOT
civildrone.vision.${module}.${pfx}FrameMeta.1.0 meta
uint16 packet_index
uint16 packet_count
truncated uint8[<=1400] payload
@sealed
EOT

  cat > "${dir}/${pfx}Feature.1.0.dsdl" <<EOT
civildrone.vision.common.Keypoint2D.1.0 keypoint
civildrone.vision.common.Descriptor256.1.0 descriptor
float32 strength
@sealed
EOT

  cat > "${dir}/${pfx}FeatureSet.1.0.dsdl" <<EOT
civildrone.vision.${module}.${pfx}FrameMeta.1.0 meta
civildrone.vision.${module}.${pfx}Feature.1.0[<=512] features
@sealed
EOT

  cat > "${dir}/${pfx}Detection.1.0.dsdl" <<EOT
civildrone.vision.common.BoundingBox2D.1.0 box
civildrone.vision.common.ObjectClass.1.0 object_class
float32 confidence
uavcan.si.sample.length.Vector3.1.0 position_estimate
@sealed
EOT

  cat > "${dir}/${pfx}DetectionSet.1.0.dsdl" <<EOT
civildrone.vision.${module}.${pfx}FrameMeta.1.0 meta
civildrone.vision.${module}.${pfx}Detection.1.0[<=256] detections
@sealed
EOT

  cat > "${dir}/${pfx}Classification.1.0.dsdl" <<EOT
civildrone.vision.common.RegionOfInterest.1.0 roi
civildrone.vision.common.ObjectClass.1.0 object_class
float32 confidence
float32[<=16] embedding
@sealed
EOT

  cat > "${dir}/${pfx}InferenceTensor.1.0.dsdl" <<EOT
civildrone.vision.common.TensorShape.1.0 shape
float32 scale
int32 zero_point
truncated uint8[<=4096] quantized_data
@sealed
EOT

  cat > "${dir}/${pfx}GraphNode.1.0.dsdl" <<EOT
uint32 node_id
civildrone.vision.common.Keypoint2D.1.0 image_point
uavcan.si.sample.length.Vector3.1.0 world_point
float32 uncertainty
@sealed
EOT

  cat > "${dir}/${pfx}GraphEdge.1.0.dsdl" <<EOT
uint32 edge_id
civildrone.vision.${module}.${pfx}GraphNode.1.0 from_node
civildrone.vision.${next_module}.${next_pfx}GraphNode.1.0 to_node
float32 information
@sealed
EOT

  cat > "${dir}/${pfx}GraphState.1.0.dsdl" <<EOT
civildrone.vision.${module}.${pfx}GraphNode.1.0[<=1024] nodes
civildrone.vision.${module}.${pfx}GraphEdge.1.0[<=2048] edges
civildrone.vision.${prev_module}.${prev_pfx}GraphNode.1.0[<=64] upstream_nodes
@sealed
EOT

  cat > "${dir}/${pfx}PipelineEvent.1.0.dsdl" <<EOT
@union
civildrone.vision.${module}.${pfx}FramePacket.1.0 packet
civildrone.vision.${module}.${pfx}FeatureSet.1.0 features
civildrone.vision.${module}.${pfx}DetectionSet.1.0 detections
civildrone.vision.${module}.${pfx}InferenceTensor.1.0 tensor
civildrone.vision.${module}.${pfx}GraphState.1.0 graph
uavcan.diagnostic.Record.1.1 diagnostic
@sealed
EOT

  cat > "${dir}/${pfx}BatchReport.1.0.dsdl" <<EOT
uavcan.node.Heartbeat.1.0 heartbeat
civildrone.core.NodeIdentity.1.0 node
civildrone.vision.${module}.${pfx}PipelineEvent.1.0[<=32] events
civildrone.vision.${next_module}.${next_pfx}PipelineEvent.1.0[<=4] downstream_events
civildrone.vision.common.InferenceRuntime.1.0 runtime
uavcan.register.Value.1.0 active_model
@sealed
EOT

  cat > "${dir}/${pfx}ControlPlane.1.0.dsdl" <<EOT
civildrone.core.NodeIdentity.1.0 requester
civildrone.vision.${module}.${pfx}ControlChannel.1.0 channel
civildrone.vision.${module}.${pfx}LowLevelControlCommand.1.0 control
civildrone.vision.common.InferenceRuntime.1.0 runtime
uavcan.register.Value.1.0 profile
@sealed
---
bool accepted
civildrone.vision.${module}.${pfx}BatchReport.1.0 report
uavcan.primitive.String.1.0 message
@sealed
EOT

  cat > "${dir}/${pfx}ModelUpdate.1.0.dsdl" <<EOT
uavcan.primitive.String.1.0 model_name
uint32 model_version
civildrone.vision.common.TensorShape.1.0 input_shape
truncated uint8[<=4096] model_chunk
uavcan.register.Value.1.0 checksum
@sealed
EOT

  cat > "${dir}/${pfx}ModelChunk.1.0.dsdl" <<EOT
uavcan.primitive.String.1.0 model_name
uint32 model_version
civildrone.vision.common.TensorShape.1.0 input_shape
truncated uint8[<=4096] model_chunk
uavcan.register.Value.1.0 checksum
@sealed
EOT
done

cat > "${NS_ROOT}/vision/VisionNodeSnapshot.1.0.dsdl" <<'EOT'
civildrone.core.NodeIdentity.1.0 node
civildrone.core.PoseKinematics.1.0 pose
EOT
for module in "${vision_modules[@]}"; do
  pfx="$(upper_first "${module}")"
  echo "civildrone.vision.${module}.${pfx}BatchReport.1.0 ${module}_report" \
    >> "${NS_ROOT}/vision/VisionNodeSnapshot.1.0.dsdl"
done
cat >> "${NS_ROOT}/vision/VisionNodeSnapshot.1.0.dsdl" <<'EOT'
@sealed
EOT

cat > "${NS_ROOT}/vision/VisionFleetSnapshot.1.0.dsdl" <<'EOT'
uavcan.time.SynchronizedTimestamp.1.0 timestamp
civildrone.vision.VisionNodeSnapshot.1.0[<=64] nodes
uavcan.primitive.String.1.0 deployment_name
@sealed
EOT

cat > "${NS_ROOT}/vision/VisionMissionControl.1.0.dsdl" <<'EOT'
civildrone.core.NodeIdentity.1.0 requester
civildrone.vision.VisionFleetSnapshot.1.0 baseline
uavcan.register.Value.1.0 mission_register
@sealed
---
bool accepted
civildrone.vision.VisionFleetSnapshot.1.0 updated
uavcan.primitive.String.1.0 status
@sealed
EOT

cat > "${NS_ROOT}/vision/VslamMapChunk.1.0.dsdl" <<'EOT'
uint32 chunk_id
civildrone.vision.vslam.VslamGraphState.1.0 graph
civildrone.vision.localization.LocalizationGraphState.1.0 localization_graph
truncated uint8[<=8192] compressed_payload
@sealed
EOT

cat > "${NS_ROOT}/vision/SceneUnderstanding.1.0.dsdl" <<'EOT'
civildrone.vision.detection.DetectionDetectionSet.1.0 detections
civildrone.vision.segmentation.SegmentationDetectionSet.1.0 segments
civildrone.vision.classification.ClassificationClassification.1.0[<=256] classes
civildrone.vision.tracking.TrackingGraphState.1.0 tracks
@sealed
EOT

cat > "${NS_ROOT}/vision/VideoSessionControl.1.0.dsdl" <<'EOT'
uint32 session_id
civildrone.vision.stream.StreamControlChannel.1.0 stream_channel
civildrone.vision.codec.CodecControlChannel.1.0 codec_channel
civildrone.vision.transport.TransportControlChannel.1.0 transport_channel
uavcan.register.Value.1.0 session_register
@sealed
---
bool accepted
civildrone.vision.VisionNodeSnapshot.1.0 snapshot
uavcan.primitive.String.1.0 message
@sealed
EOT

cat > "${NS_ROOT}/vision/VideoArchiveChunk.1.0.dsdl" <<'EOT'
uint32 archive_id
uint32 chunk_index
civildrone.vision.stream.StreamFramePacket.1.0[<=8] packets
truncated uint8[<=8192] compressed_index
@sealed
EOT

cat > "${NS_ROOT}/vision/InferenceCampaign.1.0.dsdl" <<'EOT'
uint32 campaign_id
civildrone.vision.inference.InferenceModelChunk.1.0[<=64] updates
civildrone.vision.fusion.FusionControlChannel.1.0[<=64] controls
civildrone.vision.VisionFleetSnapshot.1.0 fleet
@sealed
EOT

# Fleet-level high-fanout structures.
cat > "${NS_ROOT}/fleet/PlatformDigest.1.0.dsdl" <<'EOT'
civildrone.core.SystemSnapshot.1.0 snapshot
civildrone.vision.VisionNodeSnapshot.1.0 vision
EOT
for domain in "${domains[@]}"; do
  pfx="$(upper_first "${domain}")"
  echo "civildrone.${domain}.${pfx}Report.1.0 ${domain}_report" >> "${NS_ROOT}/fleet/PlatformDigest.1.0.dsdl"
done
cat >> "${NS_ROOT}/fleet/PlatformDigest.1.0.dsdl" <<'EOT'
@sealed
EOT

cat > "${NS_ROOT}/fleet/PlatformControlVector.1.0.dsdl" <<'EOT'
civildrone.core.NodeIdentity.1.0 target
EOT
for domain in "${domains[@]}"; do
  pfx="$(upper_first "${domain}")"
  echo "civildrone.${domain}.${pfx}Setpoint.1.0 ${domain}_setpoint" >> "${NS_ROOT}/fleet/PlatformControlVector.1.0.dsdl"
done
cat >> "${NS_ROOT}/fleet/PlatformControlVector.1.0.dsdl" <<'EOT'
@sealed
EOT

cat > "${NS_ROOT}/fleet/FleetSnapshot.1.0.dsdl" <<'EOT'
uavcan.time.SynchronizedTimestamp.1.0 timestamp
civildrone.fleet.PlatformDigest.1.0[<=64] platforms
uavcan.primitive.String.1.0 region_name
@sealed
EOT

cat > "${NS_ROOT}/fleet/FleetEvent.1.0.dsdl" <<'EOT'
@union
civildrone.fleet.PlatformDigest.1.0 digest
civildrone.fleet.FleetSnapshot.1.0 snapshot
uavcan.diagnostic.Record.1.1 diagnostic
uavcan.primitive.String.1.0 text
@sealed
EOT

cat > "${NS_ROOT}/fleet/FleetPlan.1.0.dsdl" <<'EOT'
uint32 campaign_id
civildrone.core.SurveyPlan.1.0[<=8] plans
civildrone.fleet.PlatformControlVector.1.0[<=64] initial_controls
@sealed
EOT

cat > "${NS_ROOT}/fleet/FleetCommand.1.0.dsdl" <<'EOT'
civildrone.core.NodeIdentity.1.0 requester
civildrone.fleet.FleetPlan.1.0 plan
uavcan.register.Value.1.0 command_register
@sealed
---
bool accepted
civildrone.fleet.FleetSnapshot.1.0 resulting_snapshot
uavcan.primitive.String.1.0 status
@sealed
EOT

cat > "${NS_ROOT}/fleet/FleetDiagnostics.1.0.dsdl" <<'EOT'
civildrone.fleet.FleetEvent.1.0[<=128] events
uavcan.diagnostic.Record.1.1[<=128] records
@sealed
EOT

cat > "${NS_ROOT}/fleet/SurveyCampaign.1.0.dsdl" <<'EOT'
uint32 campaign_id
civildrone.core.SurveyTask.1.0[<=256] tasks
civildrone.core.SafetyEnvelope.1.0 global_safety
@sealed
EOT

cat > "${NS_ROOT}/fleet/SurveyCampaignState.1.0.dsdl" <<'EOT'
civildrone.fleet.SurveyCampaign.1.0 campaign
civildrone.fleet.FleetSnapshot.1.0 latest_snapshot
civildrone.fleet.FleetDiagnostics.1.0 diagnostics
@sealed
EOT

cat > "${NS_ROOT}/fleet/SurveyCampaignControl.1.0.dsdl" <<'EOT'
uint8 action
civildrone.fleet.SurveyCampaign.1.0 campaign
uavcan.register.Value.1.0 config
@sealed
---
bool accepted
civildrone.fleet.SurveyCampaignState.1.0 state
@sealed
EOT

cat > "${NS_ROOT}/fleet/FleetRegisterMirror.1.0.dsdl" <<'EOT'
civildrone.core.NodeIdentity.1.0 node
uavcan.register.Value.1.0[<=128] values
@sealed
EOT

cat > "${NS_ROOT}/fleet/FleetRegisterSync.1.0.dsdl" <<'EOT'
civildrone.fleet.FleetRegisterMirror.1.0[<=64] mirrors
uavcan.time.SynchronizedTimestamp.1.0 synchronized_at
@sealed
EOT

# Synthetic workload family to expose combinatorial path complexity.
for i in $(seq 1 "${CLASSIC_WORKLOAD_COUNT}"); do
  d1="${domains[$(( (i - 1) % ${#domains[@]} ))]}"
  d2="${domains[$(( (i + 7) % ${#domains[@]} ))]}"
  d3="${domains[$(( (i + 13) % ${#domains[@]} ))]}"
  p1="$(upper_first "${d1}")"
  p2="$(upper_first "${d2}")"
  p3="$(upper_first "${d3}")"

  file_c="${NS_ROOT}/workload/Composite${i}.1.0.dsdl"
  cat > "${file_c}" <<EOT
civildrone.core.NodeIdentity.1.0 node
civildrone.core.PoseKinematics.1.0 pose
civildrone.${d1}.${p1}State.1.0 ${d1}_state
civildrone.${d2}.${p2}Report.1.0 ${d2}_report
civildrone.${d3}.${p3}History.1.0 ${d3}_history
uavcan.register.Value.1.0 tuning
truncated uint8[<=256] opaque_payload
EOT
  cat >> "${file_c}" <<'EOT'
@sealed
EOT

  file_v="${NS_ROOT}/workload/Variant${i}.1.0.dsdl"
  cat > "${file_v}" <<EOT
@union
civildrone.${d1}.${p1}Event.1.0 ${d1}_event
civildrone.${d2}.${p2}Event.1.0 ${d2}_event
civildrone.${d3}.${p3}Event.1.0 ${d3}_event
civildrone.workload.Composite${i}.1.0 composite
uavcan.primitive.String.1.0 text
@sealed
EOT
done

# Vision-heavy synthetic workload family.
for i in $(seq 1 "${VISION_WORKLOAD_COUNT}"); do
  m1="${vision_modules[$(( (i - 1) % ${#vision_modules[@]} ))]}"
  m2="${vision_modules[$(( (i + 5) % ${#vision_modules[@]} ))]}"
  m3="${vision_modules[$(( (i + 11) % ${#vision_modules[@]} ))]}"
  p1="$(upper_first "${m1}")"
  p2="$(upper_first "${m2}")"
  p3="$(upper_first "${m3}")"

  vcomp="${NS_ROOT}/workload/VisionComposite${i}.1.0.dsdl"
  cat > "${vcomp}" <<EOT
civildrone.core.NodeIdentity.1.0 node
civildrone.vision.${m1}.${p1}BatchReport.1.0 ${m1}_batch
civildrone.vision.${m2}.${p2}DetectionSet.1.0 ${m2}_detections
civildrone.vision.${m3}.${p3}GraphState.1.0 ${m3}_graph
civildrone.vision.VisionNodeSnapshot.1.0 snapshot
civildrone.fleet.PlatformDigest.1.0 fleet_digest
truncated uint8[<=4096] opaque_payload
EOT
  cat >> "${vcomp}" <<'EOT'
@sealed
EOT

  vvar="${NS_ROOT}/workload/VisionVariant${i}.1.0.dsdl"
  cat > "${vvar}" <<EOT
@union
civildrone.vision.${m1}.${p1}PipelineEvent.1.0 ${m1}_event
civildrone.vision.${m2}.${p2}PipelineEvent.1.0 ${m2}_event
civildrone.vision.${m3}.${p3}PipelineEvent.1.0 ${m3}_event
civildrone.workload.VisionComposite${i}.1.0 composite
uavcan.primitive.String.1.0 text
@sealed
EOT
done

for i in $(seq 1 "${MEGABUNDLE_COUNT}"); do
  m1="${vision_modules[$(( (i + 2) % ${#vision_modules[@]} ))]}"
  m2="${vision_modules[$(( (i + 9) % ${#vision_modules[@]} ))]}"
  p1="$(upper_first "${m1}")"
  p2="$(upper_first "${m2}")"

  mega="${NS_ROOT}/workload/MegaBundle${i}.1.0.dsdl"
  cat > "${mega}" <<EOT
uavcan.time.SynchronizedTimestamp.1.0 timestamp
civildrone.workload.Composite$(( (i % CLASSIC_WORKLOAD_COUNT) + 1 )).1.0 classic_composite
civildrone.workload.VisionComposite$(( (i % VISION_WORKLOAD_COUNT) + 1 )).1.0 vision_composite
civildrone.vision.${m1}.${p1}InferenceTensor.1.0 ${m1}_tensor
civildrone.vision.${m2}.${p2}InferenceTensor.1.0 ${m2}_tensor
civildrone.vision.InferenceCampaign.1.0 campaign
uavcan.register.Value.1.0 registry_hint
@sealed
EOT
done

echo "Generated benchmark corpus under: ${NS_ROOT}"
find "${NS_ROOT}" -type f -name '*.dsdl' | wc -l | awk '{print "Total .dsdl files:", $1}'
