#pragma once
#include <iostream>

inline __host__ std::ostream& operator<<(std::ostream& os, const uint2& a) {
  return os << "x: " << a.x << ", y: " << a.y;
}
inline __host__ std::ostream& operator<<(std::ostream& os, const int2& a) {
  return os << "x: " << a.x << ", y: " << a.y;
}
inline __host__ std::ostream& operator<<(std::ostream& os, const __uint128_t& a) {
  return os << a;
}
inline __host__ std::ostream& operator<<(std::ostream& os, const uint4& a) {
  return os << "x: " << a.x << ", y: " << a.y << ", z: " << a.z << ", w: " << a.w;
}
inline __host__ std::ostream& operator<<(std::ostream& os, const int4& a) {
  return os << "x: " << a.x << ", y: " << a.y << ", z: " << a.z << ", w: " << a.w;
}
inline __host__ std::ostream& operator<<(std::ostream& os, const ushort2& a) {
  return os << "x: " << a.x << ", y: " << a.y;
}
inline __host__ std::ostream& operator<<(std::ostream& os, const short2& a) {
  return os << "x: " << a.x << ", y: " << a.y;
}
inline __host__ std::ostream& operator<<(std::ostream& os, const ushort4& a) {
  return os << "x: " << a.x << ", y: " << a.y << ", z: " << a.z << ", w: " << a.w;
}
inline __host__ std::ostream& operator<<(std::ostream& os, const short4& a) {
  return os << "x: " << a.x << ", y: " << a.y << ", z: " << a.z << ", w: " << a.w;
}
inline __host__ std::ostream& operator<<(std::ostream& os, const ulonglong2& a) {
  return os << "x: " << a.x << ", y: " << a.y;
}
inline __host__ std::ostream& operator<<(std::ostream& os, const longlong2& a) {
  return os << "x: " << a.x << ", y: " << a.y;
}
inline __host__ std::ostream& operator<<(std::ostream& os, const ulonglong4& a) {
  return os << "x: " << a.x << ", y: " << a.y << ", z: " << a.z << ", w: " << a.w;
}
inline __host__ std::ostream& operator<<(std::ostream& os, const longlong4& a) {
  return os << "x: " << a.x << ", y: " << a.y << ", z: " << a.z << ", w: " << a.w;
}
inline __host__ std::ostream& operator<<(std::ostream& os, const float2& a) {
  return os << "x: " << a.x << ", y: " << a.y;
}
inline __host__ std::ostream& operator<<(std::ostream& os, const float4& a) {
  return os << "x: " << a.x << ", y: " << a.y << ", z: " << a.z << ", w: " << a.w;
}
inline __host__ std::ostream& operator<<(std::ostream& os, const double2& a) {
  return os << "x: " << a.x << ", y: " << a.y;
}
inline __host__ std::ostream& operator<<(std::ostream& os, const uchar2& a) {
  return os << "x: " << static_cast<unsigned>(a.x) << ", y: " << static_cast<unsigned>(a.y);
}
inline __host__ std::ostream& operator<<(std::ostream& os, const char2& a) {
  return os << "x: " << static_cast<int>(a.x) << ", y: " << static_cast<int>(a.y);
}
inline __host__ std::ostream& operator<<(std::ostream& os, const dim3& a) {
  return os << "x: " << static_cast<int>(a.x) << ", y: " << static_cast<int>(a.y)
    << ", z: " << static_cast<unsigned>(a.z);
}
inline __host__ std::ostream& operator<<(std::ostream& os, const uchar4& a) {
  return os << "x: " << static_cast<unsigned>(a.x) << ", y: " << static_cast<unsigned>(a.y)
    << ", z: " << static_cast<unsigned>(a.z) << ", w: " << static_cast<unsigned>(a.w);
}
inline __host__ std::ostream& operator<<(std::ostream& os, const char4& a) {
  return os << "x: " << static_cast<int>(a.x) << ", y: " << static_cast<int>(a.y)
    << ", z: " << static_cast<int>(a.z) << ", w: " << static_cast<int>(a.w);
}
inline __host__ __device__ bool operator==(const uint2& lhs, const uint2& rhs) {
  return (lhs.x == rhs.x && lhs.y == rhs.y);
}
inline __host__ __device__ bool operator==(const uint4& lhs, const uint4& rhs) {
  return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w);
}
inline __host__ __device__ bool operator==(const ushort2& lhs, const ushort2& rhs) {
  return (lhs.x == rhs.x && lhs.y == rhs.y);
}
inline __host__ __device__ bool operator==(const ushort4& lhs, const ushort4& rhs) {
  return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w);
}
inline __host__ __device__ bool operator==(const ulonglong2& lhs, const ulonglong2& rhs) {
  return (lhs.x == rhs.x && lhs.y == rhs.y);
}
inline __host__ __device__ bool operator==(const uchar2& lhs, const uchar2& rhs) {
  return (lhs.x == rhs.x && lhs.y == rhs.y);
}
inline __host__ __device__ bool operator==(const uchar4& lhs, const uchar4& rhs) {
  return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w);
}
inline __host__ __device__ bool operator==(const int2& lhs, const int2& rhs) {
  return (lhs.x == rhs.x && lhs.y == rhs.y);
}
inline __host__ __device__ bool operator==(const int4& lhs, const int4& rhs) {
  return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w);
}
inline __host__ __device__ bool operator==(const short2& lhs, const short2& rhs) {
  return (lhs.x == rhs.x && lhs.y == rhs.y);
}
inline __host__ __device__ bool operator==(const short4& lhs, const short4& rhs) {
  return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w);
}
inline __host__ __device__ bool operator==(const longlong2& lhs, const longlong2& rhs) {
  return (lhs.x == rhs.x && lhs.y == rhs.y);
}
inline __host__ __device__ bool operator==(const char2& lhs, const char2& rhs) {
  return (lhs.x == rhs.x && lhs.y == rhs.y);
}
inline __host__ __device__ bool operator==(const char4& lhs, const char4& rhs) {
  return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w);
}
inline __host__ __device__ bool operator==(const float2& lhs, const float2& rhs) {
  return (lhs.x == rhs.x && lhs.y == rhs.y);
}
inline __host__ __device__ bool operator==(const float4& lhs, const float4& rhs) {
  return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w);
}
inline __host__ __device__ bool operator==(const double2& lhs, const double2& rhs) {
  return (lhs.x == rhs.x && lhs.y == rhs.y);
}
inline __host__ __device__ uint4 operator+(const uint4& lhs, const uint4& rhs) {
  uint4 ret = {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w};
  return ret;
}