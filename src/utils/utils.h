#ifndef CXXNET_UTILS_UTILS_H_
#define CXXNET_UTILS_UTILS_H_
/*!
 * \file utils.h
 * \brief simple utils to support the code, adopted from xgboost
 * \author Tianqi Chen
 */
#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <map>
#include <cstdarg>
#include <cassert> // orc
#include <time.h>
#include <sys/resource.h>
#include <string.h>

#if !defined(__GNUC__)
#define fopen64 std::fopen
#endif
#ifdef _MSC_VER
// NOTE: sprintf_s is not equivalent to snprintf,
// they are equivalent when success, which is sufficient for our case
#define snprintf sprintf_s
#define vsnprintf vsprintf_s
#else
#ifdef _FILE_OFFSET_BITS
#if _FILE_OFFSET_BITS == 32
#pragma message ("Warning: FILE OFFSET BITS defined to be 32 bit")
#endif
#endif

#ifdef __APPLE__
#define off64_t off_t
#define fopen64 std::fopen
#endif

extern "C" {
#include <sys/types.h>
}
#endif

#ifdef _MSC_VER
typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;
typedef long int64_t;
#else
#include <inttypes.h>
#endif

#ifndef CHECK
#define CHECK(ARGS) cxxnet::utils::Check(ARGS, "Assert Error at %s: %d", __FILE__, __LINE__);
#endif
#define CUDA_CHECK(ARGS) CHECK(ARGS==0);

namespace textnet {
/*! \brief namespace for helper utils of the project */
namespace utils {

/*! \brief error message buffer length */
const int kPrintBuffer = 1 << 12;

#ifndef CXXNET_CUSTOMIZE_MSG_
/*!
 * \brief handling of Assert error, caused by in-apropriate input
 * \param msg error message
 */
inline void HandleAssertError(const char *msg) {
  // fprintf(stderr, "AssertError:%s\n", msg);
  fprintf(stderr, "\33[91mAssertError\33[0m:%s\n", msg);
  assert(false); // orc for looking stack
  exit(-1);
}
/*!
 * \brief handling of Check error, caused by in-apropriate input
 * \param msg error message
 */
inline void HandleCheckError(const char *msg) {
  // fprintf(stderr, "%s\n", msg);
  fprintf(stderr, "\33[91m%s\33[0m\n", msg);
  assert(false); // orc for looking stack
  exit(-1);
}
inline void HandlePrint(const char *msg) {
  char time_info[64];
  time_t t = time(0);
  strftime(time_info, sizeof(time_info), "%X", localtime(&t));
  printf("[%s] %s", time_info, msg);
}
#else
// include declarations, some one must implement this
void HandleAssertError(const char *msg);
void HandleCheckError(const char *msg);
void HandlePrint(const char *msg);
#endif

/*! \brief printf, print message to the console */
inline void Printf(const char *fmt, ...) {
  std::string msg(kPrintBuffer, '\0');
  va_list args;
  va_start(args, fmt);
  vsnprintf(&msg[0], kPrintBuffer, fmt, args);
  va_end(args);
  HandlePrint(msg.c_str());
}
/*! \brief portable version of snprintf */
inline int SPrintf(char *buf, size_t size, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int ret = vsnprintf(buf, size, fmt, args);
  va_end(args);
  return ret;
}

/*!\brief same as check, but do not exit.*/
inline bool Require(bool exp, const char *fmt, ...) {
  if (!exp) {
    std::string msg(kPrintBuffer, '\0');
    va_list args;
    va_start(args, fmt);
    vsnprintf(&msg[0], kPrintBuffer, fmt, args);
    va_end(args);
    HandlePrint(msg.c_str());
	return false;
  } else {
	return true;
  }
}

/*! \brief assert an condition is true, use this to handle debug information */
inline void Assert(bool exp, const char *fmt, ...) {
#if DEBUG // orc this if only for debug
  if (!exp) {
    std::string msg(kPrintBuffer, '\0');
    va_list args;
    va_start(args, fmt);
    vsnprintf(&msg[0], kPrintBuffer, fmt, args);
    va_end(args);
    HandleAssertError(msg.c_str());
  }
#endif
}

/*!\brief same as assert, but this is intended to be used as message for user*/
inline void Check(bool exp, const char *fmt, ...) {
  if (!exp) {
    std::string msg(kPrintBuffer, '\0');
    va_list args;
    va_start(args, fmt);
    vsnprintf(&msg[0], kPrintBuffer, fmt, args);
    va_end(args);
    HandleCheckError(msg.c_str());
  }
}

/*! \brief report error message, same as check */
inline void Error(const char *fmt, ...) {
  {
    std::string msg(kPrintBuffer, '\0');
    va_list args;
    va_start(args, fmt);
    vsnprintf(&msg[0], kPrintBuffer, fmt, args);
    va_end(args);
    HandleCheckError(msg.c_str());
  }
}

/*! \brief replace fopen, report error when the file open fails */
inline std::FILE *FopenCheck(const char *fname, const char *flag) {
  std::FILE *fp = fopen64(fname, flag);
  Check(fp != NULL, "can not open file \"%s\"\n", fname);
  return fp;
}

inline void ShowMemoryUse() {
  struct rusage ru;
  memset(&ru, 0, sizeof(struct rusage));
  getrusage(RUSAGE_SELF, &ru);
  float total_memory = 1.0 * (ru.ru_maxrss + ru.ru_ixrss + ru.ru_idrss + ru.ru_isrss) / 1024.0;
  printf("\033[33m[Memory] Total Memory Use: %.4f MB \t Resident: %ld Shared: %ld UnshareData: %ld UnshareStack: %ld \33[0m\n", 
		 total_memory, ru.ru_maxrss, ru.ru_ixrss, ru.ru_idrss, ru.ru_isrss);
}

}  // namespace utils
// easy utils that can be directly acessed in xgboost
/*! \brief get the beginning address of a vector */
template<typename T>
inline T *BeginPtr(std::vector<T> &vec) {
  if (vec.size() == 0) {
    return NULL;
  } else {
    return &vec[0];
  }
}
/*! \brief get the beginning address of a vector */
template<typename T>
inline const T *BeginPtr(const std::vector<T> &vec) {
  if (vec.size() == 0) {
    return NULL;
  } else {
    return &vec[0];
  }
}

using namespace std;

inline int str2int(string s) {
  return str2int(s.c_str());
}

inline int str2float(string s) {
  return str2float(s.c_str());
}

inline int str2int(const char *p) {
  int i = 0;
  stringstream ss(p);
  ss >> i;
  return i;
}

inline float str2float(const char *p) {
  float f = 0;
  stringstream ss(p);
  ss >> f;
  return f;
}

inline string int2str(int i){
  stringstream s;
  s << i;
  return s.str();
}

inline string float2str(float f){
  stringstream s;
  s << f;
  return s.str();
}

}  // namespace textnet
#endif  // CXXNET_UTILS_UTILS_H_
