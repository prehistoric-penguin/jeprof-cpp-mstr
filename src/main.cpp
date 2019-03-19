#define GLOG_STL_LOGGING_FOR_UNORDERED
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <glog/stl_logging.h>
#include <gperftools/profiler.h>
#include <sys/types.h>
#include <unistd.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/functional/hash.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/regex.hpp>
#include <cctype>
#include <chrono>
#include <csignal>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mmap_file.h"
#include "thread_pool.h"
#include "utils.h"
DEFINE_string(program, "", "Name of execuable file");
DEFINE_string(profile, "", "Name of profile name");
DEFINE_string(dotFileName, "/tmp/jeprof_cpp.dot", "");
DEFINE_string(thread, "", "Focous on this thread");
DEFINE_bool(useSymbolizedProfile, false, "TODO");
DEFINE_bool(useSymbolPage, false, "TODO");
DEFINE_bool(functions, true, "TODO");
DEFINE_double(nodeFraction, 0.005, "TODO");
DEFINE_double(edgeFraction, 0.001, "TODO");
DEFINE_int32(maxDegree, 8, "");
DEFINE_int32(nodeCount, 80, "TODO");
DEFINE_int32(addr2line_arguments, 1000,
             "Argument number for calling addr2line");
DEFINE_bool(showBytes, true, "");
DEFINE_bool(pdf, false, "");
DEFINE_bool(heapCheck, false, "");
DEFINE_bool(svg, false, "");
DEFINE_bool(web, false, "");

const std::string kPprofVersion = "2.0";
const std::string kObjdump = "objdump";
const std::string kNm = "nm";
const std::string kAddr2Line = "addr2line";
const std::string kCppFilt = "c++filt";
const std::string kDot = "dot";

const std::string kHeapPage = "/pprof/heap";
const std::string kProfilePage = "/pprof/prifile";
const std::string kGrowthPage = "/pprof/growth";
const std::string kContentionPage = "/pprof/contention";
const std::string kWallPage = "/pprof/wall(?:\\?.*)?";  // FIXME
const std::string KFilteredProfilePage = "/pprof/filteredprofile(?:\\?.*)?";
const std::string kSymbolPage = "/pprof/symbol";
const std::string kProgramNamePage = "/pprof/cmdline";
const std::vector<std::string> kProfiles = {
    kHeapPage,   kProfilePage,    kGrowthPage, kWallPage, KFilteredProfilePage,
    kSymbolPage, kProgramNamePage};

const std::string kUnknownBinary = "(unknown)";
const size_t kAddressLength = 16;
const std::string kDevNull = "/dev/null";

const std::string kSepSymbol = "_fini";
const std::string kTmpFileSym = "/tmp/jeprof$$.sym";  // FIXME
const std::string kTmpFilePs = "/tmp/jeprof$$";
std::string gProfileType;

detail::thread_pool gThreadPool;
detail::thread_pool gSubThreadPool;

namespace boost {
inline size_t hash_value(const boost::string_ref& r) {
  return hash_range(r.begin(), r.end());
}

template <>
inline string_ref copy_range<string_ref, iterator_range<char const*>>(
    iterator_range<char const*> const& r) {
  return string_ref(begin(r), end(r) - begin(r));
}
}  // namespace boost

namespace std {
template <>
struct hash<std::vector<size_t>> {
  size_t operator()(const std::vector<size_t>& vec) const {
    return boost::hash_value(vec);
  }
};

template <>
struct hash<std::vector<std::string>> {
  size_t operator()(const std::vector<std::string>& vec) const {
    return boost::hash_value(vec);
  }
};

template <>
struct hash<boost::string_ref> {
  size_t operator()(boost::string_ref ref) const {
    return boost::hash_range(ref.begin(), ref.end());
  }
};

template <>
struct hash<std::vector<boost::string_ref>> {
  size_t operator()(const std::vector<boost::string_ref>& vec) const {
    return boost::hash_range(vec.begin(), vec.end());
  }
};
}  // namespace std
using ProfileMap = std::unordered_map<std::vector<size_t>, size_t>;
using FinalProfileMap =
    std::unordered_map<std::vector<boost::string_ref>, size_t>;

// it shouldn't be used after profilemap's erasing or destruction operation
using ShadowProfileMap = std::unordered_map<boost::string_ref, size_t>;

using TranslatedStacks =
    std::vector<std::pair<std::vector<std::string>, size_t>>;

using SmallUint64Vector = boost::container::small_vector<size_t, 64>;
struct Profile {
  void AddEntry(const std::vector<size_t>& stack, size_t count) {
    std::lock_guard<std::mutex> guard(mutex_);
    data_[stack] += count;
  }

  ProfileMap data_;
  std::mutex mutex_;
};

struct ThreadProfile {
  std::map<std::string, Profile> data_;
};

struct PCS {
  void AddPC(const std::vector<size_t>& stack) {
    std::lock_guard<std::mutex> guard(mutex_);
    data_.insert(stack.begin(), stack.end());
  }
  std::unordered_set<size_t> data_;
  std::mutex mutex_;
};

struct Symbols {
  std::unordered_map<size_t, std::vector<std::string>> data_;
};

struct LibraryEntry {
  std::string lib_;
  size_t start_;
  size_t finish_;
  size_t offset_;
};

struct SymbolTable {
  std::unordered_map<std::string, std::vector<size_t>> data_;
};

struct Context {
  std::string vesion_;
  int period_;
  std::unique_ptr<Profile> profile_;
  std::unique_ptr<ThreadProfile> threads_;
  std::vector<LibraryEntry> libs_;
  std::unique_ptr<PCS> pcs_;
  Symbols symbols_;
};

class Marker {
 public:
  Marker(const char* f)
      : time_point_(std::chrono::steady_clock::now()), func_(f) {}

  ~Marker() {
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - time_point_)
                    .count() /
                1000.0;
    LOG(INFO) << "time cost in " << func_ << ": " << diff << " ms" << std::endl;
  }

 private:
  std::chrono::steady_clock::time_point time_point_;
  const char* func_;
};

inline std::ostream& operator<<(std::ostream& out, const Profile& p) {
  out << '[' << p.data_ << ']';
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const LibraryEntry& l) {
  out << '<' << l.lib_ << std::hex << "," << l.start_ << "," << l.finish_ << ","
      << l.offset_ << '>';
  return out;
}

template <typename Arg, typename... Ts>
std::string ShellEscape(Arg i, Ts... all) {
  static_assert(std::is_same<Arg, std::string>::value ||
                    std::is_same<Arg, const char*>::value,
                "Type missmatch, std::string or const char* required");

  std::string args[] = {i, all...};
  return boost::algorithm::join(args, " ");
}

bool IsValidSepAddress(size_t x) {
  return x != std::numeric_limits<size_t>::max();
}

using smatch_ref = boost::match_results<boost::string_ref::const_iterator>;
smatch_ref RegexMatch(boost::string_ref target, const boost::regex& re) {
  smatch_ref base;
  if (boost::regex_match(target.begin(), target.end(), base, re)) {
    return base;
  }
  return smatch_ref{};
}

boost::smatch RegexMatch(const std::string& target, const boost::regex& re);
void Usage();
boost::string_ref ReadProfileHeader(std::vector<boost::string_ref>::iterator&,
                                    std::vector<boost::string_ref>::iterator);
bool IsSymbolizedProfileFile();
bool IsProfileURL(const std::string&);
bool IsSymbolizedProfileFile(const std::string&);
void ConfigureObjTools();
std::string ConfigureTool(const std::string);
std::string FetchDynamicProfile(const std::string&, const std::string&, bool,
                                bool);
Context ReadProfile(const std::string&, const std::string&);
Context ReadThreadedHeapProfile(const std::string&, const std::string&,
                                const std::string&,
                                std::vector<boost::string_ref>::iterator&,
                                std::vector<boost::string_ref>::iterator);
std::vector<std::string> ReadMappedLibraries(
    std::vector<boost::string_ref>::iterator&,
    std::vector<boost::string_ref>::iterator);
std::vector<std::string> ReadMemoryMap(
    std::vector<boost::string_ref>::iterator&,
    std::vector<boost::string_ref>::iterator);
std::vector<size_t> AdjustSamples(int, int, size_t, size_t, size_t, size_t);
std::vector<size_t> FixCallerAddresses(boost::string_ref);
size_t AddressSub(size_t, size_t);
void AddEntries(Profile&, PCS&, const std::vector<size_t>&, size_t);
void AddEntry(Profile, const std::vector<size_t>&, int);
std::vector<LibraryEntry> ParseLibraries(const std::vector<std::string>& map,
                                         PCS& pcs);
std::string FindLibrary(const std::string& lib);
std::string DebuggingLibrary(const std::string&);

std::tuple<size_t, size_t, size_t> ParseTextSectinoHeader(const std::string&);
void ParseTextSectionHeaderFromObjdump(const std::string&);
std::vector<std::string> ExecuteCommand(const std::string&);
bool System(const std::string&);

std::tuple<size_t, size_t, size_t> ParseTextSectinoHeaderFromObjdump(
    const std::string&);
size_t AddressAdd(size_t, size_t);
void MergeSymbols(Symbols&, const Symbols&);
void MergeSymbols(Symbols&, Symbols&&);

Symbols ExtractSymbols(const std::vector<LibraryEntry>& libs, const PCS& pcSet);
void MapToSymbols(const std::string&, size_t, const std::vector<size_t>&,
                  Symbols&);
bool MapSymbolsWithNM(const std::string&, size_t, const std::vector<size_t>&,
                      Symbols&, size_t* ptr = nullptr);
SymbolTable GetProcedureBoundaries(const std::string&, const std::string&,
                                   size_t* ptr = nullptr);
SymbolTable GetProcedureBoundariesViaNm(const std::string& cmd,
                                        const std::string& regex,
                                        size_t* ptr = nullptr);
std::string ShortFunctionName(const std::string&);
void FilterAndPrint(Profile& profile, Symbols& symbols,
                    const std::vector<LibraryEntry>& libs,
                    const std::vector<std::string>& threads);
void RemoveUninterestingFrames(const Symbols& symbols, Profile& profile);
void FilterFrames(const Symbols& symbols, Profile& profile);
std::string ExtractSymbolLocation(const Symbols&, size_t);
void GetTranslatedStacksAndReduce(
    Symbols& symbols, const Profile& profile,
    const std::unordered_map<std::string, std::string>& fullnameToShortnameMap,
    TranslatedStacks& stacks, FinalProfileMap& reduced);

template <typename T>
size_t TotalProfile(const std::unordered_map<T, size_t>& profile);
std::unordered_map<std::string, size_t> ExtractCalls(const Symbols& symbols,
                                                     const Profile& profile);
std::string ExtractSymbolLocation(const Symbols& symbols, size_t addr);
void FillFullnameToshortnameMap(
    const Symbols& symbols,
    std::unordered_map<std::string, std::string>& fullnameToShortnameMap);

void TranslateStack(
    Symbols& symbols,
    const std::unordered_map<std::string, std::string>& fullnameToShortnameMap,
    const std::vector<size_t>& addrs, std::vector<std::string>&,
    std::vector<boost::string_ref>&);
ShadowProfileMap CumulativeProfile(const FinalProfileMap& profile);
ShadowProfileMap FlatProfile(const FinalProfileMap& profile);
std::string Units();
std::string Unparse(size_t num);
std::string Percent(double num, double tot);
bool PrintDot(const std::string& prog, Symbols& symbols, Profile& raw,
              const ShadowProfileMap& flat, const ShadowProfileMap& cumulative,
              size_t overallTotal,
              const std::unordered_map<std::string, std::string>&,
              const TranslatedStacks&);

bool Addr2lineExist() {
  struct Addr2lineChecker {
    Addr2lineChecker() {
      enable = System(ShellEscape(kAddr2Line, "--help", ">/dev/null 2>&1"));
    }
    bool exists() { return enable; }
    bool enable = false;
  };

  static Addr2lineChecker checker;
  return checker.exists();
}

bool IsProfileURL(const std::string& fname) {
  bool exists = boost::filesystem::exists(fname);
  DLOG(INFO) << "File:" << fname << " exists:" << exists;

  return exists;
}

bool IsSymbolizedProfileFile(const std::string& fname) {
  if (!boost::filesystem::exists(fname)) return false;

  DLOG(INFO) << "Reading file:" << FLAGS_program;
  auto profileFile = MmapReadableFile::NewRandomAccessFile(FLAGS_program);

  if (profileFile->FullText().empty()) return false;

  return true;
  // const std::string symbolPage = "m,[^/]+$,";
}

void ConfigureObjTools() {}

std::string ConfigureTool(const std::string& tool) { return tool; }

boost::string_ref ReadProfileHeader(
    std::vector<boost::string_ref>::iterator& itr,
    std::vector<boost::string_ref>::iterator end) {
  uint8_t firstChar = itr->front();
  if (!std::isprint(firstChar)) return {};

  std::string line;
  for (; itr != end; ++itr) {
    if (boost::starts_with(*itr, "%warn")) {
      DLOG(INFO) << "WARNING:" << line;
    } else if (boost::starts_with(*itr, "%")) {
      DLOG(INFO) << "Ignoring unknown command from profile header:" << line;
    } else {
      DLOG(INFO) << "Get header line:" << *itr;
      return *itr++;
    }
  }
  LOG(FATAL) << "No header lines is found";
  return line;
}

void Init() {}

std::string FetchDynamicProfile(const std::string& binaryName,
                                const std::string& profileName,
                                bool fetchNameOnly, bool encouragePatience) {
  if (!IsProfileURL(profileName)) return profileName;
  // FIXME add network profile
  return {};
}

Context ReadProfile(const std::string& program, const std::string& profile) {
  Marker m(__func__);
  static const std::string contentionMarker = "contention";
  static const std::string growthMarker = "growth";
  static const std::string symbolMarker = "symbol";
  static const std::string profileMarker = "profile";
  static const std::string heapMarker = "heap";

  DLOG(INFO) << "Read program:" << program << ", profile:" << profile;
  // std::ifstream ifs(profile, std::ios::binary);
  auto profileFile = MmapReadableFile::NewRandomAccessFile(profile);
  auto profileLines = profileFile->SplitBy();
  if (profileLines.empty()) {
    std::cerr << "Empty files input" << std::endl;
    return {};
  }
  auto profileIterator = profileLines.begin();
  auto header = ReadProfileHeader(profileIterator, profileLines.end());
  if (boost::starts_with(header,
                         (boost::format("--- %s") % symbolMarker).str())) {
    DLOG(INFO) << "Meet symbol marker:" << header;
  }
  if (boost::starts_with(header,
                         (boost::format("--- %s") % heapMarker).str()) ||
      boost::starts_with(header,
                         (boost::format("--- %s") % growthMarker).str())) {
    DLOG(INFO) << "Meet heap marker or growther marker:" << header;
  }

  // std::string profileType = "";
  // TODO regex here

  if (boost::starts_with(header, "heap")) {
    gProfileType = "heap";
    return ReadThreadedHeapProfile(program, profile, header.to_string(),
                                   profileIterator, profileLines.end());
  }

  return {};
}

int HeapProfileIndex() { return 1; }

Context ReadThreadedHeapProfile(
    const std::string& program, const std::string& profileName,
    const std::string& header, std::vector<boost::string_ref>::iterator& itr,
    std::vector<boost::string_ref>::iterator itrEnd) {
  Marker m(__func__);
  DLOG(INFO) << "Now read threaded heap profile";
  const int index = HeapProfileIndex();
  int samplingAlgorithm = 0;
  int sampleAdjustment = 0;
  std::string type = "unknown";
  static const std::string kHeapV2 = "heap_v2/";

  static const boost::regex kHeapAdjustPattern(R"(^heap_v2/(\d+))");
  auto matchRes = RegexMatch(header, kHeapAdjustPattern);

  if (!matchRes.empty()) {
    type = "_v2";
    samplingAlgorithm = 2;
    sampleAdjustment = std::stoi(matchRes[1]);
  }
  LOG(INFO) << "samplingAlgorithm:" << samplingAlgorithm
            << ", sampleAdjustment:" << sampleAdjustment;

  if (type != "_v2") {
    LOG(FATAL)
        << "Threaded map profiles require v2 sampling with a sample rate";
  }

  auto profile = std::make_unique<Profile>();
  auto threadProfiles = std::make_unique<ThreadProfile>();
  auto pcs = std::make_unique<PCS>();
  std::vector<std::string> map;
  std::string stack;
  struct StackProfile {
    boost::string_ref stack_;
    std::vector<boost::string_ref> threadData_;
  };

  // TODO maybe we need to conbine servral stacks into one
  // to avoid too many jobs
  StackProfile currentProfile;

  auto ParseOneStack = [sampleAdjustment, samplingAlgorithm, index](
                           StackProfile stackProfile, Profile& profile,
                           ThreadProfile& threadProfiles, PCS& pcs) {
    auto fixedCallerStack = FixCallerAddresses(stackProfile.stack_);
    static const boost::regex kPattern2(
        R"(^\s*(t(\*|\d+)):\s+(\d+):\s+(\d+)\s+\[\s*(\d+):\s+(\d+)\]$)");
    for (auto line : stackProfile.threadData_) {
      DLOG(INFO) << "Now parsing profile line:" << line;
      auto matchRes2 = RegexMatch(line, kPattern2);
      if (matchRes2.empty()) continue;

      auto thread = matchRes2[2];
      // Skip per thread lines if target thread not specified
      if (thread != "*" && FLAGS_thread.empty()) continue;
      size_t n1 = boost::lexical_cast<size_t>(matchRes2[3].first,
                                              matchRes2[3].length());
      size_t s1 = boost::lexical_cast<size_t>(matchRes2[4].first,
                                              matchRes2[4].length());

      size_t n2 = boost::lexical_cast<size_t>(matchRes2[5].first,
                                              matchRes2[5].length());
      size_t s2 = boost::lexical_cast<size_t>(matchRes2[6].first,
                                              matchRes2[6].length());
      std::vector<size_t> counts =
          AdjustSamples(sampleAdjustment, samplingAlgorithm, n1, s1, n2, s2);
      if (thread == "*") {
        AddEntries(profile, pcs, fixedCallerStack, counts[index]);
      } else {
        AddEntries(threadProfiles.data_[thread], pcs, fixedCallerStack,
                   counts[index]);
      }
    }
  };
  std::deque<std::future<void>> parseFutures;

  while (itr != itrEnd) {
    boost::string_ref line = *itr++;
    if (line.empty()) continue;
    if (boost::starts_with(line, "MAPPED_LIBRARIES:")) {
      DLOG(INFO) << "Read mapped libraries:" << line;
      map = ReadMappedLibraries(itr, itrEnd);
      break;
    }

    if (boost::starts_with(line, "--- Memory map:")) {
      map = ReadMemoryMap(itr, itrEnd);
      break;
    }

    static const boost::regex kPattern1(R"(^\s*@\s+(.*)$)");
    auto matchRes1 = RegexMatch(line, kPattern1);
    if (!matchRes1.empty()) {
      // stack is empty for the first call stack entry
      if (!currentProfile.stack_.empty()) {
        auto fut = gThreadPool.submit(
            [currentProfile = std::move(currentProfile), &profile,
             &threadProfiles, &pcs, ParseOneStack]() {
              ParseOneStack(std::move(currentProfile), *profile,
                            *threadProfiles, *pcs);
            });
        parseFutures.emplace_back(std::move(fut));
      }
      currentProfile.stack_ = {matchRes1[1].first,
                               static_cast<size_t>(matchRes1[1].length())};
      continue;
    }

    // Still in the header part
    if (currentProfile.stack_.empty()) continue;
    currentProfile.threadData_.emplace_back(line);
  }
  auto fut = gThreadPool.submit([currentProfile = std::move(currentProfile),
                                 &profile, &threadProfiles, &pcs,
                                 ParseOneStack]() {
    ParseOneStack(std::move(currentProfile), *profile, *threadProfiles, *pcs);
  });
  parseFutures.emplace_back(std::move(fut));

  for (auto& f : parseFutures) {
    f.get();
  }
  auto parsedLibrariese = ParseLibraries(map, *pcs);
  DLOG(INFO) << "Parsed profile:\n"
             << std::hex << profile->data_ << std::endl
             << "Parsed threadprofile:\n"
             << std::hex << threadProfiles->data_ << std::endl
             << "Parsed libraries:\n"
             << std::hex << parsedLibrariese << std::endl
             << "Parsed pcset:\n"
             << std::hex << pcs->data_ << std::endl;

  Context context = {std::string("heap"),
                     1,
                     std::move(profile),
                     std::move(threadProfiles),
                     std::move(parsedLibrariese),
                     std::move(pcs),
                     Symbols{}};
  return context;
}

std::vector<size_t> FixCallerAddresses(boost::string_ref stack) {
  std::vector<boost::string_ref> addrs;
  boost::split(addrs, stack, isspace);

  std::vector<size_t> numAddrs;
  numAddrs.reserve(16);
  std::transform(std::begin(addrs), std::end(addrs),
                 std::back_inserter(numAddrs),
                 [](const auto& s) { return stoull(s, nullptr, 16); });

  for (auto it = std::next(numAddrs.begin()), itEnd = numAddrs.end();
       it != itEnd; ++it) {
    *it = AddressSub(*it, 0x1);
  }

  DLOG(INFO) << "Caller address from:" << stack << ", to:" << std::hex
             << numAddrs;
  return numAddrs;
}

size_t AddressSub(size_t x, size_t y) {
  if (x < y) {
    LOG(ERROR) << "Can not sub:" << x << " - " << y;
    return x;
  }
  return x - y;
}

// This is a thread safe method
void AddEntries(Profile& profile, PCS& pcs, const std::vector<size_t>& stack,
                size_t count) {
  DLOG(INFO) << "Add entry for stack:" << std::hex << stack
             << ", with count:" << std::dec << count;
  profile.AddEntry(stack, count);
  pcs.AddPC(stack);
  // AddEntry(profile, stack, count);
}

void AddEntry(Profile& profile, const std::vector<size_t>& stack,
              size_t count) {
  profile.data_[stack] += count;
}

std::vector<size_t> AdjustSamples(int sampleAdjustment, int samplingAlgorithm,
                                  size_t n1, size_t s1, size_t n2, size_t s2) {
  if (sampleAdjustment) {
    if (samplingAlgorithm == 2) {
      auto adjust = [](size_t& s, size_t& n, size_t adjust) {
        double ratio = ((s * 1.0) / n) / adjust;
        double scaleFactor = 1.0 / (1.0 - exp(-ratio));
        n *= scaleFactor;
        s *= scaleFactor;
      };
      if (n1) {
        adjust(s1, n1, sampleAdjustment);
      }
      if (n2) {
        adjust(s2, n2, sampleAdjustment);
      }
    } else {
      // Remote-heap version 1, FIXME
    }
  }
  return {n1, s1, n2, s2};
}

// TODO read files can use folly generator
// mapped libraries only have limitted numbers, so dont't use string_ref here
// for simplicity
std::vector<std::string> ReadMappedLibraries(
    std::vector<boost::string_ref>::iterator& itr,
    std::vector<boost::string_ref>::iterator itrEnd) {
  Marker m(__func__);

  std::vector<std::string> result;
  for (; itr != itrEnd; ++itr) {
    result.emplace_back(boost::erase_all_copy(itr->to_string(), "\r"));
  }

  DLOG(INFO) << "Get mapped libraries section:" << result;
  return result;
}

boost::smatch RegexMatch(const std::string& target, const boost::regex& re) {
  boost::smatch base;
  if (boost::regex_match(target, base, re)) return base;
  return boost::smatch{};
}

std::vector<LibraryEntry> ParseLibraries(const std::vector<std::string>& map,
                                         PCS& pcs) {
  Marker m(__func__);

  if (FLAGS_useSymbolPage) return {};
  // TODO abs path
  std::string buildVar;
  std::string& programName = FLAGS_program;
  size_t start, finish, offset;
  std::string lib;
  std::vector<LibraryEntry> result;
  static const boost::regex kBuildNumberPattern(R"(^\s*build=(.*)$)");
  for (const auto& line : map) {
    auto buildMatchRes = RegexMatch(line, kBuildNumberPattern);
    if (!buildMatchRes.empty()) {
      buildVar = buildMatchRes[1];
      DLOG(INFO) << "Build variable:" << buildVar;
    }

    bool match = false;
    do {
      // 4f000000-4f015000 r-xp 00000000 03:01 12845071   /lib/ld-2.3.2.so
      // TODO case insensitive
      static const boost::regex kPattern1(
          R"(^([[:xdigit:]]+)-([[:xdigit:]]+)\s+..x.\s+([[:xdigit:]]+)\s+\S+:\S+\s+\d+\s+(\S+\.(so|dll|dylib|bundle)((\.\d+)+\w*(\.\d+){0,3})?)$)");
      auto matchResult1 = RegexMatch(line, kPattern1);
      if (!matchResult1.empty()) {
        start = std::stoull(matchResult1[1], nullptr, 16);
        finish = std::stoull(matchResult1[2], nullptr, 16);
        offset = std::stoull(matchResult1[3], nullptr, 16);
        lib = matchResult1[4];  // window style path

        DLOG(INFO) << "matched case 1:" << line;
        match = true;
        break;
      }

      // 4e000000-4e015000: /lib/ld-2.3.2.so
      static const boost::regex kPattern2(
          R"(\s*([[:xdigit:]]+)-([[:xdigit:]]+):\s*(\S+\.so(\.\d+)*)$)");
      auto matchResult2 = RegexMatch(line, kPattern2);

      if (!matchResult2.empty()) {
        start = std::stoull(matchResult2[1], nullptr, 16);
        finish = std::stoull(matchResult2[2], nullptr, 16);
        offset = 0;
        lib = matchResult2[3];

        DLOG(INFO) << "matched case 2:" << line;
        match = true;
        break;
      }

      // 00400000-00404000 r-xp 00000000 08:01 799604 /usr/bin/w.procps
      // TODO case insensitive
      static const boost::regex kPattern3(
          R"(^([[:xdigit:]]+)-([[:xdigit:]]+)\s+..x.\s+([[:xdigit:]]+)\s+\S+:\S+\s+\d+\s+(\S+)$)");
      auto matchResult3 = RegexMatch(line, kPattern3);
      // TODO soft link name deal
      // /usr/bin/w /usr/bin/w.procps
      if (!matchResult3.empty() && matchResult3[4] == programName) {
        start = std::stoull(matchResult3[1], nullptr, 16);
        finish = std::stoull(matchResult3[2], nullptr, 16);
        offset = std::stoull(matchResult3[3], nullptr, 16);
        lib = matchResult3[4];

        DLOG(INFO) << "matched case 3:" << line;
        match = true;
        break;
      }

      // For FreeBSD 10.0
      // 0x800600000 0x80061a000 26 0 0xfffff800035a0000 r-x 75 33 0x1004 COW
      // NC vnode /libexec/ld-elf.so.1
      static const boost::regex kPattern4(
          R"(^(0x[[:xdigit:]]+)\s(0x[[:xdigit:]]+)\s\d+\s\d+\s0x[[:xdigit:]]+\sr-x\s\d+\s\d+\s0x\d+\s(COW|NCO)\s(NC|NNC)\svnode\s(\S+\.so(\.\d+)*)$)");
      auto matchResult4 = RegexMatch(line, kPattern4);
      if (!matchResult4.empty()) {
        start = std::stoull(matchResult4[1], nullptr, 16);
        finish = std::stoull(matchResult4[2], nullptr, 16);
        offset = 0;
        // lib = FindLibrary(matchResult[5]);
        DLOG(INFO) << "matched case 4:" << line;
        match = true;
        break;
      }
    } while (0);

    if (!match) {
      DLOG(INFO) << "line don't match any case:" << line;
      continue;
    }
    // Expand "$build" variable if avalable;
    lib = boost::regex_replace(lib, boost::regex(R"(\$build\b)"), buildVar);
    lib = FindLibrary(lib);
    std::string debuggingLib = DebuggingLibrary(lib);
    if (debuggingLib.empty()) {
      auto text = ParseTextSectinoHeader(lib);
      if (std::get<0>(text)) {
        auto vmaOffset = AddressSub(std::get<1>(text), std::get<2>(text));
        offset = AddressAdd(offset, vmaOffset);
      }
    }

    DLOG(INFO) << "Add parsed library line, lib:" << lib << ",start:" << start
               << ", finish:" << finish << ",offset:" << offset;
    result.push_back({lib, start, finish, offset});
  }
  // Append special entry for additional library (not relocated)
  // FIXME
  size_t minPC = 0, maxPC = 0;
  maxPC = *std::max_element(std::begin(pcs.data_), std::end(pcs.data_));

  DLOG(INFO) << "Add parsed library line, lib:" << programName
             << ",start:" << minPC << ", finish:" << maxPC << ",offset:" << 0;

  result.push_back({programName, minPC, maxPC, 0ul});
  return result;
}

std::tuple<size_t, size_t, size_t> ParseTextSectinoHeader(
    const std::string& lib) {
  // FIXME otool
  return ParseTextSectinoHeaderFromObjdump(lib);
}

// TODO Deal with command line in stream mode, don't wait for full result
std::vector<std::string> ExecuteCommand(const std::string& cmd) {
  std::string log = "execute command|";
  constexpr size_t kCommandMaxLogLen = 100;
  if (cmd.size() > kCommandMaxLogLen) {
    log += cmd.substr(0, kCommandMaxLogLen) + "...";
  } else {
    log += cmd;
  }

  Marker m(log.data());
  std::array<char, 128> buffer;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.data(), "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  std::string result;
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  DLOG(INFO) << "Execute command:" << cmd << ",Get result:\n" << result;
  boost::algorithm::erase_all(result, "\r");
  std::vector<std::string> resultSet;
  boost::split(resultSet, result, [](char c) { return c == '\n'; });

  return resultSet;
}

// TODO This functino can use stream like method
// we don't need the whole file, just the header
std::tuple<size_t, size_t, size_t> ParseTextSectinoHeaderFromObjdump(
    const std::string& lib) {
  auto cmd = ShellEscape("objdump", "-h", lib);
  auto resultSet = ExecuteCommand(cmd);

  size_t size = 0;
  size_t vma = 0;
  size_t fileOffset = 0;
  auto RemoveContinusSpaces = [](std::string& s) {
    auto it = std::unique(s.begin(), s.end(), [](char lhs, char rhs) {
      return lhs == rhs && isspace(lhs);
    });
    s.erase(it, s.end());
  };
  for (auto line : resultSet) {
    RemoveContinusSpaces(line);
    boost::trim(line);
    std::vector<std::string> splitLine;
    boost::split(splitLine, line, isspace);

    if (splitLine.size() >= 6 && splitLine[1] == ".text") {
      size = std::stoull(splitLine[2], nullptr, 16);
      vma = std::stoull(splitLine[3], nullptr, 16);
      fileOffset = std::stoull(splitLine[5], nullptr, 16);
      break;
    }
  }
  DLOG(INFO) << "Text section header for lib:" << lib << ", size:" << size
             << ",vma:" << vma << ", fileOffset:" << fileOffset;
  return std::make_tuple(size, vma, fileOffset);
}

std::string DebuggingLibrary(const std::string& file) {
  // Carefull for multhread
  // static std::map<std::string, std::string> cache;
  // auto it = cache.find(file);
  // if (it != cache.end()) {
  //  return it->second;
  //}

  if (boost::starts_with(file, "/")) {
    std::array<std::string, 2> debugFiles = {
        (boost::format("/usr/lib/debug%s") % file).str(),
        (boost::format("/usr/lib/debug%s.debug") % file).str()};
    for (const auto& f : debugFiles) {
      if (boost::filesystem::exists(f)) {
        DLOG(INFO) << "Find debugging lib:" << f << ", for file:" << file;
        // cache.emplace(file, f);
        return f;
      }
    }
  }
  DLOG(INFO) << "Haven't find debugging lib for file:" << file;
  // cache.emplace(file, std::string{});
  return {};
}

// Aggressively search the lib_prefix values for the given library
// If all else fails, just return the name of the library unmodified.
// If the lib_prefix is "/my/path,/other/path" and $file is
// "/lib/dir/mylib.so" it will search the following locations in this order,
// until it finds a file:
//   /my/path/lib/dir/mylib.so
//   /other/path/lib/dir/mylib.so
//   /my/path/dir/mylib.so
//   /other/path/dir/mylib.so
//   /my/path/mylib.so
//   /other/path/mylib.so
//   /lib/dir/mylib.so              (returned as last resort)
std::string FindLibrary(const std::string& lib) {
  std::string suffix = lib;
  // FIXME prefix list
  return lib;
}

std::vector<std::string> ReadMemoryMap(
    std::vector<boost::string_ref>::iterator& itr,
    std::vector<boost::string_ref>::iterator itrEnd) {
  std::vector<std::string> result;
  std::string buildVar;
  std::string line;

  static const boost::regex kBuildNumberPattern(R"(^\s*build=(.*))");
  for (; itr != itrEnd; ++itr) {
    std::string line = itr->to_string();
    auto matchRes = RegexMatch(line, kBuildNumberPattern);
    if (!matchRes.empty()) buildVar = matchRes[1];

    boost::regex_replace(line, boost::regex(R"(\$build\b)"), buildVar);
    result.emplace_back(line);
  }
  return result;
}

size_t AddressAdd(size_t x, size_t y) { return x + y; }

void MergeSymbols(Symbols& lhs, const Symbols& rhs) {
  lhs.data_.insert(std::begin(rhs.data_), std::end(rhs.data_));
}

void MergeSymbols(Symbols& lhs, Symbols&& rhs) {
  lhs.data_.insert(std::make_move_iterator(std::begin(rhs.data_)),
                   std::make_move_iterator(std::end(rhs.data_)));
}

Symbols MapToSymbols2(std::string image, size_t offset,
                      std::vector<size_t> pcList) {
  Symbols res;
  MapToSymbols(image, offset, pcList, res);
  return res;
}

// TODO: smallvector smallstring replace
// TODO: think about estimate pruning (don't need all symbols,
// only care about the hot point)
Symbols ExtractSymbols(const std::vector<LibraryEntry>& libs,
                       const PCS& pcSet) {
  Marker m(__func__);

  Symbols symbols;
  auto sortedLibs = libs;
  // consider use reference_wrapper
  std::sort(
      std::begin(sortedLibs), std::end(sortedLibs),
      [](const auto& lhs, const auto& rhs) { return rhs.start_ < lhs.start_; });

  std::vector<size_t> pcs(pcSet.data_.begin(), pcSet.data_.end());
  std::sort(pcs.begin(), pcs.end());
  std::vector<std::future<Symbols>> symbolsFutures;

  for (const auto& entry : sortedLibs) {
    auto libName = entry.lib_;
    const auto debugLib = DebuggingLibrary(libName);
    if (!debugLib.empty()) libName = debugLib;

    auto finishPCIndex =
        std::upper_bound(pcs.begin(), pcs.end(), entry.finish_);
    auto startPCIndex =
        std::lower_bound(pcs.begin(), finishPCIndex, entry.start_);

    std::vector<size_t> contained{startPCIndex, finishPCIndex};
    pcs.erase(startPCIndex, finishPCIndex);

    DLOG(INFO) << "Start to extract symbols for lib:" << libName
               << ", get contained pc set:" << std::hex << contained;
    auto addr = AddressSub(entry.start_, entry.offset_);
    auto future = gThreadPool.submit([libName = std::move(libName), addr,
                                      contained = std::move(contained)]() {
      return MapToSymbols2(libName, addr, contained);
    });
    symbolsFutures.emplace_back(std::move(future));
  }
  for (auto& f : symbolsFutures) {
    MergeSymbols(symbols, f.get());
  }

  return symbols;
}

void MapToSymbols(const std::string& image, size_t offset,
                  const std::vector<size_t>& pcList, Symbols& symbols) {
  Marker m(__func__);
  if (pcList.empty()) return;

  if (!Addr2lineExist()) {
    DLOG(INFO) << "addr2line is not installed on system, use nm";
    MapSymbolsWithNM(image, offset, pcList, symbols);
    return;
  }

  Symbols nmSymbols;
  // TODO replace type with boost::any
  size_t sepAddress = std::numeric_limits<size_t>::max();
  MapSymbolsWithNM(image, offset, pcList, nmSymbols, &sepAddress);
  std::string cmd = ShellEscape(kAddr2Line, "-f", "-C", "-e", image);
  if (IsValidSepAddress(sepAddress)) {
    // TODO Don't call this function repeat, check option -i can be done only
    // once
    auto fullCmd = (boost::format("%s -i --help >/dev/null 2>&1") % cmd).str();
    if (System(fullCmd)) {
      DLOG(INFO) << "addr2line support '-i' options check pass";
      cmd += " -i";
    } else {
      sepAddress = std::numeric_limits<size_t>::max();
    }
  }
  const size_t kArgumentListMax = FLAGS_addr2line_arguments;
  struct Argument {
    size_t startPcIndex = 0;
    std::vector<std::string> addrs;
  };

  auto toHexStr = [](size_t x) { return (boost::format("%016x") % x).str(); };
  std::vector<Argument> arguments;

  Argument currentArgument;
  for (size_t i = 0, sz = pcList.size(); i < sz; ++i) {
    if (i && (i % kArgumentListMax == 0)) {
      arguments.emplace_back(std::move(currentArgument));
      currentArgument.startPcIndex = i;
    }
    currentArgument.addrs.emplace_back(toHexStr(AddressSub(pcList[i], offset)));
    if (IsValidSepAddress(sepAddress)) {
      currentArgument.addrs.emplace_back(toHexStr(sepAddress));
    }
  }
  arguments.emplace_back(std::move(currentArgument));
  DLOG(INFO) << "sepaddress is:" << IsValidSepAddress(sepAddress);

  auto ExtractPartSymbols = [&](const Argument& argument) {
    Symbols partSymbols;
    // TODO review the original source the command end with |
    std::string argumentStr = boost::join(argument.addrs, " ");
    const std::string cmdWithAddresses =
        (boost::format("%s %s") % cmd % argumentStr).str();

    auto resultSet = ExecuteCommand(cmdWithAddresses);
    DLOG(INFO) << "Total lines for command result:" << resultSet.size()
               << ", result:" << resultSet;

    size_t count = argument.startPcIndex;
    for (size_t i = 0, resultSize = resultSet.size(); i < resultSize;) {
      std::string fullFunction = resultSet[i++];

      if (i >= resultSize) {
        LOG(INFO) << "WARN: this result size is not even:" << resultSize
                  << ",last line:" << fullFunction;
        break;
      }
      std::string fileLineNum = resultSet[i++];

      if (IsValidSepAddress(sepAddress) && (fullFunction == kSepSymbol)) {
        DLOG(INFO) << "Meet sepsymbol for pcstr:" << std::hex << pcList[count];
        ++count;
        continue;
      }
      DLOG(INFO) << "Now deal with fullfunction:" << fullFunction
                 << ", fileLineNum" << fileLineNum;
      // Turn windows-style path into unix-style
      // boost::replace_all(fileLineNum, R"(\\)", "/");
      auto pc = pcList[count];
      std::string function = ShortFunctionName(fullFunction);
      auto itr = nmSymbols.data_.find(pc);
      if (itr != nmSymbols.data_.end()) {
        if (fullFunction == "??") {
          function = itr->second[0];
          fullFunction = itr->second[2];
        } else {
          // TODO don't understand this line
          // /^\Q$function\E
          if (boost::starts_with(itr->second[2], function)) {
            function = itr->second[0];
            fullFunction = itr->second[2];
          }
        }
        std::vector<std::string>& sym = partSymbols.data_[pc];
        sym.insert(sym.begin(), {std::move(function), std::move(fileLineNum),
                                 std::move(fullFunction)});
        DLOG(INFO) << (boost::format("cur symbol line:%016x => %s") % pc %
                       boost::join(sym, " "))
                          .str();
        if (!IsValidSepAddress(sepAddress)) ++count;
      }
    }
    return partSymbols;
  };
  std::vector<std::future<Symbols>> futures;
  for (auto& a : arguments) {
    auto fut = gSubThreadPool.submit([a = std::move(a), &ExtractPartSymbols]() {
      return ExtractPartSymbols(a);
    });
    futures.emplace_back(std::move(fut));
  }
  auto MergePartSymbols = [](Symbols& sum, Symbols&& part) {
    for (auto& p : part.data_) {
      auto& sym = sum.data_[p.first];
      auto& symPart = p.second;
      sym.insert(sym.begin(), std::make_move_iterator(symPart.begin()),
                 std::make_move_iterator(symPart.end()));
    }
  };
  for (auto& f : futures) {
    MergePartSymbols(symbols, f.get());
  }
  DLOG(INFO) << "Now the symbol is:" << std::hex << symbols.data_;
}

bool MapSymbolsWithNM(const std::string& image, size_t offset,
                      const std::vector<size_t>& pcList, Symbols& symbols,
                      size_t* sepAddress) {
  Marker m(__func__);

  DLOG(INFO) << "Start map symbols for image:" << image
             << ", with offset:" << std::hex << offset;
  auto symbolTable = GetProcedureBoundaries(image, ".", sepAddress);
  DLOG(INFO) << "Get symbol table:" << symbolTable.data_;
  if (symbolTable.data_.empty()) return false;

  // names sorted by value in symbol table
  std::vector<std::string> names;
  std::transform(symbolTable.data_.begin(), symbolTable.data_.end(),
                 std::back_inserter(names),
                 [](const auto& v) { return v.first; });
  // No symbols, use address
  if (names.empty()) {
    // TODO review the perl source, seems buggy
    return false;
  }
  std::sort(names.begin(), names.end(),
            [&symbolTable](const auto& lhs, const auto& rhs) {
              return symbolTable.data_[lhs][0] < symbolTable.data_[rhs][0];
            });
  DLOG(INFO) << "Sorted names with value in symboltable:" << names;
  size_t index = 0;
  auto fullName = names[0];
  auto name = ShortFunctionName(fullName);
  auto nameNum = names.size();

  const auto& sortedList = pcList;
  DCHECK(std::is_sorted(pcList.begin(), pcList.end())) << "Need to be sorted";

  for (auto pc : sortedList) {
    auto mpc = AddressSub(pc, offset);
    while (index < nameNum - 1 && mpc >= symbolTable.data_[fullName][1]) {
      fullName = names[++index];
      name = ShortFunctionName(fullName);
    }
    if (mpc < symbolTable.data_[fullName][1]) {
      symbols.data_.emplace(pc, std::vector<std::string>{name, "?", fullName});
    } else {
      std::string pcStr = (boost::format("0x%016x") % pc).str();
      symbols.data_.emplace(pc, std::vector<std::string>{pcStr, "?", pcStr});
    }
  }
  return true;
}

// This function seems very slow, need to improve
// TODO replace with tbb concurrent hashmap
std::string ShortFunctionName(const std::string& fullName) {
  class SafeCache {
   public:
    std::string Get(const std::string& k) {
      std::lock_guard<std::mutex> l(mutex_);

      auto itr = cache_.find(k);
      if (itr != cache_.end()) return itr->second;

      return {};
    }
    void Set(const std::string& k, const std::string& v) {
      std::lock_guard<std::mutex> l(mutex_);

      auto itr = cache_.find(k);
      if (itr != cache_.end()) return;

      cache_.emplace(k, v);
    }

   private:
    std::unordered_map<std::string, std::string> cache_;
    std::mutex mutex_;
  };

  static SafeCache cache;
  std::string val = cache.Get(fullName);
  if (!val.empty()) return val;

  auto name = fullName;
  auto replace = [](const std::string& s, const std::string& re,
                    const std::string& fmt) {
    return boost::regex_replace(s, boost::regex(re), fmt);
  };
  while (true) {
    auto r = replace(name, R"(\([^()]*\)(\s*const)?)", "");
    if (r == name) break;
    name = r;
  }
  while (true) {
    auto r = replace(name, R"(<[^<>]*>)", "");
    if (r == name) break;
    name = r;
  }

  name = replace(name, R"(^.*\s+(\w+::))", "$&");
  cache.Set(fullName, name);

  return name;
}

SymbolTable GetProcedureBoundaries(const std::string& image,
                                   const std::string& regex,
                                   size_t* sepAddress) {
  Marker m(__func__);

  if (image.find_first_of("/.") != 0) {
    DLOG(ERROR) << "Error file name, not start with . or /:" << image;
    return {};
  }

  std::string imageName = image;
  // TODO, seems this call for DebuggingLibrary is already done
  // in the caller, think about remove this line
  const auto debugging = DebuggingLibrary(image);
  if (!debugging.empty()) imageName = debugging;

  std::string demangleFlag, cppfiltFlag;
  std::string toDevNull = ">/dev/null 2>&1";

  // This line seems a bug in the perl source code, image -> $image
  // TODO don't test these function, move the different flag into nmCommands
  if (System(ShellEscape(kNm, "--demangle", image) + toDevNull)) {
    demangleFlag = "--demangle";
    cppfiltFlag = "";
  } else if (System(ShellEscape(kCppFilt, image) + toDevNull)) {
    cppfiltFlag = " | " + ShellEscape(kCppFilt);
  }

  std::string flattenFlag;
  if (System(ShellEscape(kNm, "-f", image) + toDevNull)) {
    flattenFlag = "-f";
  }

  const std::string tail =
      (boost::format(" 2>/dev/null %s") % cppfiltFlag).str();
  // TODO the 6nm binary command is omitted here
  std::vector<std::string> nmCommands = {
      ShellEscape(kNm, "-n", flattenFlag, demangleFlag, image) + tail,
      ShellEscape(kNm, "-D", "-n", flattenFlag, demangleFlag, image) + tail};

  if (false) {  // nm_pdb related for windows
  }
  for (const auto& c : nmCommands) {
    auto symbolTable = GetProcedureBoundariesViaNm(c, regex, sepAddress);
    if (!symbolTable.data_.empty()) return symbolTable;
  }

  return {};
}

SymbolTable GetProcedureBoundariesViaNm(const std::string& cmd,
                                        const std::string& regex,
                                        size_t* sepAddress) {
  Marker m(__func__);

  SymbolTable table;
  auto resultSet = ExecuteCommand(cmd);

  auto CheckAddSymbol = [&table, &regex](const std::string& name,
                                         const auto& start, const auto& last) {
    if (name.empty()) return;
    if (regex.empty() || regex == "." ||
        !RegexMatch(name, boost::regex(regex)).empty()) {
      size_t startVal = std::stoull(start, nullptr, 16);
      size_t lastVal = std::stoull(last, nullptr, 16);

      DLOG(INFO) << "Add line into symbol table, name:" << name << std::hex
                 << ", start:" << startVal << ",last:" << lastVal;
      table.data_.emplace(name, std::vector<size_t>{startVal, lastVal});
    }
  };
  std::string lastStart = "0";
  std::string routine;
  for (const auto& line : resultSet) {
    static const boost::regex kSymbolPattern(R"(^\s*([0-9a-f]+) (.) (..*))");
    auto matchRes = RegexMatch(line, kSymbolPattern);
    if (!matchRes.empty()) {
      // DLOG(INFO) << "Line matched:" << line;
      auto startVal = matchRes[1];
      char type = (matchRes[2].str())[0];
      std::string thisRoutine = matchRes[3];

      if (startVal == lastStart && (type == 't' || type == 'T')) {
        routine = thisRoutine;
        continue;
      } else if (startVal == lastStart) {
        continue;
      }

      if (thisRoutine == kSepSymbol) {
        if (sepAddress) *sepAddress = std::stoull(startVal, nullptr, 16);
      }
      // TODO consider not append into name, since
      // in ShoftFunctionName we will remove it
      thisRoutine += (boost::format("<%016x>") % startVal).str();
      CheckAddSymbol(routine, lastStart, startVal);
      lastStart = startVal;
      routine = thisRoutine;
    } else if (boost::starts_with(line, "Load image name:")) {
      // For windows;
      DLOG(INFO) << "Use Image:";
    } else if (boost::starts_with(line, "PDB file name:")) {
      // For windows;
      DLOG(INFO) << "Use PDB:";
    }
  }
  // Deal with last routine
  CheckAddSymbol(routine, lastStart, lastStart);
  return table;
}

bool System(const std::string& cmd) {
  auto log = std::string(__func__ + std::string("|execute command:") + cmd);
  Marker m(log.data());

  return system(cmd.data()) == 0;
}

const std::string kSkipFunctions[] = {
    "calloc", "cfree", "malloc", "newImpl", "void* newImpl", "free", "memalign",
    "posix_memalign", "aligned_alloc", "pvalloc", "valloc", "realloc",
    "mallocx", "rallocx", "xallocx", "dallocx", "sdallocx", "tc_calloc",
    "tc_cfree", "tc_malloc", "tc_free", "tc_memalign", "tc_posix_memalign",
    "tc_pvalloc", "tc_valloc", "tc_realloc", "tc_new", "tc_delete",
    "tc_newarray", "tc_deletearray", "tc_new_nothrow", "tc_newarray_nothrow",
    "do_malloc",
    "::do_malloc",  // new name -- got moved to an unnamed ns
    "::do_malloc_or_cpp_alloc", "DoSampledAllocation", "simple_alloc::allocate",
    "__malloc_alloc_template::allocate", "__builtin_delete", "__builtin_new",
    "__builtin_vec_delete", "__builtin_vec_new", "operator new",
    "operator new[]",
    // The entry to our memory-allocation routines on OS X
    "malloc_zone_malloc", "malloc_zone_calloc", "malloc_zone_valloc",
    "malloc_zone_realloc", "malloc_zone_memalign", "malloc_zone_free",
    // These mark the beginning/end of our custom sections
    "__start_google_malloc", "__stop_google_malloc", "__start_malloc_hook",
    "__stop_malloc_hook"};

void RemoveUninterestingFrames(const Symbols& symbols, Profile& profile) {
  Marker m(__func__);
  ProfilerStart(__func__);

  const std::string skipRegexPattern = "NOMATCH";
  std::unordered_set<std::string> skip;
  if (gProfileType == "heap" || gProfileType == "growth") {
    for (const auto& f : kSkipFunctions) {
      skip.insert(f);
      skip.insert("_" + f);
    }
    // TODO add tcmalloc related logic
  }
  if (gProfileType == "cup") {
    // skiped logic
  }
  ProfileMap result;
  std::vector<size_t> path;
  for (const auto& p : profile.data_) {
    for (size_t addr : p.first) {
      auto it = symbols.data_.find(addr);
      if (it != symbols.data_.end()) {
        const auto& func = it->second[0];
        if (skip.count(func)) {  // TODO skip_regexp
          path.clear();
          continue;  // The perl logci seems strange
        }
      }
      path.emplace_back(addr);
    }
    size_t count = p.second;
    result[std::move(path)] += count;
  }
  DLOG(INFO) << "Result after remove uninteresting frames:" << std::hex
             << result;
  // TODO add filter
  // FilterFrames(symbols, result);
  profile.data_ = std::move(result);
  ProfilerFlush();
}

void FilterFrames(const Symbols& symbols, Profile& profile) {
  // if opt_retain opt_exclude
  return;
  // TODO add remain logic
}

void FilterAndPrint(Profile& profile, Symbols& symbols,
                    const std::vector<LibraryEntry>& libs,
                    const std::vector<std::string>& threads) {
  Marker m(__func__);
  std::unordered_map<std::string, std::string> fullnameToShortnameMap;
  FillFullnameToshortnameMap(symbols, fullnameToShortnameMap);
  DLOG(INFO) << "Get fullname to shortname map:" << fullnameToShortnameMap;

  auto total = TotalProfile(profile.data_);
  DLOG(INFO) << "Total Profile:" << std::hex << total;
  RemoveUninterestingFrames(symbols, profile);
  DLOG(INFO) << "After remove unteresting frames:" << std::hex << profile.data_;
  TranslatedStacks translatedStacks;
  FinalProfileMap reduced;
  GetTranslatedStacksAndReduce(symbols, profile, fullnameToShortnameMap,
                               translatedStacks, reduced);
  // auto calls = ExtractCalls(symbols, profile);
  // DLOG(INFO) << "Extracted calls:" << std::hex << calls;
  // reduced's life time depend on translatedStacks
  DLOG(INFO) << "The reduced profile:" << std::hex << reduced;
  // flat and cumulative depends on reduced profile's life time
  auto flat = FlatProfile(reduced);
  DLOG(INFO) << "The flat profile:" << flat;
  auto cumulative = CumulativeProfile(reduced);
  DLOG(INFO) << "The cumulative profile:" << cumulative;

  PrintDot(FLAGS_program, symbols, profile, flat, cumulative, total,
           fullnameToShortnameMap, translatedStacks);
}

size_t ShadowProfileMapGet(const ShadowProfileMap& map, boost::string_ref s) {
  auto it = map.find(s);
  if (it == map.end()) return 0ull;

  return it->second;
}

bool PrintDot(
    const std::string& prog, Symbols& symbols, Profile& raw,
    const ShadowProfileMap& flat, const ShadowProfileMap& cumulative,
    size_t overallTotal,
    const std::unordered_map<std::string, std::string>& fullnameToShortnameMap,
    const TranslatedStacks& translatedStacks) {
  Marker m(__func__);
  ProfilerStart(__func__);

  // TODO we may calculated the focused nodes at the very beginning
  // then we can reduce most of the symbol fetching operation
  // since only about 100 nodes are drawed by default
  // prunning early for a large program and profile is needed
  auto localTotal = TotalProfile(flat);
  size_t nodeLimit = FLAGS_nodeFraction * localTotal;
  size_t edgeLimit = FLAGS_edgeFraction * localTotal;
  size_t nodeCount = FLAGS_nodeCount;

  std::vector<boost::string_ref> list;
  std::transform(cumulative.begin(), cumulative.end(), std::back_inserter(list),
                 [](const auto& v) { return v.first; });
  std::sort(list.begin(), list.end(),
            [&cumulative](const auto& lhs, const auto& rhs) {
              auto lv = ShadowProfileMapGet(cumulative, lhs);
              auto rv = ShadowProfileMapGet(cumulative, rhs);
              return lv != rv ? rv < lv : lhs < rhs;
            });

  auto last = std::min(nodeCount - 1, list.size() - 1);
  while (last >= 0 && ShadowProfileMapGet(cumulative, list[last]) <= nodeLimit)
    --last;

  if (last < 0) {
    std::cerr << "No nodes to print" << std::endl;
    return false;
  }

  if (nodeLimit > 0 || edgeLimit > 0) {
    // TODO complement this line
    std::cerr << "Dropping nodes with:" << std::endl;
  }

  // TODO write into dot
  std::string output = "| dot -Tps2 | ps2pdf - -";
  std::string dotFile = FLAGS_dotFileName;
  std::ofstream ofs(dotFile);
  ofs << boost::format("digraph \"%s; %s %s\" {\n") % prog %
             Unparse(overallTotal) % Units();

  if (FLAGS_pdf) {
    ofs << "size=\"8,11\"\n";
  }
  ofs << "node [width=0.375,height=0.25];\n";
  ofs << boost::format(
             "Legend [shape=box,fontsize=24,shape=plaintext,"
             "label=\"%s\\l%s\\l%s\\l%s\\l%s\\l\"];\n") %
             prog %
             (boost::format("Total %s: %s") % Units() % Unparse(overallTotal))
                 .str() %
             (boost::format("Focusing on: %s") % Unparse(localTotal)).str() %
             (boost::format("Dropped nodes with <= %s abs(%s)") %
              Unparse(nodeLimit) % Units())
                 .str() %
             (boost::format("Dropped edges with <= %s %s") %
              Unparse(edgeLimit) % Units())
                 .str();

  std::unordered_map<boost::string_ref, size_t> node;
  size_t nextNode = 1;
  for (auto it = list.begin(), itEnd = std::next(list.begin(), last + 1);
       it != itEnd; ++it) {
    const auto& a = *it;
    auto f = ShadowProfileMapGet(flat, a);
    auto c = ShadowProfileMapGet(cumulative, a);

    double fs = 8;
    if (localTotal > 0) {
      fs = 8 + (50.0 * sqrt(fabs(f * 1.0 / localTotal)));
    }

    node[a] = nextNode++;
    std::string sym = a.to_string();

    boost::replace_all(sym, " ", "\\n");
    boost::replace_all(sym, "::", "\\n");

    std::string extra;
    if (f != c) {
      extra =
          (boost::format("\\rof %s (%s)") % Unparse(c) % Percent(c, localTotal))
              .str();
    }
    std::string style;
    if (FLAGS_heapCheck) {
      DLOG(FATAL) << "Not implement";
    }
    ofs << boost::format(
               "N%d [label=\"%s\\n%s (%s)%s\\r"
               "\",shape=box,fontsize=%.1f%s];\n") %
               node[a] % sym % Unparse(f) % Percent(f, localTotal) % extra %
               fs % style;
  }
  using Edge = std::map<std::array<std::string, 2>, size_t>;
  Edge edge;
  for (const auto& stack : translatedStacks) {
    size_t n = stack.second;
    const auto& translated = stack.first;
    for (int i = 1, sz = static_cast<int>(translated.size()); i < sz; ++i) {
      auto& src = translated[i];
      auto& dst = translated[i - 1];

      if (node.count(src) && node.count(dst)) {
        edge[std::array<std::string, 2>{src, dst}] += n;
      }
    }
  }

  std::vector<std::reference_wrapper<Edge::value_type>> edgeList(edge.begin(),
                                                                 edge.end());
  // b compare to a
  std::sort(edgeList.begin(), edgeList.end(),
            [](const auto& lhs, const auto& rhs) {
              const auto& lv = lhs.get().second;
              const auto& rv = rhs.get().second;
              return lv != rv ? rv < lv : lhs.get() < rhs.get();
            });

  std::unordered_map<std::string, size_t> outDegree, inDegree;
  for (const auto& p : edgeList) {
    const auto& src = p.get().first[0];
    const auto& dst = p.get().first[1];
    size_t n = p.get().second;

    bool keep = false;
    if (inDegree[dst] == 0) {
      keep = true;
    } else if (static_cast<size_t>(abs(n)) <= edgeLimit) {
      keep = false;
    } else if (outDegree[src] >= static_cast<size_t>(FLAGS_maxDegree) ||
               inDegree[dst] >= static_cast<size_t>(FLAGS_maxDegree)) {
      keep = false;
    } else {
      keep = true;
    }
    if (!keep) continue;

    ++outDegree[src];
    ++inDegree[dst];

    double fraction =
        std::min(fabs(localTotal ? (3.0 * (1.0 * n / localTotal)) : 0), 1.0);
    double w = fraction * 2;
    if (w < 1 && (FLAGS_web || FLAGS_svg)) {
      w = 1;
    }
    int edgeWeight = static_cast<int>(std::min(pow(fabs(n), 0.7), 100000.0));
    std::string style = (boost::format("setlinewidth(%f)") % w).str();

    if (boost::algorithm::contains(dst, "(inline)")) {
      style += ",dashed";
    }
    ofs << boost::format("N%s -> N%s [label=%s, weight=%d, style=\"%s\"];\n") %
               node[src] % node[dst] % Unparse(n) % edgeWeight % style;
  }
  ofs << "}\n";
  ProfilerFlush();
  return true;
}

std::string Percent(double num, double tot) {
  if (tot != 0) {
    return (boost::format("%.1f%%") % (num * 100.0 / tot)).str();
  }
  return num == 0 ? "nan" : ((num > 0) ? "+inf" : "-inf");
}

std::string Unparse(size_t num) {
  if (FLAGS_showBytes) {
    return std::to_string(num);
  }
  return std::to_string(1.0 * num / (1024 * 1024));
}

std::string Units() {
  if (FLAGS_showBytes) {
    return "B";
  }
  return "MB";
}

ShadowProfileMap FlatProfile(const FinalProfileMap& profile) {
  ShadowProfileMap result;
  Marker m(__func__);

  for (const auto& p : profile) result[p.first.front()] += p.second;
  return result;
}

ShadowProfileMap CumulativeProfile(const FinalProfileMap& profile) {
  Marker m(__func__);
  ProfilerStart(__func__);

  ShadowProfileMap result;
  for (const auto& p : profile) {
    for (const auto& a : p.first) {
      result[a] += p.second;
    }
  }
  ProfilerFlush();
  return result;
}

void GetTranslatedStacksAndReduce(
    Symbols& symbols, const Profile& profile,
    const std::unordered_map<std::string, std::string>& fullnameToShortnameMap,
    TranslatedStacks& stacks, FinalProfileMap& reduced) {
  Marker m(__func__);
  ProfilerStart(__func__);

  for (const auto& p : profile.data_) {
    std::vector<std::string> stackFrame;
    std::vector<boost::string_ref> reducedFrame;

    TranslateStack(symbols, fullnameToShortnameMap, p.first, stackFrame,
                   reducedFrame);
    stacks.emplace_back(std::move(stackFrame), p.second);
    reduced[std::move(reducedFrame)] += p.second;
  }
  ProfilerFlush();
}

// This function may modify the symbols
// the reduced lines's string is a view of translated stacks
void TranslateStack(
    Symbols& symbols,
    const std::unordered_map<std::string, std::string>& fullnameToShortnameMap,
    const std::vector<size_t>& addrs, std::vector<std::string>& stack,
    std::vector<boost::string_ref>& reduced) {
  std::unordered_set<std::string> seen = {""};
  SmallUint64Vector indexList;
  size_t index = 0;

  for (size_t i = 0, sz = addrs.size(); i < sz; ++i) {
    size_t a = addrs[i];
    // If the symbols does not exist, we add a address based symbol
    std::vector<std::string>& symList = symbols.data_[a];
    if (symList.empty()) {
      const std::string& aStr = (boost::format("%016x") % a).str();
      DLOG(INFO) << std::hex << "Address not find in symbols:" << a;
      symList = {aStr, "", aStr};
    }
    for (int j = static_cast<int>(symList.size() - 1); j >= 2; j -= 3) {
      std::string func = symList[j - 2];
      // const auto& fileLine = symList[j - 1];
      const auto& fullFunc = symList[j];

      auto it = fullnameToShortnameMap.find(fullFunc);
      if (it != fullnameToShortnameMap.end()) {
        func = it->second;
      }
      if (j > 2) {
        func.append(" (inline)");
      }

      /*
      if (!RegexMatch(func, boost::regex(R"Callback.*::Run$")).empty()) {
        std::string caller = (i > 0) ? addrs[i - 1] : 0;
        func = "Run#" + ShortIdFor(caller);
      }
      */

      // TODO: check the flag's mean
      if (FLAGS_functions) {
        if (func == "??") {
          stack.emplace_back((boost::format("%016x") % a).str());
        } else {
          stack.emplace_back(std::move(func));
        }
      }
      const auto& latest = stack.back();
      ++index;
      if (seen.count(latest)) {
        continue;
      }

      seen.insert(latest);
      indexList.emplace_back(index - 1);
    }
  }
  // Be carefull with string_ref and std::string
  // string is with sso, a short string's ptr
  // will change when it is moved in vector
  for (size_t i : indexList) {
    reduced.emplace_back(stack[i]);
  }
  DLOG(INFO) << "Translate addresses:" << std::hex << addrs
             << ", into:" << stack << std::endl
             << "The corresponding reduced stack:" << reduced;
}

void FillFullnameToshortnameMap(
    const Symbols& symbols,
    std::unordered_map<std::string, std::string>& fullnameToShortnameMap) {
  Marker m(__func__);
  ProfilerStart(__func__);

  std::unordered_map<boost::string_ref, boost::string_ref> shortnamesSeenOnce;
  std::unordered_set<boost::string_ref> shortNamesSeenMoreThanOnce;
  static const boost::regex kAddressPattern(R"(.*<[[:xdigit:]]+>$)");
  // TODO rewrite this function, we don't need to traverse the
  // symbols twice, at the first we meet a shot name, we trust it
  // to be the only one, when we find it's already meeted before,
  // we go back to amend it to be seen more than once function
  for (const auto& s : symbols.data_) {
    const auto& shortName = s.second[0];
    const auto& fullName = s.second[2];
    if (RegexMatch(fullName, kAddressPattern).empty()) {
      DLOG(INFO) << "Skip function full name not end with address:" << fullName;
      continue;
    }
    auto it = shortnamesSeenOnce.find(shortName);
    if (it != shortnamesSeenOnce.end() && it->second != fullName) {
      shortNamesSeenMoreThanOnce.insert(shortName);
    } else {
      shortnamesSeenOnce.emplace(shortName, fullName);
    }
  }

  // TODO: This can be faster
  static const boost::regex kAddressPattern2(R"(<0*([^>]*)>$)");
  for (const auto& s : symbols.data_) {
    const auto& shortName = s.second[0];
    const auto& fullName = s.second[2];

    if (fullnameToShortnameMap.count(fullName)) continue;
    if (shortNamesSeenMoreThanOnce.count(shortName)) {
      auto matchRes = RegexMatch(fullName, kAddressPattern2);
      if (!matchRes.empty()) {
        fullnameToShortnameMap.emplace(
            fullName, (boost::format("%s@%s") % shortName % matchRes[1]).str());
      }
    }
  }
  DLOG(INFO) << "Get full name to short name map:" << fullnameToShortnameMap;
  ProfilerFlush();
}

std::string ExtractSymbolLocation(const Symbols& symbols, size_t addr) {
  static const std::string kUnknown = "??:0:unknown";
  auto it = symbols.data_.find(addr);
  if (it == symbols.data_.end()) return kUnknown;

  std::string file = (it->second)[1];
  if (file == "?") file = "??:0";
  return (boost::format("%s:%016x") % file % addr).str();
}

std::unordered_map<std::string, size_t> ExtractCalls(const Symbols& symbols,
                                                     const Profile& profile) {
  Marker m(__func__);

  std::unordered_map<std::string, size_t> calls;
  for (const auto& p : profile.data_) {
    size_t count = p.second;
    const auto& addrs = p.first;
    // The string is long here, maybe we can replace with pointer
    auto destination = ExtractSymbolLocation(symbols, addrs[0]);
    calls.emplace(destination, count);

    for (auto it = std::next(addrs.begin()), itEnd = addrs.end(); it != itEnd;
         ++it) {
      const auto& source = ExtractSymbolLocation(symbols, *it);
      auto call = (boost::format("%s -> %s") % source % destination).str();
      calls.emplace(std::move(call), count);
      destination = source;
    }
  }
  DLOG(INFO) << "Extract these calls:" << calls;
  return calls;
}

// TODO: this function don't need actually, can be sumed up when insert
template <typename T>
size_t TotalProfile(const std::unordered_map<T, size_t>& data) {
  return std::accumulate(
      data.begin(), data.end(), 0ull,
      [](size_t sum, const auto& p) { return sum + p.second; });
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  if (FLAGS_program.empty() || FLAGS_profile.empty()) {
    std::cerr << "Wrong input" << std::endl;
    return 0;
  }

  Init();
  auto data = ReadProfile(FLAGS_program, FLAGS_profile);
  Symbols symbolMap;
  MergeSymbols(symbolMap, data.symbols_);
  Symbols symbol;
  if (FLAGS_useSymbolizedProfile) {
  } else if (FLAGS_useSymbolPage) {
  } else {
    symbol = ExtractSymbols(data.libs_, *data.pcs_);
  }

  // check opt_thread
  FilterAndPrint(*data.profile_, symbol, data.libs_, {});

  google::ShutdownGoogleLogging();

  return 0;
}
