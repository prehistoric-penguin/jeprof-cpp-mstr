#define GLOG_STL_LOGGING_FOR_UNORDERED
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <glog/stl_logging.h>
#include <sys/types.h>
#include <unistd.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/functional/hash.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/regex.hpp>
#include <cctype>
#include <csignal>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <iostream>
#include <functional>
#include <map>
#include <regex>
#include <string>
#include <vector>
#include <chrono>

DEFINE_string(program, "", "Name of execuable file");
DEFINE_string(profile, "", "Name of profile name");
DEFINE_string(dotFileName, "/tmp/jeprof_cpp.dot", "");
DEFINE_bool(useSymbolizedProfile, false, "TODO");
DEFINE_bool(useSymbolPage, false, "TODO");
DEFINE_bool(functions, true, "TODO");
DEFINE_double(nodeFraction, 0.005, "TODO");
DEFINE_double(edgeFraction, 0.001, "TODO");
DEFINE_int32(maxDegree, 8, "");
DEFINE_int32(nodeCount, 80, "TODO");
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

struct HashVector {
  size_t operator()(const std::vector<size_t>& vec) const {
    return boost::hash_value(vec);
  }
};

struct Profile {
  using ProfileMap = std::unordered_map<std::vector<size_t>, size_t, HashVector>;
   ProfileMap data_;
};

struct ThreadProfile {
  std::map<std::string, Profile> data_;
};

struct PCS {
  std::unordered_set<size_t> data_;
};

struct Symbols {
  std::map<size_t, std::vector<std::string>> data_;
};

struct LibraryEntry {
  std::string lib_;
  size_t start_;
  size_t finish_;
  size_t offset_;
};

struct Libraries {};

struct SymbolTable {
  std::map<std::string, std::vector<size_t>> data_;
};

struct Context {
  std::string vesion_;
  int period_;
  Profile profile_;
  ThreadProfile threads_;
  std::vector<LibraryEntry> libs_;
  PCS pcs_;
  Symbols symbols_;
};

class Marker {
 public:
  Marker(const char* f)
      : time_point_(std::chrono::steady_clock::now()), func_(f) {}

  ~Marker() {
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - time_point_).count() / 1000.0;
    std::cout << "time cost in " << func_ << ": " << diff << " ms" << std::endl;
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
                "Type missmatch, std::string or const char* require");

  std::string args[] = {i, all...};
  return boost::algorithm::join(args, " ");
}

bool IsValidSepAddress(size_t x) {
  return x != std::numeric_limits<size_t>::max();
}

boost::smatch RegexMatch(const std::string& target, const boost::regex& re);
void Usage();
std::string ReadProfileHeader(std::ifstream& ifs);
bool IsSymbolizedProfileFile();
bool IsProfileURL(const std::string&);
bool IsSymbolizedProfileFile(const std::string&);
void ConfigureObjTools();
std::string ConfigureTool(const std::string);
std::string FetchDynamicProfile(const std::string&, const std::string&, bool,
                                bool);
Context ReadProfile(const std::string&, const std::string&);
Context ReadThreadedHeapProfile(const std::string&, const std::string&,
                                const std::string&, std::ifstream&);
std::vector<std::string> ReadMappedLibraries(std::ifstream&);
std::vector<std::string> ReadMemoryMap(std::ifstream&);
std::vector<int> AdjustSamples(int, int, int, int, int, int);
std::vector<size_t> FixCallerAddresses(const std::string&);
size_t AddressSub(size_t, size_t);
void AddEntries(Profile&, PCS&, const std::vector<size_t>&, int);
void AddEntry(Profile, const std::vector<size_t>&, int);
std::vector<LibraryEntry> ParseLibraries(const std::vector<std::string>& map,
                                         PCS& pcs);
std::string FindLibrary(const std::string& lib);
std::string DebuggingLibrary(const std::string&);

std::tuple<size_t, size_t, size_t> ParseTextSectinoHeader(const std::string&);
void ParseTextSectionHeaderFromObjdump(const std::string&);
std::string ExecuteCommand(const std::string&);
bool System(const std::string&);

std::tuple<size_t, size_t, size_t> ParseTextSectinoHeaderFromObjdump(
    const std::string&);
size_t AddressAdd(size_t, size_t);
Symbols MergeSymbols(Symbols&, Symbols&);
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
void FilterAndPrint(Profile profile, const Symbols& symbols,
                    const std::vector<LibraryEntry>& libs,
                    const std::vector<std::string>& threads);
void RemoveUninterestingFrames(const Symbols& symbols, Profile& profile);
void FilterFrames(const Symbols& symbols, Profile& profile);
std::string ExtractSymbolLocation(const Symbols&, size_t);
std::map<std::string, size_t> ExtractCalls(const Symbols& symbols,
                                           const Profile& profile);
void FillFullnameToshortnameMap(
    const Symbols& symbols,
    std::map<std::string, std::string>& fullnameToShortnameMap);
// TODO, combine these two function into one
size_t TotalProfile(const Profile::ProfileMap& data) {
  return std::accumulate(
      data.begin(), data.end(), 0ull,
      [](size_t sum, const auto& p) { return sum + p.second; });
}
template <typename T>
size_t TotalProfile(const std::map<T, size_t>& profile);
std::map<std::string, size_t> ExtractCalls(const Symbols& symbols,
                                           const Profile& profile);
std::string ExtractSymbolLocation(const Symbols& symbols, size_t addr);
void FillFullnameToshortnameMap(
    const Symbols& symbols,
    std::map<std::string, std::string>& fullnameToShortnameMap);
std::vector<std::string> TranslateStack(
    const Symbols& symbols,
    const std::map<std::string, std::string>& fullnameToShortnameMap,
    const std::vector<size_t>& addrs);
std::map<std::vector<std::string>, size_t> ReduceProfile(
    const Symbols& symbols, const Profile& profile);
std::map<std::string, size_t> CumulativeProfile(
    const std::map<std::vector<std::string>, size_t>& profile);
std::map<std::string, size_t> FlatProfile(
    const std::map<std::vector<std::string>, size_t>& profile);
std::string Units();
double Unparse(size_t num);
std::string Percent(double num, double tot);
bool PrintDot(const std::string& prog, const Symbols& symbols, Profile& raw,
              const std::map<std::string, size_t>& flat,
              const std::map<std::string, size_t>& cumulative,
              size_t overallTotal);

bool IsProfileURL(const std::string& fname) {
  bool exists = boost::filesystem::exists(fname);
  LOG(INFO) << "File:" << fname << " exists:" << exists;

  return exists;
}

bool IsSymbolizedProfileFile(const std::string& fname) {
  if (!boost::filesystem::exists(fname)) return false;

  LOG(INFO) << "Reading file:" << FLAGS_program;
  std::ifstream ifs(FLAGS_program, std::ios::binary);
  auto firstLine = ReadProfileHeader(ifs);

  if (firstLine.empty()) return false;

  return true;
  // const std::string symbolPage = "m,[^/]+$,";
}

void ConfigureObjTools() {}

std::string ConfigureTool(const std::string& tool) { return tool; }

std::string ReadProfileHeader(std::ifstream& ifs) {
  uint8_t firstChar = ifs.peek();
  if (!std::isprint(firstChar)) return {};

  std::string line;
  while (std::getline(ifs, line)) {
    boost::erase_all(line, "\r");
    if (boost::starts_with(line, "%warn")) {
      LOG(INFO) << "WARNING:" << line;
    } else if (boost::starts_with(line, "%")) {
      LOG(INFO) << "Ignoring unknown command from profile header:" << line;
    } else {
      LOG(INFO) << "Get header line:" << line;
      return line;
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

  LOG(INFO) << "Read program:" << program << ", profile:" << profile;
  std::ifstream ifs(profile, std::ios::binary);
  std::string header = ReadProfileHeader(ifs);
  if (boost::starts_with(header,
                         (boost::format("--- %s") % symbolMarker).str())) {
    LOG(INFO) << "Meet symbol marker:" << header;
  }
  if (boost::starts_with(header,
                         (boost::format("--- %s") % heapMarker).str()) ||
      boost::starts_with(header,
                         (boost::format("--- %s") % growthMarker).str())) {
    LOG(INFO) << "Meet heap marker or growther marker:" << header;
  }

  // std::string profileType = "";
  // TODO regex here

  Context result;
  if (boost::starts_with(header, "heap")) {
    gProfileType = "heap";
    result = ReadThreadedHeapProfile(program, profile, header, ifs);
  }

  return result;
}

int HeapProfileIndex() { return 1; }

Context ReadThreadedHeapProfile(const std::string& program,
                                const std::string& profileName,
                                const std::string& header, std::ifstream& ifs) {
  Marker m(__func__);
  LOG(INFO) << "Now read threaded heap profile";
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

  Profile profile;
  ThreadProfile threadProfiles;
  PCS pcs;
  std::vector<std::string> map;
  std::string stack;

  std::string line;
  // TODO can be parallel here
  while (std::getline(ifs, line)) {
    boost::erase_all(line, "\r");
    if (boost::starts_with(line, "MAPPED_LIBRARIES:")) {
      LOG(INFO) << "Read mapped libraries:" << line;
      map = ReadMappedLibraries(ifs);
      break;
    }

    if (boost::starts_with(line, "--- Memory map:")) {
      map = ReadMemoryMap(ifs);
      break;
    }

    boost::trim(line);
    static const boost::regex kPattern1(R"(^@\s+(.*)$)");
    auto matchRes1 = RegexMatch(line, kPattern1);
    if (!matchRes1.empty()) {
      stack = matchRes1[1];
      continue;
    }
    static const boost::regex kPattern2(
        R"(^\s*(t(\*|\d+)):\s+(\d+):\s+(\d+)\s+\[\s*(\d+):\s+(\d+)\]$)");
    auto matchRes2 = RegexMatch(line, kPattern2);
    if (!matchRes2.empty()) {
      if (stack.empty()) {
        // Still in the header, so this is just a per-thread summary
        continue;
      }
      std::string thread = matchRes2[2];
      int n1 = std::stoi(matchRes2[3]);
      int s1 = std::stoi(matchRes2[4]);

      int n2 = std::stoi(matchRes2[5]);
      int s2 = std::stoi(matchRes2[6]);

      std::vector<int> counts =
          AdjustSamples(sampleAdjustment, samplingAlgorithm, n1, s1, n2, s2);
      if (thread == "*") {
        AddEntries(profile, pcs, FixCallerAddresses(stack), counts[index]);

      } else {
        AddEntries(threadProfiles.data_[thread], pcs, FixCallerAddresses(stack),
                   counts[index]);
      }
    }
  }
  Context context = {std::string("heap"),      1,   profile,  threadProfiles,
                     ParseLibraries(map, pcs), pcs, Symbols{}};
  LOG(INFO) << "Parsed profile:\n"
            << std::hex << profile.data_ << std::endl
            << "Parsed threadprofile:\n"
            << std::hex << threadProfiles.data_ << std::endl
            << "Parsed libraries:\n"
            << std::hex << context.libs_ << std::endl
            << "Parsed pcset:\n"
            << std::hex << pcs.data_ << std::endl;
  return context;
}

std::vector<size_t> FixCallerAddresses(const std::string& stack) {
  std::vector<std::string> addrs;
  boost::split(addrs, stack, isspace);

  std::vector<size_t> numAddrs;
  std::transform(
      std::begin(addrs), std::end(addrs), std::back_inserter(numAddrs),
      [](const std::string& s) { return std::stoull(s, nullptr, 16); });

  for (auto it = std::next(numAddrs.begin()), itEnd = numAddrs.end();
       it != itEnd; ++it) {
    *it = AddressSub(*it, 0x1);
  }

  LOG(INFO) << "Caller address from:" << stack << ", to:" << std::hex
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

void AddEntries(Profile& profile, PCS& pcs, const std::vector<size_t>& stack,
                int count) {
  LOG(INFO) << "Add entry for stack:" << std::hex << stack
            << ", with count:" << count;
  pcs.data_.insert(stack.begin(), stack.end());
  profile.data_[stack] += count;
  // AddEntry(profile, stack, count);
}

void AddEntry(Profile& profile, const std::vector<size_t>& stack,
              size_t count) {
  profile.data_[stack] += count;
}

std::vector<int> AdjustSamples(int sampleAdjustment, int samplingAlgorithm,
                               int n1, int s1, int n2, int s2) {
  if (sampleAdjustment) {
    if (samplingAlgorithm == 2) {
      auto adjust = [](int& s, int& n, int adjust) {
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
std::vector<std::string> ReadMappedLibraries(std::ifstream& ifs) {
  Marker m(__func__);

  std::vector<std::string> result;
  std::string line;
  while (std::getline(ifs, line)) {
    boost::erase_all(line, "\r");
    result.emplace_back(std::move(line));
  }
  LOG(INFO) << "Get mapped libraries section:" << result;
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
      LOG(INFO) << "Build variable:" << buildVar;
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

        LOG(INFO) << "matched case 1:" << line;
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

        LOG(INFO) << "matched case 2:" << line;
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

        LOG(INFO) << "matched case 3:" << line;
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
        LOG(INFO) << "matched case 4:" << line;
        match = true;
        break;
      }
    } while (0);

    if (!match) {
      LOG(INFO) << "line don't match any case:" << line;
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

    LOG(INFO) << "Add parsed library line, lib:" << lib << ",start:" << start
              << ", finish:" << finish << ",offset:" << offset;
    result.push_back({lib, start, finish, offset});
  }
  // Append special entry for additional library (not relocated)
  // FIXME
  size_t minPC = 0, maxPC = 0;
  maxPC = *std::max_element(std::begin(pcs.data_), std::end(pcs.data_));

  LOG(INFO) << "Add parsed library line, lib:" << programName
            << ",start:" << minPC << ", finish:" << maxPC << ",offset:" << 0;

  result.push_back({programName, minPC, maxPC, 0ul});
  return result;
}

std::tuple<size_t, size_t, size_t> ParseTextSectinoHeader(
    const std::string& lib) {
  // FIXME otool
  return ParseTextSectinoHeaderFromObjdump(lib);
}

std::string ExecuteCommand(const std::string& cmd) {
  auto log = std::string("execute command|" + cmd); 
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
  LOG(INFO) << "Execute command:" << cmd << ",Get result:\n" << result;
  return result;
}

// TODO This functino can use stream like method
// we don't need the whole file, just the header
std::tuple<size_t, size_t, size_t> ParseTextSectinoHeaderFromObjdump(
    const std::string& lib) {
  auto cmd = ShellEscape("objdump", "-h", lib);
  auto result = boost::algorithm::erase_all_copy(ExecuteCommand(cmd), "\r");

  std::vector<std::string> resultSet;
  boost::split(resultSet, result, [](char c) { return c == '\n'; });

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
  LOG(INFO) << "Text section header for lib:" << lib << ", size:" << size
            << ",vma:" << vma << ", fileOffset:" << fileOffset;
  return std::make_tuple(size, vma, fileOffset);
}

std::string DebuggingLibrary(const std::string& file) {
  // Carefull for multhread
  static std::map<std::string, std::string> cache;
  auto it = cache.find(file);
  if (it != cache.end()) {
    return it->second;
  }

  if (boost::starts_with(file, "/")) {
    std::array<std::string, 2> debugFiles = {
        (boost::format("/usr/lib/debug%s") % file).str(),
        (boost::format("/usr/lib/debug%s.debug") % file).str()};
    for (const auto& f : debugFiles) {
      if (boost::filesystem::exists(f)) {
        LOG(INFO) << "Find debugging lib:" << f << ", for file:" << file;
        cache.emplace(file, f);
        return f;
      }
    }
  }
  LOG(INFO) << "Haven't find debugging lib for file:" << file;
  cache.emplace(file, std::string{});
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

std::vector<std::string> ReadMemoryMap(std::ifstream& ifs) {
  std::vector<std::string> result;
  std::string buildVar;
  std::string line;

  static const boost::regex kBuildNumberPattern(R"(^\s*build=(.*))");
  while (std::getline(ifs, line)) {
    boost::erase_all(line, "\r");
    auto matchRes = RegexMatch(line, kBuildNumberPattern);
    if (!matchRes.empty()) buildVar = matchRes[1];

    boost::regex_replace(line, boost::regex(R"(\$build\b)"), buildVar);
    result.emplace_back(line);
  }
  return result;
}

size_t AddressAdd(size_t x, size_t y) { return x + y; }

Symbols MergeSymbols(Symbols& lhs, Symbols& rhs) {
  Symbols result = lhs;
  result.data_.insert(std::begin(rhs.data_), std::end(rhs.data_));

  return result;
}

// TODO: smallvector smallstring replace
Symbols ExtractSymbols(const std::vector<LibraryEntry>& libs,
                       const PCS& pcSet) {
  Marker m(__func__);

  Symbols symbols;
  auto sortedLibs = libs;
  // consider use reference_wrapper
  std::sort(
      std::begin(sortedLibs), std::end(sortedLibs),
      [](const auto& lhs, const auto& rhs) { return rhs.start_ < lhs.start_; });

  std::vector<size_t> pcs;
  std::transform(pcSet.data_.begin(), pcSet.data_.end(),
                 std::back_inserter(pcs),
                 [](const auto& p) { return p; });
  std::sort(pcs.begin(), pcs.end());
  for (const auto& entry : sortedLibs) {
    auto libName = entry.lib_;
    const auto debugLib = DebuggingLibrary(libName);
    if (!debugLib.empty()) libName = debugLib;

    size_t startPCIndex, finishPCIndex;
    // TODO lowbound or upperbound
    for (finishPCIndex = pcs.size(); finishPCIndex > 0; --finishPCIndex) {
      if (pcs[finishPCIndex - 1] <= entry.finish_) break;
    }
    for (startPCIndex = finishPCIndex; startPCIndex > 0; --startPCIndex) {
      if (pcs[startPCIndex - 1] < entry.start_) break;
    }

    std::vector<size_t> contained{std::next(pcs.begin(), startPCIndex),
                                  std::next(pcs.begin(), finishPCIndex)};
    pcs.erase(std::next(pcs.begin(), startPCIndex),
              std::next(pcs.begin(), finishPCIndex));

    LOG(INFO) << "Start to extract symbols for lib:" << libName
              << ", get contained pc set:" << std::hex << contained;
    MapToSymbols(libName, AddressSub(entry.start_, entry.offset_), contained,
                 symbols);
  }

  return symbols;
}

void MapToSymbols(const std::string& image, size_t offset,
                  const std::vector<size_t>& pcList, Symbols& symbols) {
  Marker m(__func__);

  if (pcList.empty()) return;

  std::string cmd = ShellEscape(kAddr2Line, "-f", "-C", "-e", image);
  if (!System(ShellEscape(kAddr2Line, "--help", ">/dev/null 2>&1"))) {
    LOG(INFO) << "addr2line is not installed on system, use nm";
    MapSymbolsWithNM(image, offset, pcList, symbols);
    return;
  }

  Symbols nmSymbols;
  size_t sepAddress = std::numeric_limits<size_t>::max();
  MapSymbolsWithNM(image, offset, pcList, nmSymbols, &sepAddress);
  if (IsValidSepAddress(sepAddress)) {
    auto fullCmd = (boost::format("%s -i --help >/dev/null 2>&1") % cmd).str();
    if (System(fullCmd)) {
      LOG(INFO) << "addr2line support '-i' options check pass";
      cmd += " -i";
    } else {
      sepAddress = std::numeric_limits<size_t>::max();
    }
  }
  const std::string tmpFileSym =
      (boost::format("/tmp/jeprof_cpp%d.sym") % getpid()).str();
  LOG(INFO) << "Write data into tmp sys file:" << tmpFileSym
            << ", with content:" << std::endl
            << std::hex << pcList;

  std::ofstream ofs(tmpFileSym);
  LOG(INFO) << "sepaddress is:" << IsValidSepAddress(sepAddress);
  auto toHexStr = [](size_t x) { return (boost::format("%016x") % x).str(); };
  for (auto pc : pcList) {
    ofs << toHexStr(AddressSub(pc, offset)) << std::endl;
    if (IsValidSepAddress(sepAddress)) ofs << toHexStr(sepAddress) << std::endl;
  }
  ofs.close();

  // TODO review the original source the command end with |
  const std::string cmdWithTmpFile =
      (boost::format("%s <%s") % cmd % tmpFileSym).str();
  auto result = ExecuteCommand(cmdWithTmpFile);
  std::vector<std::string> resultSet;
  boost::split(resultSet, result, [](char c) { return c == '\n'; });

  LOG(INFO) << "Total lines for command result:" << resultSet.size();
  size_t count = 0;
  for (size_t i = 0, resultSize = resultSet.size(); i < resultSize;) {
    auto line = resultSet[i++];
    boost::erase_all(line, "\r");
    std::string fullFunction = line;

    if (i >= resultSize) {
      LOG(INFO) << "WARN: this result size is not even:" << resultSize;
      break;
    }
    line = resultSet[i++];
    boost::erase_all(line, "\r");
    std::string fileLineNum = line;

    if (IsValidSepAddress(sepAddress) && (fullFunction == kSepSymbol)) {
      LOG(INFO) << "Meet sepsymbol for pcstr:" << std::hex << pcList[count];
      ++count;
      continue;
    }
    LOG(INFO) << "Now deal with fullfunction:" << fullFunction
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
      std::vector<std::string>& sym = symbols.data_[pc];
      sym.insert(sym.begin(), {function, fileLineNum, fullFunction});
      LOG(INFO) << (boost::format("cur symbol line:%016x => %s") % pc %
                    boost::join(sym, " "))
                       .str();
      if (!IsValidSepAddress(sepAddress)) ++count;
    }
  }
  LOG(INFO) << "Now the symbol is:" << std::hex << symbols.data_;
}

bool MapSymbolsWithNM(const std::string& image, size_t offset,
                      const std::vector<size_t>& pcList, Symbols& symbols,
                      size_t* sepAddress) {
  Marker m(__func__);

  LOG(INFO) << "Start map symbols for image:" << image
            << ", with offset:" << std::hex << offset;
  auto symbolTable = GetProcedureBoundaries(image, ".", sepAddress);
  LOG(INFO) << "Get symbol table:" << symbolTable.data_;
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
  LOG(INFO) << "Sorted names with value in symboltable:" << names;
  size_t index = 0;
  auto fullName = names[0];
  auto name = ShortFunctionName(fullName);
  auto nameNum = names.size();

  auto sortedList = pcList;
  std::sort(sortedList.begin(), sortedList.end());

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
std::string ShortFunctionName(const std::string& fullName) {
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

  return replace(name, R"(^.*\s+(\w+::))", "$&");
}

SymbolTable GetProcedureBoundaries(const std::string& image,
                                   const std::string& regex,
                                   size_t* sepAddress) {
  Marker m(__func__);

  if (image.find_first_of("/.") != 0) {
    LOG(ERROR) << "Error file name, not start with . or /:" << image;
    return {};
  }

  std::string imageName = image;
  const auto debugging = DebuggingLibrary(image);
  if (!debugging.empty()) imageName = debugging;

  std::string demangleFlag, cppfiltFlag;
  std::string toDevNull = ">/dev/null 2>&1";

  // This line seems a bug in the perl source code, image -> $image
  if (System(ShellEscape(kNm, "--demangle", "image") + toDevNull)) {
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
      std::string(ShellEscape(kNm, "-n", flattenFlag, demangleFlag, image) +
                  tail),
      std::string(
          ShellEscape(kNm, "-D", "-n", flattenFlag, demangleFlag, image) +
          tail)};

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
  auto cmdResult = ExecuteCommand(cmd);
  std::string routine;

  auto CheckAddSymbol = [&table, &regex](const std::string& name,
                                         const auto& start, const auto& last) {
    if (name.empty()) return;
    if (regex.empty() || regex == "." ||
        !RegexMatch(name, boost::regex(regex)).empty()) {
      size_t startVal = std::stoull(start, nullptr, 16);
      size_t lastVal = std::stoull(last, nullptr, 16);

      LOG(INFO) << "Add line into symbol table, name:" << name << std::hex
                << ", start:" << startVal << ",last:" << lastVal;
      table.data_.emplace(name, std::vector<size_t>{startVal, lastVal});
    }
  };
  std::vector<std::string> resultSet;
  boost::split(resultSet, cmdResult, [](char c) { return c == '\n'; });
  std::string lastStart = "0";
  for (auto line : resultSet) {
    boost::erase_all(line, "\r");
    static const boost::regex kSymbolPattern(R"(^\s*([0-9a-f]+) (.) (..*))");
    auto matchRes = RegexMatch(line, kSymbolPattern);
    if (!matchRes.empty()) {
      // LOG(INFO) << "Line matched:" << line;
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
      LOG(INFO) << "Use Image:";
    } else if (boost::starts_with(line, "PDB file name:")) {
      // For windows;
      LOG(INFO) << "Use PDB:";
    }
  }
  // Deal with last routine
  CheckAddSymbol(routine, lastStart, lastStart);
  return table;
}

bool System(const std::string& cmd) { return system(cmd.data()) == 0; }

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

  const std::string skipRegexPattern = "NOMATCH";

  // TODO seems can be replaced with set
  std::set<std::string> skip;
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
  Profile::ProfileMap result;
  for (const auto& p : profile.data_) {
    size_t count = p.second;
    std::vector<size_t> path;
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
    result[path] += count;
    // AddEntry(result, path, count);
  }
  LOG(INFO) << "Result after remove uninteresting frames:" << std::hex
            << result;
  // TODO add filter
  // FilterFrames(symbols, result);
  profile.data_ = result;
}

void FilterFrames(const Symbols& symbols, Profile& profile) {
  // if opt_retain opt_exclude
  return;
  // TODO add remain logic
}

void FilterAndPrint(Profile profile, const Symbols& symbols,
                    const std::vector<LibraryEntry>& libs,
                    const std::vector<std::string>& threads) {
  Marker m(__func__);

  auto total = TotalProfile(profile.data_);
  LOG(INFO) << "Total Profile:" << std::hex << total;
  RemoveUninterestingFrames(symbols, profile);
  LOG(INFO) << "After remove unteresting frames:" << std::hex << profile.data_;
  auto calls = ExtractCalls(symbols, profile);
  LOG(INFO) << "Extracted calls:" << std::hex << calls;
  auto reduced = ReduceProfile(symbols, profile);
  LOG(INFO) << "The reduced profile:" << std::hex << reduced;
  auto flat = FlatProfile(reduced);
  LOG(INFO) << "The flat profile:" << std::hex << flat;
  auto cumulative = CumulativeProfile(reduced);
  LOG(INFO) << "The cumulative profile:" << std::hex << cumulative;

  PrintDot(FLAGS_program, symbols, profile, flat, cumulative, total);
}

bool PrintDot(const std::string& prog, const Symbols& symbols, Profile& raw,
              const std::map<std::string, size_t>& flatMap,
              const std::map<std::string, size_t>& cumulativeMap,
              size_t overallTotal) {
  Marker m(__func__);

  auto flat = flatMap;
  auto cumulative = cumulativeMap;
  auto localTotal = TotalProfile(flat);
  size_t nodeLimit = FLAGS_nodeFraction * localTotal;
  size_t edgeLimit = FLAGS_edgeFraction * localTotal;
  size_t nodeCount = FLAGS_nodeCount;
  std::vector<std::string> list;
  std::transform(cumulative.begin(), cumulative.end(), std::back_inserter(list),
                 [](const auto& v) { return v.first; });
  std::sort(list.begin(), list.end(),
            [&cumulative](const std::string& lhs, const std::string& rhs) {
              auto lv = cumulative[lhs];
              auto rv = cumulative[rhs];
              return lv != rv ? rv < lv : lhs < rhs;
            });
  auto last = std::min(nodeCount - 1, list.size() - 1);
  while (last >= 0 && cumulative[list[last]] <= nodeLimit) --last;

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

  std::map<std::string, size_t> node;
  size_t nextNode = 1;
  for (auto it = list.begin(), itEnd = std::next(list.begin(), last + 1);
       it != itEnd; ++it) {
    const auto& a = *it;
    auto f = flat[a];
    auto c = cumulative[a];

    double fs = 8;
    if (localTotal > 0) {
      fs = 8 + (50.0 * sqrt(fabs(f * 1.0 / localTotal)));
    }

    node[a] = nextNode++;
    auto sym = a;

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
      LOG(FATAL) << "Not implement";
    }
    ofs << boost::format(
               "N%d [label=\"%s\\n%s (%s)%s\\r"
               "\",shape=box,fontsize=%.1f%s];\n") %
               node[a] % sym % Unparse(f) % Percent(f, localTotal) % extra %
               fs % style;
  }
  using Edge = std::map<std::array<std::string, 2>, size_t>;
  Edge edge;
  size_t n = 0;
  std::map<std::string, std::string> fullnameToshortnameMap;
  // TODO, this function is called twice
  FillFullnameToshortnameMap(symbols, fullnameToshortnameMap);

  for (const auto& r : raw.data_) {
    n = r.second;
    auto translated = TranslateStack(symbols, fullnameToshortnameMap, r.first);
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

  std::map<std::string, size_t> outDegree, inDegree;
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
  return true;
}

std::string Percent(double num, double tot) {
  if (tot != 0) {
    return (boost::format("%.1f%%") % (num * 100.0 / tot)).str();
  }
  return num == 0 ? "nan" : ((num > 0) ? "+inf" : "-inf");
}

double Unparse(size_t num) {
  if (FLAGS_showBytes) {
    return num;
  }
  return 1.0 * num / (1024 * 1024);
}

std::string Units() {
  if (FLAGS_showBytes) {
    return "B";
  }
  return "MB";
}

std::map<std::string, size_t> FlatProfile(
    const std::map<std::vector<std::string>, size_t>& profile) {
  std::map<std::string, size_t> result;
  Marker m(__func__);

  for (const auto& p : profile) result[p.first.front()] += p.second;
  return result;
}

std::map<std::string, size_t> CumulativeProfile(
    const std::map<std::vector<std::string>, size_t>& profile) {
  Marker m(__func__);

  std::map<std::string, size_t> result;
  for (const auto& p : profile) {
    for (const auto& a : p.first) {
      result[a] += p.second;
    }
  }
  return result;
}

std::map<std::vector<std::string>, size_t> ReduceProfile(
    const Symbols& symbols, const Profile& profile) {
  Marker m(__func__);

  std::map<std::vector<std::string>, size_t> result;
  std::map<std::string, std::string> fullnameToShortnameMap;
  FillFullnameToshortnameMap(symbols, fullnameToShortnameMap);

  LOG(INFO) << "Get fullname to shortname map:" << fullnameToShortnameMap;
  for (const auto& p : profile.data_) {
    auto count = p.second;
    auto translated = TranslateStack(symbols, fullnameToShortnameMap, p.first);
    std::vector<std::string> path;
    std::set<std::string> seen = {""};
    for (const auto& e : translated) {
      if (seen.count(e)) continue;

      seen.insert(e);
      path.emplace_back(e);
    }
    LOG(INFO) << "Reduce line:" << translated << ", into:" << path;
    result[path] += count;
  }
  return result;
}

std::vector<std::string> TranslateStack(
    const Symbols& symbols,
    const std::map<std::string, std::string>& fullnameToShortnameMap,
    const std::vector<size_t>& addrs) {

  std::vector<std::string> result;
  for (size_t i = 0, sz = addrs.size(); i < sz; ++i) {
    size_t a = addrs[i];
    // if opt_disasm opt_list
    std::vector<std::string> symList;
    // TODO, conbine it define into if
    auto it = symbols.data_.find(a);
    if (it == symbols.data_.end()) {
      LOG(INFO) << std::hex << "Address not find in symbols:" << a;
      const std::string& aStr = (boost::format("%016x") % a).str();
      symList = {aStr, "", aStr};
    } else {
      symList = it->second;
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

      if (FLAGS_functions) {
        if (func == "??") {
          result.emplace_back((boost::format("%016x") % a).str());
        } else {
          result.emplace_back(func);
        }
      }
    }
  }
  LOG(INFO) << "Translate addresses:" << std::hex << addrs
            << ", into:" << result;
  return result;
}

void FillFullnameToshortnameMap(
    const Symbols& symbols,
    std::map<std::string, std::string>& fullnameToShortnameMap) {
  std::map<std::string, std::string> shortnamesSeenOnce;
  std::set<std::string> shortNamesSeenMoreThanOnce;
  static const boost::regex kAddressPattern(R"(.*<[[:xdigit:]]+>$)");
  for (const auto& s : symbols.data_) {
    const auto& shortName = s.second[0];
    const auto& fullName = s.second[2];
    if (RegexMatch(fullName, kAddressPattern).empty()) {
      LOG(INFO) << "Skip function full name not end with address:" << fullName;
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
  LOG(INFO) << "Get full name to short name map:" << fullnameToShortnameMap;
}

std::string ExtractSymbolLocation(const Symbols& symbols, size_t addr) {
  auto it = symbols.data_.find(addr);
  if (it == symbols.data_.end()) return "??:0:unknown";

  std::string file = (it->second)[1];
  if (file == "?") file = "??:0";
  return (boost::format("%s:%016x") % file % addr).str();
}

std::map<std::string, size_t> ExtractCalls(const Symbols& symbols,
                                           const Profile& profile) {
  Marker m(__func__);

  std::map<std::string, size_t> calls;
  for (const auto& p : profile.data_) {
    size_t count = p.second;
    const auto& addrs = p.first;
    // The string is long here, maybe we can replace with pointer
    auto destination = ExtractSymbolLocation(symbols, addrs[0]);
    calls.emplace(destination, count);

    for (auto it = std::next(addrs.begin()), itEnd = addrs.end(); it != itEnd;
         ++it) {
      const auto& source = ExtractSymbolLocation(symbols, *it);
      const auto& call =
          (boost::format("%s -> %s") % source % destination).str();
      calls.emplace(call, count);
      destination = source;
    }
  }
  LOG(INFO) << "Extract these calls:" << calls;
  return calls;
}

// TODO: this function don't need actually, can be sumed up when insert
template <typename T>
size_t TotalProfile(const std::map<T, size_t>& data) {
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
  symbolMap = MergeSymbols(symbolMap, data.symbols_);
  // std::map<> sourceCache;
  // std::string profileName = "a.out";
  // std::string fileName = "a.out";
  Symbols symbol;
  if (FLAGS_useSymbolizedProfile) {
  } else if (FLAGS_useSymbolPage) {
  } else {
    symbol = ExtractSymbols(data.libs_, data.pcs_);
  }

  // check opt_thread
  FilterAndPrint(data.profile_, symbol, data.libs_, {});

  /*
  if (!data.threads_.data_.empty()) {
    for (const auto& t :data.threads_.data_) {
    }
  }
  */
  google::ShutdownGoogleLogging();

  return 0;
}
